#![allow(dead_code, unused_variables)]

mod context;
pub mod dbsp;
pub mod error;
pub mod expr;
mod function;
mod interpreter;
pub mod operator;
mod operators;
pub mod relation;
mod resolver;
mod scalar;
pub mod stmt;
mod util;
pub mod variable;

use context::{InterpreterContext, ProgramContext, ResolverContext};
use error::IncLogError;
use interpreter::Interpreter;
use resolver::Resolver;
use stmt::{Code, Stmt};
use variable::Value;

// Var: Variable
// Val: Value
// Stmt: Statement
// Expr: Expression
// Ident: Identifier

#[derive(Default)]
pub struct IncLog {
    program_context: ProgramContext,
    had_syntax_err: bool,
    had_runtime_err: bool,
}

impl IncLog {
    pub fn new() -> Self {
        Self {
            program_context: ProgramContext::new(),
            had_syntax_err: false,
            had_runtime_err: false,
        }
    }
    pub fn run_and_print(&mut self, source: String) {
        match self.run(source) {
            Ok(Some(val)) => println!("{}", val),
            Ok(None) => (),
            Err(err) => eprintln!("{}", err),
        }
    }
    pub fn run(&mut self, source: String) -> Result<Option<Value>, IncLogError> {
        self.parse(source).and_then(|stmts| self.execute(stmts))
    }
    pub fn parse(&mut self, source: String) -> Result<Code, IncLogError> {
        // Should actually parse the input string and create an expression
        // or a list of statements.
        todo!()
    }
    pub fn execute(
        &mut self,
        code: impl IntoIterator<Item = Stmt>,
    ) -> Result<Option<Value>, IncLogError> {
        self.program_context
            .program
            .extend_program(code.into_iter());
        let mut resolver_ctx = ResolverContext::new(&mut self.program_context.scopes);
        Resolver::new()
            .resolve(
                self.program_context.program.unexecuted_code_mut(),
                &mut resolver_ctx,
            )
            .map_err(|err| self.ack_syntax_err(err))
            .and_then(|()| {
                let mut interpreter_ctx =
                    InterpreterContext::new(&mut self.program_context.environment);
                Interpreter::new()
                    .interpret(
                        self.program_context.program.unexecuted_code(),
                        &mut interpreter_ctx,
                    )
                    .map_err(|err| self.ack_runtime_err(err))
            })
    }
    fn ack_syntax_err<T: Into<IncLogError>>(&mut self, err: T) -> IncLogError {
        self.had_syntax_err = true;
        err.into()
    }
    fn ack_runtime_err<T: Into<IncLogError>>(&mut self, err: T) -> IncLogError {
        self.had_runtime_err = true;
        err.into()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        dbsp::{DbspInput, DbspInputs, DbspOutput},
        expr::{
            AliasExpr, CartesianProductExpr, DifferenceExpr, DistinctExpr, EquiJoinExpr,
            FixedPointIterExpr, ProjectionExpr, SelectionExpr, UnionExpr,
        },
        relation::{RelationSchema, TupleKey, TupleValue},
        scalar::ScalarTypedValue,
        stmt::BlockStmt,
    };
    use ::dbsp::{RootCircuit, zset};
    use expr::{AssignExpr, BinaryExpr, CallExpr, Expr, Literal, LiteralExpr, VarExpr};
    use operator::Operator;
    use stmt::{ExprStmt, Stmt, VarStmt};

    #[test]
    fn test_variable_init_assign() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let initialization = vec![Stmt::from(VarStmt {
            name: "a".to_string(),
            initializer: Some(Expr::from(LiteralExpr {
                value: Literal::Uint(1),
            })),
        })];

        let assignment = vec![Stmt::from(ExprStmt {
            expr: Expr::from(AssignExpr::new(
                "a",
                Expr::from(LiteralExpr {
                    value: Literal::Uint(2),
                }),
            )),
        })];

        assert_eq!(inclog.execute(initialization)?.unwrap(), Value::Uint(1));

        assert_eq!(inclog.execute(assignment)?.unwrap(), Value::Uint(2));

        Ok(())
    }

    // A function with two parameters which adds two values.
    fn new_add_function_expr() -> Expr {
        Expr::from(expr::FunctionExpr {
            parameters: vec!["a".to_string(), "b".to_string()],
            body: BlockStmt {
                stmts: vec![Stmt::from(ExprStmt {
                    expr: Expr::from(BinaryExpr {
                        operator: Operator::Addition,
                        left: Expr::from(VarExpr::new("a")),
                        right: Expr::from(VarExpr::new("b")),
                    }),
                })],
            },
        })
    }

    #[test]
    fn test_function_declarations() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let anonymous_function = vec![Stmt::from(ExprStmt {
            expr: new_add_function_expr(),
        })];

        let named_function = vec![Stmt::from(VarStmt {
            name: "add".to_string(),
            initializer: Some(new_add_function_expr()),
        })];

        let result = inclog.execute(anonymous_function)?.unwrap();
        assert_eq!(format!("{}", result), "<anonymous fn(a, b)>");

        let result = inclog.execute(named_function)?.unwrap();
        assert_eq!(format!("{}", result), "<fn add(a, b)>");

        Ok(())
    }

    #[test]
    fn test_function_call() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let function_call = vec![
            Stmt::from(VarStmt {
                name: "add".to_string(),
                initializer: Some(new_add_function_expr()),
            }),
            Stmt::from(ExprStmt {
                expr: Expr::from(CallExpr {
                    callee: Expr::from(VarExpr::new("add")),
                    arguments: vec![
                        Expr::from(LiteralExpr {
                            value: Literal::Uint(1),
                        }),
                        Expr::from(LiteralExpr {
                            value: Literal::Uint(2),
                        }),
                    ],
                }),
            }),
        ];

        let result = inclog.execute(function_call)?.unwrap();
        assert_eq!(Value::Uint(3), result);

        Ok(())
    }

    fn new_selection_expr(
        root_circuit: &mut RootCircuit,
    ) -> Result<(DbspInputs, DbspOutput), anyhow::Error> {
        let mut dbsp_inputs = DbspInputs::default();

        let code = [
            Stmt::from(VarStmt {
                name: "add".to_string(),
                initializer: Some(new_add_function_expr()),
            }),
            Stmt::from(VarStmt {
                name: "constant".to_string(),
                initializer: Some(Expr::from(LiteralExpr {
                    value: Literal::Uint(1),
                })),
            }),
            Stmt::from(VarStmt {
                name: "selected".to_string(),
                initializer: Some(Expr::from(SelectionExpr {
                    condition: Expr::from(BinaryExpr {
                        operator: Operator::GreaterEqual,
                        // TODO: Try more complex logical expression with and/or.
                        left: Expr::from(VarExpr::new("weight")),
                        right: Expr::from(CallExpr {
                            callee: Expr::from(VarExpr::new("add")),
                            arguments: vec![
                                Expr::from(VarExpr::new("constant")),
                                Expr::from(LiteralExpr {
                                    value: Literal::Uint(1),
                                }),
                            ],
                        }),
                    }),
                    relation: Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "edges",
                            ["from", "to", "weight", "active"],
                            ["from", "to"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    )),
                })),
            }),
            Stmt::from(VarStmt {
                name: "projected".to_string(),
                initializer: Some(Expr::from(ProjectionExpr {
                    relation: Expr::from(VarExpr::new("selected")),
                    attributes: ["from", "to", "weight"]
                        .into_iter()
                        .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                        .chain([(
                            // Here we create an entirely new column.
                            "product_from_to".to_string(),
                            Expr::from(BinaryExpr {
                                operator: Operator::Multiplication,
                                left: Expr::from(VarExpr::new("from")),
                                right: Expr::from(VarExpr::new("to")),
                            }),
                        )])
                        .collect(),
                })),
            }),
        ];

        match IncLog::new().execute(code) {
            Ok(Some(Value::Relation(relation))) => {
                let relation = relation.borrow();
                let output_handle = relation.inner.output();
                let output_schema = relation.schema.clone();
                Ok((dbsp_inputs, DbspOutput::new(output_schema, output_handle)))
            }
            result => panic!("Expected a relation, got {:?}", result),
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct Edge {
        from: u64,
        to: u64,
        weight: u64,
        active: bool,
    }

    impl Edge {
        fn new(from: u64, to: u64, weight: u64) -> Self {
            Self {
                from,
                to,
                weight,
                active: true,
            }
        }
    }

    impl From<Edge> for TupleKey {
        fn from(edge: Edge) -> Self {
            TupleKey {
                data: vec![
                    ScalarTypedValue::Uint(edge.from),
                    ScalarTypedValue::Uint(edge.to),
                ],
            }
        }
    }

    impl From<Edge> for TupleValue {
        fn from(edge: Edge) -> Self {
            TupleValue {
                data: vec![
                    ScalarTypedValue::Uint(edge.from),
                    ScalarTypedValue::Uint(edge.to),
                    ScalarTypedValue::Uint(edge.weight),
                    ScalarTypedValue::Bool(edge.active),
                ],
            }
        }
    }

    #[test]
    fn test_selection_and_projection() -> Result<(), anyhow::Error> {
        // TODO: test multithreaded runtime
        let (circuit, (inputs, output)) = RootCircuit::build(new_selection_expr)?;
        let edges_input = inputs.get("edges").unwrap();

        let data1 = [Edge::new(0, 1, 1), Edge::new(1, 2, 2), Edge::new(2, 3, 3)];

        let data2 = [Edge::new(3, 4, 1), Edge::new(4, 5, 2), Edge::new(5, 6, 3)];

        println!("Insert of data1:");

        edges_input.insert_with_same_weight(data1.iter(), 2);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(1_u64, 2_u64, 2_u64, 2_u64) => 2,
                tuple!(2_u64, 3_u64, 3_u64, 6_u64) => 2,
            }
        );

        println!("Insert of data2:");

        edges_input.insert_with_same_weight(data2.iter(), 1);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(4_u64, 5_u64, 2_u64, 20_u64) => 1,
                tuple!(5_u64, 6_u64, 3_u64, 30_u64) => 1,
            }
        );

        println!("Removal of data1:");

        edges_input.insert_with_same_weight(data1.iter(), -1);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(1_u64, 2_u64, 2_u64, 2_u64) => -1,
                tuple!(2_u64, 3_u64, 3_u64, 6_u64) => -1,
            }
        );

        Ok(())
    }

    #[derive(Clone, Debug)]
    struct Person {
        person_id: u64,
        name: String,
        age: u64,
        profession_id: u64,
    }

    impl From<Person> for TupleKey {
        fn from(person: Person) -> Self {
            TupleKey {
                data: vec![ScalarTypedValue::Uint(person.person_id)],
            }
        }
    }

    impl From<Person> for TupleValue {
        fn from(person: Person) -> Self {
            TupleValue {
                data: vec![
                    ScalarTypedValue::Uint(person.person_id),
                    ScalarTypedValue::String(person.name),
                    ScalarTypedValue::Uint(person.age),
                    ScalarTypedValue::Uint(person.profession_id),
                ],
            }
        }
    }

    #[derive(Clone, Debug)]
    struct Profession {
        profession_id: u64,
        name: String,
    }

    impl From<Profession> for TupleKey {
        fn from(profession: Profession) -> Self {
            TupleKey {
                data: vec![ScalarTypedValue::Uint(profession.profession_id)],
            }
        }
    }

    impl From<Profession> for TupleValue {
        fn from(profession: Profession) -> Self {
            TupleValue {
                data: vec![
                    ScalarTypedValue::Uint(profession.profession_id),
                    ScalarTypedValue::String(profession.name),
                ],
            }
        }
    }

    fn persons_and_professions_data() -> (Vec<Person>, Vec<Profession>) {
        let persons: Vec<Person> = vec![
            Person {
                person_id: 0,
                name: "Alice".to_string(),
                age: 20,
                profession_id: 0,
            },
            Person {
                person_id: 1,
                name: "Bob".to_string(),
                age: 30,
                profession_id: 1,
            },
            Person {
                person_id: 2,
                name: "Charlie".to_string(),
                age: 40,
                profession_id: 0,
            },
        ];

        let professions: Vec<Profession> = vec![
            Profession {
                profession_id: 0,
                name: "Engineer".to_string(),
            },
            Profession {
                profession_id: 1,
                name: "Doctor".to_string(),
            },
        ];
        (persons, professions)
    }

    #[test]
    fn test_standard_join() -> Result<(), anyhow::Error> {
        let (circuit, (inputs, output)) = RootCircuit::build(|root_circuit| {
            let mut dbsp_inputs = DbspInputs::default();

            let code = [
                Stmt::from(VarStmt {
                    name: "person".to_string(),
                    initializer: Some(Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "person",
                            ["person_id", "name", "age", "profession_id"],
                            ["person_id"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }),
                Stmt::from(VarStmt {
                    name: "profession".to_string(),
                    initializer: Some(Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "profession",
                            ["profession_id", "name"],
                            ["profession_id"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }),
                Stmt::from(VarStmt {
                    name: "joined".to_string(),
                    initializer: Some(Expr::from(EquiJoinExpr {
                        left: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("person")),
                            alias: "pers".to_string(),
                        }),
                        right: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("profession")),
                            alias: "prof".to_string(),
                        }),
                        // TODO: Shall we force aliasing here? Technically, it isn't
                        // required because the left attribute only operates on the left relation
                        // and the right attribute only operates on the right relation.
                        // Also, shall expressions be allowed here?
                        on: vec![(
                            Expr::from(VarExpr::new("profession_id")),
                            Expr::from(VarExpr::new("profession_id")),
                        )],
                        // attributes: None,
                        attributes: Some(
                            // Here, we filter out the duplicated profession_id column.
                            [
                                ("person_id", "pers.person_id"),
                                ("person_name", "pers.name"),
                                ("age", "pers.age"),
                                ("profession_id", "prof.profession_id"),
                                ("profession_name", "prof.name"),
                            ]
                            .into_iter()
                            .map(|(name, identifier)| {
                                (name.to_string(), Expr::from(VarExpr::new(identifier)))
                            })
                            .collect(),
                        ),
                    })),
                }),
            ];

            match IncLog::new().execute(code) {
                Ok(Some(Value::Relation(relation))) => {
                    let relation = relation.borrow();
                    let output_handle = relation.inner.output();
                    let output_schema = relation.schema.clone();
                    Ok((dbsp_inputs, DbspOutput::new(output_schema, output_handle)))
                }
                result => panic!("Expected a relation, got {:?}", result),
            }
        })?;
        let person_input = inputs.get("person").unwrap();
        let profession_input = inputs.get("profession").unwrap();

        let (persons, professions) = persons_and_professions_data();
        person_input.insert_with_same_weight(persons.iter(), 1);
        profession_input.insert_with_same_weight(professions.iter(), 1);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(0_u64, "Alice", 20_u64, 0_u64, "Engineer") => 1,
                tuple!(2_u64, "Charlie", 40_u64, 0_u64, "Engineer") => 1,
                tuple!(1_u64, "Bob", 30_u64, 1_u64, "Doctor") => 1,
            }
        );

        Ok(())
    }

    #[test]
    fn test_cartesian_product() -> Result<(), anyhow::Error> {
        let (circuit, (inputs, output)) = RootCircuit::build(|root_circuit| {
            let mut dbsp_inputs = DbspInputs::default();

            let code = [
                Stmt::from(VarStmt {
                    name: "person".to_string(),
                    initializer: Some(Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "person",
                            ["person_id", "name", "age", "profession_id"],
                            ["person_id"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }),
                Stmt::from(VarStmt {
                    name: "profession".to_string(),
                    initializer: Some(Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "profession",
                            ["profession_id", "name"],
                            ["profession_id"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }),
                Stmt::from(VarStmt {
                    name: "joined".to_string(),
                    initializer: Some(Expr::from(CartesianProductExpr::new(
                        Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("person")),
                            alias: "pers".to_string(),
                        }),
                        Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("profession")),
                            alias: "prof".to_string(),
                        }),
                        None,
                    ))),
                }),
            ];

            match IncLog::new().execute(code) {
                Ok(Some(Value::Relation(relation))) => {
                    let relation = relation.borrow();
                    let output_handle = relation.inner.output();
                    let output_schema = relation.schema.clone();
                    Ok((dbsp_inputs, DbspOutput::new(output_schema, output_handle)))
                }
                result => panic!("Expected a relation, got {:?}", result),
            }
        })?;
        let person_input = inputs.get("person").unwrap();
        let profession_input = inputs.get("profession").unwrap();

        let (persons, professions) = persons_and_professions_data();
        person_input.insert_with_same_weight(persons.iter(), 1);
        profession_input.insert_with_same_weight(professions.iter(), 1);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_debug_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(0_u64, "Alice", 20_u64, 0_u64, 0_u64, "Engineer") => 1,
                tuple!(0_u64, "Alice", 20_u64, 0_u64, 1_u64, "Doctor") => 1,
                tuple!(1_u64, "Bob", 30_u64, 1_u64, 0_u64, "Engineer") => 1,
                tuple!(1_u64, "Bob", 30_u64, 1_u64, 1_u64, "Doctor") => 1,
                tuple!(2_u64, "Charlie", 40_u64, 0_u64, 0_u64, "Engineer") => 1,
                tuple!(2_u64, "Charlie", 40_u64, 0_u64, 1_u64, "Doctor") => 1,
            }
        );

        Ok(())
    }

    #[test]
    fn test_self_join() -> Result<(), anyhow::Error> {
        let (circuit, (inputs, output)) = RootCircuit::build(|root_circuit| {
            let mut dbsp_inputs = DbspInputs::default();

            let code = [
                Stmt::from(VarStmt {
                    name: "edges".to_string(),
                    initializer: Some(Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "edges",
                            ["from", "to", "weight", "active"],
                            ["from", "to"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }),
                Stmt::from(VarStmt {
                    name: "len_1".to_string(),
                    initializer: Some(Expr::from(ProjectionExpr {
                        relation: Expr::from(VarExpr::new("edges")),
                        attributes: ["from", "to"]
                            .into_iter()
                            .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                            .chain(
                                [
                                    ("cumulated_weight", Expr::from(VarExpr::new("weight"))),
                                    (
                                        "hopcount",
                                        Expr::from(LiteralExpr {
                                            value: Literal::Uint(1),
                                        }),
                                    ),
                                ]
                                .map(|(name, expr)| (name.to_string(), expr)),
                            )
                            .collect(),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "len_2".to_string(),
                    initializer: Some(Expr::from(EquiJoinExpr {
                        left: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("len_1")),
                            alias: "cur".to_string(),
                        }),
                        right: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("edges")),
                            alias: "next".to_string(),
                        }),
                        on: vec![(
                            Expr::from(VarExpr::new("to")),
                            Expr::from(VarExpr::new("from")),
                        )],
                        attributes: Some(
                            [
                                ("start", Expr::from(VarExpr::new("cur.from"))),
                                ("end", Expr::from(VarExpr::new("next.to"))),
                                (
                                    "cumulated_weight",
                                    Expr::from(BinaryExpr {
                                        operator: Operator::Addition,
                                        left: Expr::from(VarExpr::new("cur.cumulated_weight")),
                                        right: Expr::from(VarExpr::new("next.weight")),
                                    }),
                                ),
                                (
                                    "hopcount",
                                    Expr::from(BinaryExpr {
                                        operator: Operator::Addition,
                                        left: Expr::from(VarExpr::new("cur.hopcount")),
                                        right: Expr::from(LiteralExpr {
                                            value: Literal::Uint(1),
                                        }),
                                    }),
                                ),
                            ]
                            .into_iter()
                            .map(|(name, expr)| (name.to_string(), expr))
                            .collect(),
                        ),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "len_3".to_string(),
                    initializer: Some(Expr::from(EquiJoinExpr {
                        left: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("len_2")),
                            alias: "cur".to_string(),
                        }),
                        right: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("edges")),
                            alias: "next".to_string(),
                        }),
                        on: vec![(
                            Expr::from(VarExpr::new("end")),
                            Expr::from(VarExpr::new("from")),
                        )],
                        attributes: Some(
                            [
                                ("start", Expr::from(VarExpr::new("cur.start"))),
                                ("end", Expr::from(VarExpr::new("next.to"))),
                                (
                                    "cumulated_weight",
                                    Expr::from(BinaryExpr {
                                        operator: Operator::Addition,
                                        left: Expr::from(VarExpr::new("cur.cumulated_weight")),
                                        right: Expr::from(VarExpr::new("next.weight")),
                                    }),
                                ),
                                (
                                    "hopcount",
                                    Expr::from(BinaryExpr {
                                        operator: Operator::Addition,
                                        left: Expr::from(VarExpr::new("cur.hopcount")),
                                        right: Expr::from(LiteralExpr {
                                            value: Literal::Uint(1),
                                        }),
                                    }),
                                ),
                            ]
                            .into_iter()
                            .map(|(name, expr)| (name.to_string(), expr))
                            .collect(),
                        ),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "len_4".to_string(),
                    initializer: Some(Expr::from(EquiJoinExpr {
                        left: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("len_3")),
                            alias: "cur".to_string(),
                        }),
                        right: Expr::from(AliasExpr {
                            relation: Expr::from(VarExpr::new("edges")),
                            alias: "next".to_string(),
                        }),
                        on: vec![(
                            Expr::from(VarExpr::new("end")),
                            Expr::from(VarExpr::new("from")),
                        )],
                        attributes: Some(
                            [
                                ("start", Expr::from(VarExpr::new("cur.start"))),
                                ("end", Expr::from(VarExpr::new("next.to"))),
                                (
                                    "cumulated_weight",
                                    Expr::from(BinaryExpr {
                                        operator: Operator::Addition,
                                        left: Expr::from(VarExpr::new("cur.cumulated_weight")),
                                        right: Expr::from(VarExpr::new("next.weight")),
                                    }),
                                ),
                                (
                                    "hopcount",
                                    Expr::from(BinaryExpr {
                                        operator: Operator::Addition,
                                        left: Expr::from(VarExpr::new("cur.hopcount")),
                                        right: Expr::from(LiteralExpr {
                                            value: Literal::Uint(1),
                                        }),
                                    }),
                                ),
                            ]
                            .into_iter()
                            .map(|(name, expr)| (name.to_string(), expr))
                            .collect(),
                        ),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "full_closure".to_string(),
                    initializer: Some(Expr::from(UnionExpr {
                        relations: ["len_1", "len_2", "len_3", "len_4"]
                            .into_iter()
                            .map(|name| Expr::from(VarExpr::new(name)))
                            .collect(),
                    })),
                }),
            ];

            match IncLog::new().execute(code) {
                Ok(Some(Value::Relation(relation))) => {
                    let relation = relation.borrow();
                    let output_handle = relation.inner.output();
                    let output_schema = relation.schema.clone();
                    Ok((dbsp_inputs, DbspOutput::new(output_schema, output_handle)))
                }
                result => panic!("Expected a relation, got {:?}", result),
            }
        })?;
        let edges_input = inputs.get("edges").unwrap();

        let init_data = [
            Edge::new(0, 1, 1),
            // This edge is omitted: Edge::new(1, 2, 1),
            Edge::new(2, 3, 2),
            Edge::new(3, 4, 2),
        ];

        println!("Insert of init_data:");

        edges_input.insert_with_same_weight(init_data.iter(), 1);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(0_u64, 1_u64, 1_u64, 1_u64) => 1,
                tuple!(2_u64, 3_u64, 2_u64, 1_u64) => 1,
                tuple!(2_u64, 4_u64, 4_u64, 2_u64) => 1,
                tuple!(3_u64, 4_u64, 2_u64, 1_u64) => 1,
            }
        );

        let extra_data = [Edge::new(1, 2, 1)];

        println!("Insert of extra_data:");

        edges_input.insert_with_same_weight(extra_data.iter(), 1);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(0_u64, 2_u64, 2_u64, 2_u64) => 1,
                tuple!(1_u64, 2_u64, 1_u64, 1_u64) => 1,
                tuple!(0_u64, 3_u64, 4_u64, 3_u64) => 1,
                tuple!(1_u64, 3_u64, 3_u64, 2_u64) => 1,
                tuple!(0_u64, 4_u64, 6_u64, 4_u64) => 1,
                tuple!(1_u64, 4_u64, 5_u64, 3_u64) => 1,
            }
        );

        Ok(())
    }

    #[test]
    fn test_iteration() -> Result<(), anyhow::Error> {
        let (circuit, (inputs, output)) = RootCircuit::build(|root_circuit| {
            let mut dbsp_inputs = DbspInputs::default();

            let code = [
                Stmt::from(VarStmt {
                    name: "edges".to_string(),
                    initializer: Some(Expr::from(ProjectionExpr {
                        relation: Expr::from(DbspInput::add(
                            RelationSchema::new(
                                "edges",
                                ["from", "to", "weight", "active"],
                                ["from", "to"],
                            )?,
                            root_circuit,
                            &mut dbsp_inputs,
                        )),
                        attributes: ["from", "to", "weight"]
                            .into_iter()
                            .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                            .collect(),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "base".to_string(),
                    initializer: Some(Expr::from(ProjectionExpr {
                        relation: Expr::from(VarExpr::new("edges")),
                        attributes: ["from", "to"]
                            .into_iter()
                            .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                            .chain(
                                [
                                    ("cumulated_weight", Expr::from(VarExpr::new("weight"))),
                                    (
                                        "hopcount",
                                        Expr::from(LiteralExpr {
                                            value: Literal::Uint(1),
                                        }),
                                    ),
                                ]
                                .map(|(name, expr)| (name.to_string(), expr)),
                            )
                            .collect(),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "closure".to_string(),
                    initializer: Some(Expr::from(FixedPointIterExpr {
                        circuit: root_circuit.clone(),
                        imports: ["edges"]
                            .into_iter()
                            .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                            .collect(),
                        accumulator: ("accumulator".to_string(), Expr::from(VarExpr::new("base"))),
                        step: BlockStmt {
                            stmts: vec![Stmt::from(ExprStmt {
                                expr: Expr::from(EquiJoinExpr {
                                    left: Expr::from(AliasExpr {
                                        relation: Expr::from(VarExpr::new("accumulator")),
                                        alias: "cur".to_string(),
                                    }),
                                    right: Expr::from(AliasExpr {
                                        relation: Expr::from(VarExpr::new("edges")),
                                        alias: "next".to_string(),
                                    }),
                                    on: vec![(
                                        Expr::from(VarExpr::new("to")),
                                        Expr::from(VarExpr::new("from")),
                                    )],
                                    attributes: Some(
                                        [
                                            ("start", Expr::from(VarExpr::new("cur.from"))),
                                            ("end", Expr::from(VarExpr::new("next.to"))),
                                            (
                                                "cumulated_weight",
                                                Expr::from(BinaryExpr {
                                                    operator: Operator::Addition,
                                                    left: Expr::from(VarExpr::new(
                                                        "cur.cumulated_weight",
                                                    )),
                                                    right: Expr::from(VarExpr::new("next.weight")),
                                                }),
                                            ),
                                            (
                                                "hopcount",
                                                Expr::from(BinaryExpr {
                                                    operator: Operator::Addition,
                                                    left: Expr::from(VarExpr::new("cur.hopcount")),
                                                    right: Expr::from(LiteralExpr {
                                                        value: Literal::Uint(1),
                                                    }),
                                                }),
                                            ),
                                        ]
                                        .into_iter()
                                        .map(|(name, expr)| (name.to_string(), expr))
                                        .collect(),
                                    ),
                                }),
                            })],
                        },
                    })),
                }),
            ];

            match IncLog::new().execute(code) {
                Ok(Some(Value::Relation(relation))) => {
                    let relation = relation.borrow();
                    let output_handle = relation.inner.output();
                    let output_schema = relation.schema.clone();
                    Ok((dbsp_inputs, DbspOutput::new(output_schema, output_handle)))
                }
                result => panic!("Expected a relation, got {:?}", result),
            }
        })?;
        let edges_input = inputs.get("edges").unwrap();

        let init_data = [
            Edge::new(0, 1, 1),
            Edge::new(1, 2, 1),
            Edge::new(2, 3, 2),
            Edge::new(3, 4, 2),
        ];

        println!("Insert of init_data:");

        edges_input.insert_with_same_weight(init_data.iter(), 1);

        circuit.step()?;

        let batch = output.to_batch();
        println!("{}", batch.as_table());
        assert_eq!(
            batch.as_debug_zset(),
            zset! {
                tuple!(0_u64, 1_u64, 1_u64, 1_u64) => 1,
                tuple!(0_u64, 2_u64, 2_u64, 2_u64) => 1,
                tuple!(1_u64, 2_u64, 1_u64, 1_u64) => 1,
                tuple!(0_u64, 3_u64, 4_u64, 3_u64) => 1,
                tuple!(1_u64, 3_u64, 3_u64, 2_u64) => 1,
                tuple!(2_u64, 3_u64, 2_u64, 1_u64) => 1,
                tuple!(0_u64, 4_u64, 6_u64, 4_u64) => 1,
                tuple!(1_u64, 4_u64, 5_u64, 3_u64) => 1,
                tuple!(2_u64, 4_u64, 4_u64, 2_u64) => 1,
                tuple!(3_u64, 4_u64, 2_u64, 1_u64) => 1,
            }
        );

        Ok(())
    }

    #[derive(Copy, Clone, Debug)]
    struct PredRel {
        from_node_id: u64,
        from_counter: u64,
        to_node_id: u64,
        to_counter: u64,
    }

    impl PredRel {
        fn new(from_node_id: u64, from_counter: u64, to_node_id: u64, to_counter: u64) -> Self {
            Self {
                from_node_id,
                from_counter,
                to_node_id,
                to_counter,
            }
        }
    }

    impl From<PredRel> for TupleKey {
        fn from(pred_rel: PredRel) -> Self {
            TupleKey::from_iter([
                pred_rel.from_node_id,
                pred_rel.from_counter,
                pred_rel.to_node_id,
                pred_rel.to_counter,
            ])
        }
    }

    impl From<PredRel> for TupleValue {
        fn from(pred_rel: PredRel) -> Self {
            TupleValue::from_iter([
                pred_rel.from_node_id,
                pred_rel.from_counter,
                pred_rel.to_node_id,
                pred_rel.to_counter,
            ])
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct SetOp {
        node_id: u64,
        counter: u64,
        key: u64,
        value: u64,
    }

    impl SetOp {
        fn new(node_id: u64, counter: u64, key: u64, value: u64) -> Self {
            Self {
                node_id,
                counter,
                key,
                value,
            }
        }
    }

    impl From<SetOp> for TupleKey {
        fn from(set_op: SetOp) -> Self {
            TupleKey::from_iter([set_op.node_id, set_op.counter])
        }
    }

    impl From<SetOp> for TupleValue {
        fn from(set_op: SetOp) -> Self {
            TupleValue::from_iter([set_op.node_id, set_op.counter, set_op.key, set_op.value])
        }
    }

    struct Replica {
        node_id: u64,
        counter: u64,
    }

    impl Replica {
        fn new(node_id: u64) -> Self {
            Self {
                node_id,
                counter: 0,
            }
        }
        fn generate_set(&mut self, key: u64, value: u64) -> SetOp {
            // TODO: pred op
            let result = SetOp {
                node_id: self.node_id,
                counter: self.counter,
                key,
                value,
            };
            self.counter += 1;
            result
        }
    }

    #[test]
    fn test_mvr_store_crdt() -> Result<(), anyhow::Error> {
        let (circuit, (inputs, output)) = RootCircuit::build(|root_circuit| {
            let mut dbsp_inputs = DbspInputs::default();

            let code = [
                // Inputs start.
                Stmt::from(VarStmt {
                    name: "pred_rel".to_string(),
                    initializer: Some(Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "pred_rel",
                            ["from_node_id", "from_counter", "to_node_id", "to_counter"],
                            ["from_node_id", "from_counter", "to_node_id", "to_counter"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }),
                Stmt::from(VarStmt {
                    name: "set_op".to_string(),
                    initializer: Some(Expr::from(DbspInput::add(
                        RelationSchema::new(
                            "set_op",
                            ["node_id", "counter", "key", "value"],
                            ["node_id", "counter"],
                        )?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }),
                // Inputs end.
                Stmt::from(VarStmt {
                    name: "overwritten".to_string(),
                    initializer: Some(Expr::from(DistinctExpr {
                        relation: Expr::from(ProjectionExpr {
                            relation: Expr::from(VarExpr::new("pred_rel")),
                            attributes: [("node_id", "from_node_id"), ("counter", "from_counter")]
                                .into_iter()
                                .map(|(name, origin)| {
                                    (name.to_string(), Expr::from(VarExpr::new(origin)))
                                })
                                .collect(),
                        }),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "overwrites".to_string(),
                    initializer: Some(Expr::from(DistinctExpr {
                        relation: Expr::from(ProjectionExpr {
                            relation: Expr::from(VarExpr::new("pred_rel")),
                            attributes: [("node_id", "to_node_id"), ("counter", "to_counter")]
                                .into_iter()
                                .map(|(name, origin)| {
                                    (name.to_string(), Expr::from(VarExpr::new(origin)))
                                })
                                .collect(),
                        }),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "root".to_string(),
                    initializer: Some(Expr::from(DifferenceExpr {
                        left: Expr::from(ProjectionExpr {
                            relation: Expr::from(VarExpr::new("set_op")),
                            attributes: ["node_id", "counter"]
                                .into_iter()
                                .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                                .collect(),
                        }),
                        right: Expr::from(VarExpr::new("overwrites")),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "leaf".to_string(),
                    initializer: Some(Expr::from(DifferenceExpr {
                        left: Expr::from(ProjectionExpr {
                            relation: Expr::from(VarExpr::new("set_op")),
                            attributes: ["node_id", "counter"]
                                .into_iter()
                                .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                                .collect(),
                        }),
                        right: Expr::from(VarExpr::new("overwritten")),
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "causally_ready".to_string(),
                    initializer: Some(Expr::from(FixedPointIterExpr {
                        circuit: root_circuit.clone(),
                        imports: ["pred_rel"]
                            .into_iter()
                            .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                            .collect(),
                        accumulator: ("accumulator".to_string(), Expr::from(VarExpr::new("root"))),
                        step: BlockStmt {
                            stmts: vec![Stmt::from(ExprStmt {
                                expr: Expr::from(EquiJoinExpr {
                                    left: Expr::from(AliasExpr {
                                        relation: Expr::from(VarExpr::new("accumulator")),
                                        alias: "cur".to_string(),
                                    }),
                                    right: Expr::from(AliasExpr {
                                        relation: Expr::from(VarExpr::new("pred_rel")),
                                        alias: "next".to_string(),
                                    }),
                                    on: vec![
                                        (
                                            Expr::from(VarExpr::new("node_id")),
                                            Expr::from(VarExpr::new("from_node_id")),
                                        ),
                                        (
                                            Expr::from(VarExpr::new("counter")),
                                            Expr::from(VarExpr::new("from_counter")),
                                        ),
                                    ],
                                    attributes: Some(
                                        [
                                            (
                                                "node_id",
                                                Expr::from(VarExpr::new("next.to_node_id")),
                                            ),
                                            (
                                                "counter",
                                                Expr::from(VarExpr::new("next.to_counter")),
                                            ),
                                        ]
                                        .into_iter()
                                        .map(|(name, expr)| (name.to_string(), expr))
                                        .collect(),
                                    ),
                                }),
                            })],
                        },
                    })),
                }),
                Stmt::from(VarStmt {
                    name: "mvr_store".to_string(),
                    initializer: Some(Expr::from(EquiJoinExpr {
                        left: Expr::from(VarExpr::new("causally_ready")),
                        right: Expr::from(EquiJoinExpr {
                            left: Expr::from(VarExpr::new("leaf")),
                            right: Expr::from(VarExpr::new("set_op")),
                            on: vec![
                                (
                                    Expr::from(VarExpr::new("node_id")),
                                    Expr::from(VarExpr::new("node_id")),
                                ),
                                (
                                    Expr::from(VarExpr::new("counter")),
                                    Expr::from(VarExpr::new("counter")),
                                ),
                            ],
                            // With `attributes: None` the query does not work because
                            // the fields `node_id` and `counter` are both duplicated in
                            // the tuple output. The EquiJoin below then indexes upon
                            // both duplicated fields for its `right` operand
                            // and no join match is found with its `left` operand.
                            // Welcome to the funny world of relational algebra's semantics.
                            attributes: Some(
                                [
                                    ("node_id", Expr::from(VarExpr::new("node_id"))),
                                    ("counter", Expr::from(VarExpr::new("counter"))),
                                    ("key", Expr::from(VarExpr::new("key"))),
                                    ("value", Expr::from(VarExpr::new("value"))),
                                ]
                                .into_iter()
                                .map(|(name, expr)| (name.to_string(), expr))
                                .collect(),
                            ),
                        }),
                        on: vec![
                            (
                                Expr::from(VarExpr::new("node_id")),
                                Expr::from(VarExpr::new("node_id")),
                            ),
                            (
                                Expr::from(VarExpr::new("counter")),
                                Expr::from(VarExpr::new("counter")),
                            ),
                        ],
                        attributes: Some(
                            [
                                ("key", Expr::from(VarExpr::new("key"))),
                                ("value", Expr::from(VarExpr::new("value"))),
                            ]
                            .into_iter()
                            .map(|(name, expr)| (name.to_string(), expr))
                            .collect(),
                        ),
                    })),
                }),
            ];

            match IncLog::new().execute(code) {
                Ok(Some(Value::Relation(relation))) => {
                    let relation = relation.borrow();
                    let output_handle = relation.inner.output();
                    let output_schema = relation.schema.clone();
                    Ok((dbsp_inputs, DbspOutput::new(output_schema, output_handle)))
                }
                result => panic!("Expected a relation, got {:?}", result),
            }
        })?;

        let pred_rel_input = inputs.get("pred_rel").unwrap();
        let set_op_input = inputs.get("set_op").unwrap();

        // The operation history is as follows:
        // In first step (just one root operation setting register with key 1 to value 1):
        //
        // set_0_0(1, 1)
        //
        // In second step (concurrent writes by replica 0 and 1):
        //
        //               ---> set_0_1(1, 2)
        // set_0_0(1, 1)
        //               ---> set_1_0(1, 3)
        //
        // In third step (replica 1 does a "merge" operation overwriting the previous conflict):
        //
        //               ---> set_0_1(1, 2)
        // set_0_0(1, 1)                    ---> set_1_2(1, 4)
        //               ---> set_1_0(1, 3)
        //

        let pred_rel_data = [
            vec![],
            vec![PredRel::new(0, 0, 0, 1), PredRel::new(0, 0, 1, 0)],
            vec![PredRel::new(0, 1, 1, 2), PredRel::new(1, 0, 1, 2)],
        ];

        let set_op_data = [
            vec![SetOp::new(0, 0, 1, 1)],
            vec![SetOp::new(0, 1, 1, 2), SetOp::new(1, 0, 1, 3)],
            vec![SetOp::new(1, 2, 1, 4)],
        ];

        let mut expected = [
            zset! {
                tuple!(1_u64, 1_u64) => 1,
            },
            zset! {
                tuple!(1_u64, 1_u64) => -1,
                tuple!(1_u64, 2_u64) => 1,
                tuple!(1_u64, 3_u64) => 1,
            },
            zset! {
                tuple!(1_u64, 2_u64) => -1,
                tuple!(1_u64, 3_u64) => -1,
                tuple!(1_u64, 4_u64) => 1,
            },
        ]
        .into_iter();

        for (pred_rel_step, set_op_step) in pred_rel_data.iter().zip(set_op_data.iter()) {
            pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
            set_op_input.insert_with_same_weight(set_op_step.iter(), 1);

            circuit.step()?;

            let batch = output.to_batch();
            println!("{}", batch.as_table());
            assert_eq!(batch.as_zset(), expected.next().unwrap());
        }

        Ok(())
    }
}
