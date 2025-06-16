#![allow(dead_code, unused_variables)]

mod context;
pub mod dbsp;
pub mod error;
pub mod expr;
mod function;
mod interpreter;
pub mod operator;
mod operators;
mod optimizer;
pub mod relation;
mod resolver;
pub mod scalar;
pub mod stmt;
#[cfg(feature = "test")]
pub mod test_helper;
pub mod type_resolver;
mod util;
pub mod variable;

use crate::{
    dbsp::{DbspInputs, DbspOutput},
    error::{RuntimeError, SyntaxError},
    optimizer::Optimizer,
    stmt::Code,
};
use context::{InterpreterContext, ProgramContext, ResolverContext};
use dbsp::{DbspError, DbspHandle, RootCircuit, Runtime};
use error::IncLogError;
use interpreter::Interpreter;
use resolver::Resolver;
use std::num::NonZeroUsize;
use stmt::Stmt;
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

#[derive(Clone)]
pub struct IncDataLog {
    threads: NonZeroUsize,
    optimize: bool,
}

impl Default for IncDataLog {
    fn default() -> Self {
        Self {
            threads: NonZeroUsize::new(1).unwrap(),
            optimize: true,
        }
    }
}

impl IncDataLog {
    pub fn new(threads: NonZeroUsize, optimize: bool) -> Self {
        Self { threads, optimize }
    }
    pub fn build_circuit_from_ir<F, Code>(
        &self,
        intermediate_representation: F,
    ) -> Result<(DbspHandle, DbspInputs, DbspOutput), anyhow::Error>
    where
        Code: IntoIterator<Item = Stmt>,
        F: Fn(&mut RootCircuit, &mut DbspInputs) -> Result<Code, SyntaxError>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let optimizer = self.init_optimizer();
        let (circuit, (inputs, output)) = self.init_dbsp_runtime(move |root_circuit| {
            let mut inputs = DbspInputs::default();
            let naive_program = intermediate_representation(root_circuit, &mut inputs)?
                .into_iter()
                .collect();
            let optimized_program = if let Some(optimizer) = optimizer {
                optimizer.optimize(naive_program)?
            } else {
                naive_program
            };
            Self::build_circuit(inputs, optimized_program).map_err(anyhow::Error::from)
        })?;

        Ok((circuit, inputs, output))
    }
    pub fn build_circuit_from_parser<F>(
        &self,
        parser: F,
    ) -> Result<(DbspHandle, DbspInputs, DbspOutput), anyhow::Error>
    where
        F: Fn(&mut RootCircuit) -> Result<(DbspInputs, Code), SyntaxError> + Clone + Send + 'static,
    {
        let optimizer = self.init_optimizer();
        let (circuit, (inputs, output)) = self.init_dbsp_runtime(move |root_circuit| {
            let (inputs, naive_program) = parser(root_circuit)?;
            let optimized_program = if let Some(optimizer) = optimizer {
                optimizer.optimize(naive_program)?
            } else {
                naive_program
            };
            Self::build_circuit(inputs, optimized_program).map_err(anyhow::Error::from)
        })?;

        Ok((circuit, inputs, output))
    }
    fn init_dbsp_runtime<F, T>(&self, constructor: F) -> Result<(DbspHandle, T), DbspError>
    where
        F: FnOnce(&mut RootCircuit) -> Result<T, anyhow::Error> + Clone + Send + 'static,
        T: Send + 'static,
    {
        Runtime::init_circuit(usize::from(self.threads), constructor)
    }
    fn init_optimizer(&self) -> Option<Optimizer> {
        if self.optimize {
            Some(Optimizer::default())
        } else {
            None
        }
    }
    fn build_circuit(
        inputs: DbspInputs,
        program: Code,
    ) -> Result<(DbspInputs, DbspOutput), RuntimeError> {
        let output = IncLog::default().execute(program);

        let output = match output {
            Ok(Some(Value::Relation(relation))) => {
                let relation = relation.borrow();
                let output_handle = relation.inner.output();
                let output_schema = relation.schema.clone();
                DbspOutput::new(output_schema, output_handle)
            }
            result => {
                return Err(RuntimeError::new(format!(
                    "Expected a relation as program's output, got {:?}",
                    result
                )));
            }
        };

        Ok((inputs, output))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        dbsp::{DbspInput, zset},
        expr::{
            AliasExpr, CartesianProductExpr, DifferenceExpr, DistinctExpr, EquiJoinExpr,
            FixedPointIterExpr, ProjectionExpr, SelectionExpr, UnionExpr,
        },
        relation::TupleValue,
        scalar::ScalarTypedValue,
        stmt::BlockStmt,
        test_helper::setup_inc_data_log,
    };
    use expr::{AssignExpr, BinaryExpr, CallExpr, Expr, Literal, LiteralExpr, VarExpr};
    use operator::Operator;
    use stmt::{ExprStmt, Stmt, VarStmt};
    use test_helper::{Edge, InputEntity, Person, PredRel, Profession, SetOp};

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

    #[test]
    fn test_selection_and_projection() -> Result<(), anyhow::Error> {
        let (mut circuit, inputs, output) =
            setup_inc_data_log().build_circuit_from_ir(|root_circuit, dbsp_inputs| {
                Ok([
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
                                Edge::schema(),
                                root_circuit,
                                dbsp_inputs,
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
                ])
            })?;

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

    #[test]
    fn test_standard_join() -> Result<(), anyhow::Error> {
        let (mut circuit, inputs, output) =
            setup_inc_data_log().build_circuit_from_ir(|root_circuit, dbsp_inputs| {
                let code = [
                    Stmt::from(VarStmt {
                        name: "person".to_string(),
                        initializer: Some(Expr::from(DbspInput::add(
                            Person::schema(),
                            root_circuit,
                            dbsp_inputs,
                        ))),
                    }),
                    Stmt::from(VarStmt {
                        name: "profession".to_string(),
                        initializer: Some(Expr::from(DbspInput::add(
                            Profession::schema(),
                            root_circuit,
                            dbsp_inputs,
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
                            on: vec![(
                                Expr::from(VarExpr::new("profession_id")),
                                Expr::from(VarExpr::new("profession_id")),
                            )],
                            // attributes: None,
                            attributes: Some(
                                // Here, we filter out the duplicated profession_id column
                                // that occurs after the join.
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
                Ok(code)
            })?;

        let person_input = inputs.get("person").unwrap();
        let profession_input = inputs.get("profession").unwrap();

        person_input.insert_with_same_weight(&Vec::from_iter(Person::data()), 1);
        profession_input.insert_with_same_weight(&Vec::from_iter(Profession::data()), 1);

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
        let (mut circuit, inputs, output) =
            setup_inc_data_log().build_circuit_from_ir(|root_circuit, dbsp_inputs| {
                let code = [
                    Stmt::from(VarStmt {
                        name: "person".to_string(),
                        initializer: Some(Expr::from(DbspInput::add(
                            Person::schema(),
                            root_circuit,
                            dbsp_inputs,
                        ))),
                    }),
                    Stmt::from(VarStmt {
                        name: "profession".to_string(),
                        initializer: Some(Expr::from(DbspInput::add(
                            Profession::schema(),
                            root_circuit,
                            dbsp_inputs,
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

                Ok(code)
            })?;

        let person_input = inputs.get("person").unwrap();
        let profession_input = inputs.get("profession").unwrap();

        person_input.insert_with_same_weight(&Vec::from_iter(Person::data()), 1);
        profession_input.insert_with_same_weight(&Vec::from_iter(Profession::data()), 1);

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
        let (mut circuit, inputs, output) =
            setup_inc_data_log().build_circuit_from_ir(|root_circuit, dbsp_inputs| {
                let code = [
                    Stmt::from(VarStmt {
                        name: "edges".to_string(),
                        initializer: Some(Expr::from(DbspInput::add(
                            Edge::schema(),
                            root_circuit,
                            dbsp_inputs,
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
                Ok(code)
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
        let (mut circuit, inputs, output) =
            setup_inc_data_log().build_circuit_from_ir(|root_circuit, dbsp_inputs| {
                let code = [
                    Stmt::from(VarStmt {
                        name: "edges".to_string(),
                        initializer: Some(Expr::from(ProjectionExpr {
                            relation: Expr::from(DbspInput::add(
                                Edge::schema(),
                                root_circuit,
                                dbsp_inputs,
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
                            accumulator: (
                                "accumulator".to_string(),
                                Expr::from(VarExpr::new("base")),
                            ),
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
                                                        right: Expr::from(VarExpr::new(
                                                            "next.weight",
                                                        )),
                                                    }),
                                                ),
                                                (
                                                    "hopcount",
                                                    Expr::from(BinaryExpr {
                                                        operator: Operator::Addition,
                                                        left: Expr::from(VarExpr::new(
                                                            "cur.hopcount",
                                                        )),
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
                Ok(code)
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

    #[test]
    fn test_mvr_store_crdt() -> Result<(), anyhow::Error> {
        let (mut circuit, inputs, output) =
            setup_inc_data_log().build_circuit_from_ir(|root_circuit, dbsp_inputs| {
                let code = [
                    // Inputs start.
                    Stmt::from(VarStmt {
                        name: "pred".to_string(),
                        initializer: Some(Expr::from(DbspInput::add(
                            PredRel::schema(),
                            root_circuit,
                            dbsp_inputs,
                        ))),
                    }),
                    Stmt::from(VarStmt {
                        name: "set".to_string(),
                        initializer: Some(Expr::from(DbspInput::add(
                            SetOp::schema(),
                            root_circuit,
                            dbsp_inputs,
                        ))),
                    }),
                    // Inputs end.
                    Stmt::from(VarStmt {
                        name: "overwritten".to_string(),
                        initializer: Some(Expr::from(DistinctExpr {
                            relation: Expr::from(ProjectionExpr {
                                relation: Expr::from(VarExpr::new("pred")),
                                attributes: [("NodeId", "FromNodeId"), ("Counter", "FromCounter")]
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
                                relation: Expr::from(VarExpr::new("pred")),
                                attributes: [("NodeId", "ToNodeId"), ("Counter", "ToCounter")]
                                    .into_iter()
                                    .map(|(name, origin)| {
                                        (name.to_string(), Expr::from(VarExpr::new(origin)))
                                    })
                                    .collect(),
                            }),
                        })),
                    }),
                    Stmt::from(VarStmt {
                        name: "isRoot".to_string(),
                        initializer: Some(Expr::from(DifferenceExpr {
                            left: Expr::from(ProjectionExpr {
                                relation: Expr::from(VarExpr::new("set")),
                                attributes: ["NodeId", "Counter"]
                                    .into_iter()
                                    .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                                    .collect(),
                            }),
                            right: Expr::from(VarExpr::new("overwrites")),
                        })),
                    }),
                    Stmt::from(VarStmt {
                        name: "isLeaf".to_string(),
                        initializer: Some(Expr::from(DifferenceExpr {
                            left: Expr::from(ProjectionExpr {
                                relation: Expr::from(VarExpr::new("set")),
                                attributes: ["NodeId", "Counter"]
                                    .into_iter()
                                    .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                                    .collect(),
                            }),
                            right: Expr::from(VarExpr::new("overwritten")),
                        })),
                    }),
                    Stmt::from(VarStmt {
                        name: "isCausallyReady".to_string(),
                        initializer: Some(Expr::from(FixedPointIterExpr {
                            circuit: root_circuit.clone(),
                            imports: ["pred"]
                                .into_iter()
                                .map(|name| (name.to_string(), Expr::from(VarExpr::new(name))))
                                .collect(),
                            accumulator: (
                                "isCausallyReady".to_string(),
                                Expr::from(VarExpr::new("isRoot")),
                            ),
                            step: BlockStmt {
                                stmts: vec![Stmt::from(ExprStmt {
                                    expr: Expr::from(EquiJoinExpr {
                                        left: Expr::from(AliasExpr {
                                            relation: Expr::from(VarExpr::new("isCausallyReady")),
                                            alias: "cur".to_string(),
                                        }),
                                        right: Expr::from(AliasExpr {
                                            relation: Expr::from(VarExpr::new("pred")),
                                            alias: "next".to_string(),
                                        }),
                                        on: vec![
                                            (
                                                Expr::from(VarExpr::new("NodeId")),
                                                Expr::from(VarExpr::new("FromNodeId")),
                                            ),
                                            (
                                                Expr::from(VarExpr::new("Counter")),
                                                Expr::from(VarExpr::new("FromCounter")),
                                            ),
                                        ],
                                        attributes: Some(
                                            [
                                                (
                                                    "NodeId",
                                                    Expr::from(VarExpr::new("next.ToNodeId")),
                                                ),
                                                (
                                                    "Counter",
                                                    Expr::from(VarExpr::new("next.ToCounter")),
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
                        name: "mvrStore".to_string(),
                        initializer: Some(Expr::from(EquiJoinExpr {
                            left: Expr::from(VarExpr::new("isCausallyReady")),
                            right: Expr::from(EquiJoinExpr {
                                left: Expr::from(VarExpr::new("isLeaf")),
                                right: Expr::from(VarExpr::new("set")),
                                on: vec![
                                    (
                                        Expr::from(VarExpr::new("NodeId")),
                                        Expr::from(VarExpr::new("NodeId")),
                                    ),
                                    (
                                        Expr::from(VarExpr::new("Counter")),
                                        Expr::from(VarExpr::new("Counter")),
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
                                        ("NodeId", Expr::from(VarExpr::new("NodeId"))),
                                        ("Counter", Expr::from(VarExpr::new("Counter"))),
                                        ("Key", Expr::from(VarExpr::new("Key"))),
                                        ("Value", Expr::from(VarExpr::new("Value"))),
                                    ]
                                    .into_iter()
                                    .map(|(name, expr)| (name.to_string(), expr))
                                    .collect(),
                                ),
                            }),
                            on: vec![
                                (
                                    Expr::from(VarExpr::new("NodeId")),
                                    Expr::from(VarExpr::new("NodeId")),
                                ),
                                (
                                    Expr::from(VarExpr::new("Counter")),
                                    Expr::from(VarExpr::new("Counter")),
                                ),
                            ],
                            attributes: Some(
                                [
                                    ("Key", Expr::from(VarExpr::new("Key"))),
                                    ("Value", Expr::from(VarExpr::new("Value"))),
                                ]
                                .into_iter()
                                .map(|(name, expr)| (name.to_string(), expr))
                                .collect(),
                            ),
                        })),
                    }),
                ];
                Ok(code)
            })?;

        let pred_rel_input = inputs.get("pred").unwrap();
        let set_op_input = inputs.get("set").unwrap();

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
