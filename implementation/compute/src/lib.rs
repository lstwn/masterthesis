#![allow(dead_code, unused_variables)]

mod context;
mod dbsp;
mod error;
mod expr;
mod function;
mod interpreter;
mod operator;
mod relation;
mod resolver;
mod scalar;
mod stmt;
mod util;
mod variable;

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

struct IncLog {
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
        expr::{ProjectionExpr, SelectionExpr},
        relation::{Schema, TupleKey, TupleValue},
        scalar::ScalarTypedValue,
    };
    use ::dbsp::RootCircuit;
    use expr::{AssignExpr, BinaryExpr, CallExpr, Expr, Literal, LiteralExpr, VarExpr};
    use operator::Operator;
    use stmt::{ExprStmt, Stmt, VarStmt};

    #[test]
    fn test_variable_init_assign() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let initialization = vec![Stmt::Var(Box::new(VarStmt {
            name: "a".to_string(),
            initializer: Some(Expr::Literal(Box::new(LiteralExpr {
                value: Literal::Uint(1),
            }))),
        }))];

        let assignment = vec![Stmt::Expr(Box::new(ExprStmt {
            expr: Expr::Assign(Box::new(AssignExpr::new(
                "a".to_string(),
                Expr::Literal(Box::new(LiteralExpr {
                    value: Literal::Uint(2),
                })),
            ))),
        }))];

        assert_eq!(inclog.execute(initialization)?.unwrap(), Value::Uint(1));

        assert_eq!(inclog.execute(assignment)?.unwrap(), Value::Uint(2));

        Ok(())
    }

    // A function with two parameters which adds two values.
    fn new_add_function_expr() -> Expr {
        Expr::Function(Box::new(expr::FunctionExpr {
            parameters: vec!["a".to_string(), "b".to_string()],
            body: stmt::BlockStmt {
                stmts: vec![Stmt::Expr(Box::new(ExprStmt {
                    expr: Expr::Binary(Box::new(BinaryExpr {
                        operator: Operator::Addition,
                        left: Expr::Var(Box::new(VarExpr::new("a".to_string()))),
                        right: Expr::Var(Box::new(VarExpr::new("b".to_string()))),
                    })),
                }))],
            },
        }))
    }

    #[test]
    fn test_function_declarations() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let anonymous_function = vec![Stmt::Expr(Box::new(ExprStmt {
            expr: new_add_function_expr(),
        }))];

        let named_function = vec![Stmt::Var(Box::new(VarStmt {
            name: "add".to_string(),
            initializer: Some(new_add_function_expr()),
        }))];

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
            Stmt::Var(Box::new(VarStmt {
                name: "add".to_string(),
                initializer: Some(new_add_function_expr()),
            })),
            Stmt::Expr(Box::new(ExprStmt {
                expr: Expr::Call(Box::new(CallExpr {
                    callee: Expr::Var(Box::new(VarExpr::new("add".to_string()))),
                    arguments: vec![
                        Expr::Literal(Box::new(LiteralExpr {
                            value: Literal::Uint(1),
                        })),
                        Expr::Literal(Box::new(LiteralExpr {
                            value: Literal::Uint(2),
                        })),
                    ],
                })),
            })),
        ];

        let result = inclog.execute(function_call)?.unwrap();
        assert_eq!(Value::Uint(3), result);

        Ok(())
    }

    fn new_selection_expr(
        root_circuit: &mut RootCircuit,
    ) -> Result<(DbspInputs, DbspOutput), anyhow::Error> {
        let mut dbsp_inputs = DbspInputs::new();

        let code = [
            Stmt::Var(Box::new(VarStmt {
                name: "add".to_string(),
                initializer: Some(new_add_function_expr()),
            })),
            Stmt::Var(Box::new(VarStmt {
                name: "constant".to_string(),
                initializer: Some(Expr::Literal(Box::new(LiteralExpr {
                    value: Literal::Uint(1),
                }))),
            })),
            Stmt::Var(Box::new(VarStmt {
                name: "selected".to_string(),
                initializer: Some(Expr::Selection(Box::new(SelectionExpr {
                    condition: Expr::Binary(Box::new(BinaryExpr {
                        operator: Operator::GreaterEqual,
                        // TODO: Try more complex logical expression with and/or.
                        left: Expr::Var(Box::new(VarExpr::new("weight".to_string()))),
                        right: Expr::Call(Box::new(CallExpr {
                            callee: Expr::Var(Box::new(VarExpr::new("add".to_string()))),
                            arguments: vec![
                                Expr::Var(Box::new(VarExpr::new("constant".to_string()))),
                                Expr::Literal(Box::new(LiteralExpr {
                                    value: Literal::Uint(1),
                                })),
                            ],
                        })),
                    })),
                    relation: Expr::Literal(Box::new(DbspInput::new(
                        "edges",
                        Schema::new(["from", "to", "weight", "active"], ["from", "to"])?,
                        root_circuit,
                        &mut dbsp_inputs,
                    ))),
                }))),
            })),
            Stmt::Var(Box::new(VarStmt {
                name: "projected".to_string(),
                initializer: Some(Expr::Projection(Box::new(ProjectionExpr {
                    attributes: vec![
                        ("from".to_string(), None),
                        ("to".to_string(), None),
                        ("weight".to_string(), None),
                        (
                            // Here we create an entirely new column.
                            "mapped_weight".to_string(),
                            Some(Expr::Binary(Box::new(BinaryExpr {
                                operator: Operator::Multiplication,
                                left: Expr::Var(Box::new(VarExpr::new("from".to_string()))),
                                right: Expr::Var(Box::new(VarExpr::new("to".to_string()))),
                            }))),
                        ),
                    ],
                    relation: Expr::Var(Box::new(VarExpr::new("selected".to_string()))),
                }))),
            })),
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
    fn test_selection() -> Result<(), anyhow::Error> {
        // TODO: test multithreaded runtime
        let (circuit, (inputs, output)) = RootCircuit::build(new_selection_expr)?;
        let edges_input = inputs.get("edges").unwrap();

        let data1 = vec![Edge::new(0, 1, 1), Edge::new(1, 2, 2), Edge::new(2, 3, 3)];

        let data2 = vec![Edge::new(3, 4, 1), Edge::new(4, 5, 2), Edge::new(5, 6, 3)];

        println!("Insert of data1:");

        edges_input.insert_with_same_weight(data1.iter(), 1);

        circuit.step()?;

        println!("{}", output.to_table());

        println!("Insert of data2:");

        edges_input.insert_with_same_weight(data2.iter(), 1);

        circuit.step()?;

        println!("{}", output.to_table());

        println!("Removal of data1:");

        edges_input.insert_with_same_weight(data1.iter(), -1);

        circuit.step()?;

        println!("{}", output.to_table());

        Ok(())
    }
}
