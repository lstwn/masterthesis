#![allow(dead_code, unused_variables)]

mod dbsp_playground;
mod env;
mod error;
mod expr;
mod function;
mod interpreter;
mod operator;
mod relation;
mod scalar;
mod stmt;
mod util;
mod variable;

use env::{Resolver, Val};
use error::IncLogError;
use interpreter::Interpreter;
use stmt::Program;

// Var: Variable
// Val: Value
// Stmt: Statement
// Expr: Expression
// Ident: Identifier

struct IncLog {
    interpreter: Interpreter,
    had_syntax_err: bool,
    had_runtime_err: bool,
}

impl IncLog {
    fn new() -> Self {
        Self {
            interpreter: Interpreter::new(),
            had_syntax_err: false,
            had_runtime_err: false,
        }
    }
    fn run_and_print(&mut self, source: String) {
        match self.run(source) {
            Ok(Some(val)) => println!("{}", val),
            Ok(None) => (),
            Err(err) => eprintln!("{}", err),
        }
    }
    fn run(&mut self, source: String) -> Result<Option<Val>, IncLogError> {
        self.parse(source)
            .and_then(|program| self.execute(&program))
    }
    // Result<Expr, ParsingError (SyntaxError)>
    fn parse(&mut self, source: String) -> Result<Program, IncLogError> {
        // Should actually parse the input string and create an expression
        // or a list of statements.
        // let expr = expr::Expr::Lit(Box::new(expr::LitExpr {
        //     value: scalar::ScalarTypedValue::Uint(2),
        // }));
        // Ok(expr)
        todo!()
    }
    fn execute(&mut self, program: &Program) -> Result<Option<Val>, IncLogError> {
        Resolver::new(&mut self.interpreter)
            .resolve(program)
            .map_err(|err| self.ack_syntax_err(err))
            .and_then(|()| {
                self.interpreter
                    .interpret(program)
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
    use crate::expr::{AssignExpr, CallExpr};

    use super::*;
    use expr::{BinaryExpr, Expr, LitExpr, VarExpr};
    use operator::Operator;
    use scalar::ScalarTypedValue;
    use stmt::{ExprStmt, Stmt, VarStmt};

    #[test]
    fn test_variable_init_assign() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let initialization = Program::from(vec![Stmt::Var(Box::new(VarStmt {
            name: "a".to_string(),
            initializer: Some(Expr::Lit(Box::new(LitExpr {
                value: ScalarTypedValue::Uint(1),
            }))),
        }))]);

        let assignment = Program::from(vec![
            // TODO: What if we don't have a variable declaration but reuse the
            // variable from above like in a REPL session?
            Stmt::Var(Box::new(VarStmt {
                name: "a".to_string(),
                initializer: Some(Expr::Lit(Box::new(LitExpr {
                    value: ScalarTypedValue::Uint(1),
                }))),
            })),
            Stmt::Expr(Box::new(ExprStmt {
                expr: Expr::Assign(Box::new(AssignExpr {
                    name: "a".to_string(),
                    value: Expr::Lit(Box::new(LitExpr {
                        value: ScalarTypedValue::Uint(2),
                    })),
                })),
            })),
        ]);

        assert_eq!(inclog.execute(&initialization)?.unwrap(), Val::Uint(1));

        assert_eq!(inclog.execute(&assignment)?.unwrap(), Val::Uint(2));

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
                        left: Expr::Var(Box::new(VarExpr {
                            name: "a".to_string(),
                        })),
                        right: Expr::Var(Box::new(VarExpr {
                            name: "b".to_string(),
                        })),
                    })),
                }))],
            },
        }))
    }

    #[test]
    fn test_function_declarations() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let anonymous_function = Program::from(vec![Stmt::Expr(Box::new(ExprStmt {
            expr: new_add_function_expr(),
        }))]);

        let named_function = Program::from(vec![Stmt::Var(Box::new(VarStmt {
            name: "add".to_string(),
            initializer: Some(new_add_function_expr()),
        }))]);

        let result = inclog.execute(&anonymous_function)?.unwrap();
        assert_eq!(format!("{}", result), "<anonymous fn(a, b)>");

        let result = inclog.execute(&named_function)?.unwrap();
        assert_eq!(format!("{}", result), "<fn add(a, b)>");

        Ok(())
    }

    #[test]
    fn test_function_call() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let function_call = Program::from(vec![
            Stmt::Var(Box::new(VarStmt {
                name: "add".to_string(),
                initializer: Some(new_add_function_expr()),
            })),
            Stmt::Expr(Box::new(ExprStmt {
                expr: Expr::Call(Box::new(CallExpr {
                    callee: Expr::Var(Box::new(VarExpr {
                        name: "add".to_string(),
                    })),
                    arguments: vec![
                        Expr::Lit(Box::new(LitExpr {
                            value: ScalarTypedValue::Uint(1),
                        })),
                        Expr::Lit(Box::new(LitExpr {
                            value: ScalarTypedValue::Uint(2),
                        })),
                    ],
                })),
            })),
        ]);

        let result = inclog.execute(&function_call)?.unwrap();
        assert_eq!(Val::Uint(3), result);

        Ok(())
    }
}
