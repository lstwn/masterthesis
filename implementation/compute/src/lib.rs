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
    use super::*;
    use expr::{BinaryExpr, Expr, LitExpr, VarExpr};
    use operator::Operator;
    use scalar::ScalarTypedValue;
    use stmt::{ExprStmt, Stmt, VarStmt};

    #[test]
    fn test_inclog() -> Result<(), IncLogError> {
        let mut inclog = IncLog::new();

        let program = Program::from(vec![
            Stmt::Var(Box::new(VarStmt {
                name: "a".to_string(),
                initializer: Some(Expr::Lit(Box::new(LitExpr {
                    value: ScalarTypedValue::Uint(1),
                }))),
            })),
            Stmt::Var(Box::new(VarStmt {
                name: "b".to_string(),
                initializer: Some(Expr::Lit(Box::new(LitExpr {
                    value: ScalarTypedValue::Uint(2),
                }))),
            })),
            Stmt::Expr(Box::new(ExprStmt {
                expr: Expr::Binary(Box::new(BinaryExpr {
                    operator: Operator::Addition,
                    left: Expr::Var(Box::new(VarExpr {
                        name: "a".to_string(),
                    })),
                    right: Expr::Var(Box::new(VarExpr {
                        name: "b".to_string(),
                    })),
                })),
            })),
        ]);

        assert_eq!(Some(Val::Uint(3)), inclog.execute(&program)?);
        Ok(())
    }
}
