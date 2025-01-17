mod env;
mod error;
mod expr;
mod interpreter;
mod operator;
mod relation;
mod scalar;
mod stmt;
mod util;
mod variable;

use env::Resolver;
use error::IncLogError;
use expr::Expr;
use interpreter::Interpreter;
use scalar::ScalarTypedValue;

// Var: Variable
// Val: Value
// Stmt: Statement
// Expr: Expression
// Ident: Identifier

struct IncLog {
    had_syntax_err: bool,
    had_runtime_err: bool,
}

impl IncLog {
    fn new() -> Self {
        Self {
            had_syntax_err: false,
            had_runtime_err: false,
        }
    }
    fn run_and_print(&mut self, source: String) {
        match self.run(source) {
            Ok(val) => println!("{}", val),
            Err(err) => eprintln!("{}", err),
        }
    }
    fn run(&mut self, source: String) -> Result<ScalarTypedValue, IncLogError> {
        self.parse(source).and_then(|expr| self.execute(&expr))
    }
    // Result<Expr, ParsingError (SyntaxError)>
    fn parse(&mut self, source: String) -> Result<Expr, IncLogError> {
        // Should actually parse the input string and create an expression
        // or a list of statements.
        let expr = expr::Expr::Lit(Box::new(expr::LitExpr {
            value: scalar::ScalarTypedValue::Uint(2),
        }));
        Ok(expr)
    }
    fn execute(&mut self, expr: &Expr) -> Result<ScalarTypedValue, IncLogError> {
        Resolver::new()
            .to_environment(expr)
            .map_err(|err| self.ack_syntax_err(err))
            .and_then(|env| {
                Interpreter::new(env)
                    .evaluate(expr)
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
