use std::fmt::{self, Debug, Display};

use crate::{env::Environment, stmt::FunctionStmt};

#[derive(Clone)]
pub struct Function {
    pub declaration: FunctionStmt,
    pub environment: Environment,
}

impl Function {
    pub fn new(declaration: FunctionStmt, environment: Environment) -> Self {
        Self {
            declaration,
            environment,
        }
    }
    pub fn arity(&self) -> usize {
        self.declaration.parameters.len()
    }
    // pub fn call(&mut self, interpreter: Interpreter, arguments: Vec<ScalarTypedValue>) {
    //     self.environment.begin_scope();
    //     for arg in arguments.into_iter() {
    //         self.environment.define_var(arg);
    //     }
    //     interpreter.execute_block(&self.declaration.body, &mut self.environment);
    //     self.environment.end_scope();
    // }
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {}>", self.declaration.name)
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {}>", self.declaration.name)
    }
}
