use crate::{env::Environment, expr::FunctionExpr};
use std::fmt::{self, Debug, Display};

#[derive(Clone)]
pub struct Function {
    pub name: Option<String>,
    pub declaration: FunctionExpr,
    pub environment: Environment,
}

impl Function {
    pub fn new(name: Option<String>, declaration: FunctionExpr, environment: Environment) -> Self {
        Self {
            name,
            declaration,
            environment,
        }
    }
    pub fn arity(&self) -> usize {
        self.declaration.parameters.len()
    }
    fn to_string_helper(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params = format!("({})", self.declaration.parameters.join(", "));
        if let Some(name) = &self.name {
            write!(f, "<fn {}{}>", name, params)
        } else {
            write!(f, "<anonymous fn{}>", params)
        }
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string_helper(f)
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string_helper(f)
    }
}
