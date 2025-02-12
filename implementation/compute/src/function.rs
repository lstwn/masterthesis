use std::fmt::{self, Display};

use crate::stmt::FunctionStmt;

pub struct Function {
    declaration: FunctionStmt,
    // Is an index pointing to the parent env enough?
    // Environments needs some notion of a parent!
    // Vec<Arc<Mutex<Environment>>> ? Then, copying is cheap
    // or single threaded: Vec<Rc<RefCell<Environment>>>?
    // environment: Environment,
}

impl Function {
    pub fn new(declaration: FunctionStmt) -> Self {
        Self {
            declaration,
            // environment: Environment::new(),
        }
    }
    pub fn arity(&self) -> usize {
        self.declaration.parameters.len()
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<fn {}>", self.declaration.name)
    }
}
