use crate::{
    env::{Environment, NodeRef, VarIdent, Variable},
    stmt::Program,
};
use std::collections::HashMap;

pub struct ProgramContext {
    /// We have to keep the AST in memory because functions may be stored
    /// for longer in the environment than the AST is available in a REPL
    /// session.
    /// Read-only during resolution and interpretation.
    pub program: Program,
    /// Side table to store the resolved VarIdents for each variable.
    /// Writable during resolution. Read-only during interpretation.
    pub side_table: HashMap<NodeRef, VarIdent>,
    /// The environment stores the variables and their values.
    /// Writable and _changing_ during interpretation.
    pub environment: Environment,
    /// Scope stack for the resolver.
    /// Writable during resolution and not accessed during interpretation.
    pub scopes: Vec<HashMap<String, Variable>>,
}

impl ProgramContext {
    pub fn new() -> Self {
        Self {
            program: Program::empty(),
            side_table: HashMap::new(),
            environment: Environment::new(),
            scopes: Vec::new(),
        }
    }
}

pub struct InterpreterContext<'a> {
    pub side_table: &'a HashMap<NodeRef, VarIdent>,
    pub environment: &'a mut Environment,
}

impl InterpreterContext<'_> {
    pub fn with_new_environment<'a>(
        &'a mut self,
        environment: &'a mut Environment,
    ) -> InterpreterContext<'a> {
        InterpreterContext {
            side_table: self.side_table,
            environment,
        }
    }
}

pub struct ResolverContext<'a> {
    pub scopes: &'a mut Vec<HashMap<String, Variable>>,
    pub side_table: &'a mut HashMap<NodeRef, VarIdent>,
}

impl ResolverContext<'_> {
    pub fn resolve<T: Into<NodeRef>>(&mut self, expr: T, ident: VarIdent) -> () {
        self.side_table.insert(expr.into(), ident);
    }
}
