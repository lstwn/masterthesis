use crate::{
    env::{Environment, ScopeStack},
    stmt::Program,
};

pub struct ProgramContext {
    /// We have to keep the AST in memory because functions may be stored
    /// for longer in the environment than the AST is available in a REPL
    /// session.
    /// Read-only during resolution and interpretation.
    pub program: Program,
    /// The environment stores the variables and their values.
    /// Writable and _changing_ during interpretation.
    pub environment: Environment,
    /// Scope stack for the resolver.
    /// Writable during resolution and not accessed during interpretation.
    pub scopes: ScopeStack,
}

impl ProgramContext {
    pub fn new() -> Self {
        Self {
            program: Program::empty(),
            environment: Environment::new(),
            scopes: ScopeStack::new(),
        }
    }
}

#[derive(Debug)]
pub struct InterpreterContext<'a> {
    pub environment: &'a mut Environment,
}

impl InterpreterContext<'_> {
    pub fn new<'a>(environment: &'a mut Environment) -> InterpreterContext<'a> {
        InterpreterContext { environment }
    }
}

pub struct ResolverContext<'a> {
    pub scopes: &'a mut ScopeStack,
    pub is_tuple_context: bool,
}

impl ResolverContext<'_> {
    pub fn new<'a>(scopes: &'a mut ScopeStack) -> ResolverContext<'a> {
        ResolverContext {
            scopes,
            is_tuple_context: false,
        }
    }
    pub fn begin_tuple_context(&mut self) {
        self.is_tuple_context = true;
    }
    pub fn end_tuple_context(&mut self) {
        self.is_tuple_context = false;
    }
}
