use std::collections::HashMap;

use crate::{
    relation::{Schema, Tuple},
    resolver::ScopeStack,
    stmt::Program,
    variable::Environment,
};

pub struct ProgramContext {
    /// Stores the program code executed so far by the interpreter.
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
    /// If the interpreter runs within a DBSP context, we store the currently
    /// processing tuple here for making each of its fields accessible
    /// as a variable.
    // No need to wrap it in an Option because HashMap::new() does not allocate!
    pub tuple_vars: HashMap<String, crate::scalar::ScalarTypedValue>,
}

impl InterpreterContext<'_> {
    pub fn new<'a>(environment: &'a mut Environment) -> InterpreterContext<'a> {
        InterpreterContext {
            environment,
            tuple_vars: HashMap::new(),
        }
    }
    pub fn begin_tuple_ctx<T: Tuple>(&mut self, schema: &Schema, tuple: &T) {
        self.tuple_vars = schema
            .all_attributes
            .iter()
            .map(|(name, index)| (name.clone(), tuple.data(*index).clone()))
            .collect();
    }
    pub fn end_tuple_ctx(&mut self) {
        self.tuple_vars.clear();
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
