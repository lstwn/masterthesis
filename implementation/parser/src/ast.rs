//! This module contains a representation of an AST (Abstract Syntax Tree)
//! for a Datalog variant.

use compute::expr::{Expr, VarExpr};

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Program {
    pub rules: Vec<Rule>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub head: Head,
    pub body: Body,
}

impl Rule {
    pub fn name(&self) -> &String {
        self.head.name()
    }
    pub fn is_extensional(&self) -> bool {
        // An extensional rule has no body.
        self.body.atoms.is_empty()
    }
    pub fn is_intensional(&self) -> bool {
        // An intensional rule has a body.
        !self.body.atoms.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Head {
    pub name: VarStmt,
    /// Here, we allow the variables to be an expression to create new columns.
    pub variables: Vec<VarStmt>,
}

impl Head {
    pub fn name(&self) -> &String {
        &self.name.identifier.inner
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Body {
    pub atoms: Vec<Atom>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Atom {
    Positive(Predicate),
    Negative(Predicate),
    Comparison(Expr),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Predicate {
    pub name: VarExpr,
    /// Here, the variables are just identifiers or aliases.
    pub variables: Vec<VarStmt>,
}

impl Predicate {
    pub fn name(&self) -> &String {
        &self.name.name
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarStmt {
    pub identifier: Identifier,
    pub initializer: Option<Expr>,
}

impl VarStmt {
    pub fn new<T: Into<Identifier>>(name: T) -> Self {
        VarStmt {
            identifier: name.into(),
            initializer: None,
        }
    }
    pub fn with_alias<T: Into<Identifier>>(target_name: T, origin: T) -> Self {
        VarStmt {
            identifier: target_name.into(),
            initializer: Some(Expr::from(VarExpr::from(origin.into()))),
        }
    }
    pub fn with_expr<T: Into<Identifier>>(target_name: T, origin: Expr) -> Self {
        VarStmt {
            identifier: target_name.into(),
            initializer: Some(origin),
        }
    }
    pub fn into_projection_attribute(self) -> (String, Expr) {
        let name = self.identifier.inner;
        let expr = self
            .initializer
            .unwrap_or_else(|| Expr::from(VarExpr::new(name.clone())));
        (name, expr)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identifier {
    pub inner: String,
}

impl<T: Into<String>> From<T> for Identifier {
    fn from(value: T) -> Self {
        Identifier {
            inner: value.into(),
        }
    }
}

impl From<Identifier> for VarExpr {
    fn from(value: Identifier) -> Self {
        VarExpr::new(value.inner)
    }
}
