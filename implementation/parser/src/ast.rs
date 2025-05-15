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
        &self.head.name.name
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
    pub name: VarExpr,
    /// Here, we allow the variables to be an expression to create new columns.
    pub variables: Vec<Expr>,
}

impl Head {
    pub fn name(&self) -> &String {
        &self.name.name
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
    /// Here, the variables are just identifiers.
    pub variables: Vec<VarExpr>,
}

impl Predicate {
    pub fn name(&self) -> &String {
        &self.name.name
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
