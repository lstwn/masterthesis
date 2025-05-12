//! This module contains a representation of an AST (Abstract Syntax Tree)
//! for a Datalog variant.

use compute::expr::{Expr, VarExpr};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub rules: Vec<Rule>,
}

impl Default for Program {
    fn default() -> Self {
        Self { rules: Vec::new() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub head: Head,
    pub body: Body,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Head {
    pub name: VarExpr,
    /// Here, we allow the variables to be an expression to create new columns.
    pub variables: Vec<Expr>,
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
