use super::{operator::Operator, scalar::ScalarTypedValue};
use crate::util::MemAddr;

// TODO: FunctionExpr

#[derive(Clone)]
pub enum Expr {
    Ternary(Box<TernaryExpr>),
    Binary(Box<BinaryExpr>),
    Unary(Box<UnaryExpr>),
    Var(Box<VarExpr>),
    Lit(Box<LitExpr>),
    Call(Box<CallExpr>),
}

#[derive(Clone)]
pub struct TernaryExpr {
    pub operator: Operator,
    pub left: Expr,
    pub mid: Expr,
    pub right: Expr,
}

#[derive(Clone)]
pub struct BinaryExpr {
    pub operator: Operator,
    pub left: Expr,
    pub right: Expr,
}

#[derive(Clone)]
pub struct UnaryExpr {
    pub operator: Operator,
    pub operand: Expr,
}

#[derive(Clone)]
pub struct VarExpr {
    pub name: String,
}

#[derive(Clone)]
pub struct LitExpr {
    pub value: ScalarTypedValue,
}

#[derive(Clone)]
pub struct CallExpr {
    pub callee: Expr,
    pub arguments: Vec<Expr>,
}

pub trait ExprVisitor<T, C> {
    fn visit_expr(&mut self, expr: &Expr, ctx: C) -> T;
    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: C) -> T;
    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: C) -> T;
    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: C) -> T;
    fn visit_var_expr(&mut self, expr: &VarExpr, ctx: C) -> T;
    fn visit_lit_expr(&mut self, expr: &LitExpr, ctx: C) -> T;
    fn visit_call_expr(&mut self, expr: &CallExpr, ctx: C) -> T;
}

impl MemAddr for Expr {}
impl MemAddr for TernaryExpr {}
impl MemAddr for BinaryExpr {}
impl MemAddr for UnaryExpr {}
impl MemAddr for VarExpr {}
impl MemAddr for LitExpr {}
impl MemAddr for CallExpr {}
