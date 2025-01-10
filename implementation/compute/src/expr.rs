use super::{operator::Operator, scalar::ScalarTypedValue};

pub enum Expr {
    Nary(Box<NaryExpr>),
    Ternary(Box<TernaryExpr>),
    Binary(Box<BinaryExpr>),
    Unary(Box<UnaryExpr>),
    Variable(Box<IdentExpr>),
    Literal(Box<LitExpr>),
}

pub struct NaryExpr {
    pub operator: Operator,
    pub operands: Vec<Expr>,
}

pub struct TernaryExpr {
    pub operator: Operator,
    pub left: Expr,
    pub mid: Expr,
    pub right: Expr,
}

pub struct BinaryExpr {
    pub operator: Operator,
    pub left: Expr,
    pub right: Expr,
}

pub struct UnaryExpr {
    pub operator: Operator,
    pub operand: Expr,
}

pub struct IdentExpr {
    pub name: String,
}

pub struct LitExpr {
    pub value: ScalarTypedValue,
}

pub trait Visitor<T, C> {
    fn visit_expr(&mut self, expr: &Expr, ctx: C) -> T;
    fn visit_nary_expr(&mut self, expr: &NaryExpr, ctx: C) -> T;
    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: C) -> T;
    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: C) -> T;
    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: C) -> T;
    fn visit_ident_expr(&mut self, expr: &IdentExpr, ctx: C) -> T;
    fn visit_lit_expr(&mut self, expr: &LitExpr, ctx: C) -> T;
}
