use crate::{expr::Expr, util::MemAddr};

pub struct Program {
    pub stmts: Vec<Stmt>,
}

impl From<Vec<Stmt>> for Program {
    fn from(stmts: Vec<Stmt>) -> Self {
        Self { stmts }
    }
}

pub enum Stmt {
    Var(Box<VarStmt>),
    Expr(Box<ExprStmt>),
}

pub struct VarStmt {
    pub name: String,
    pub initializer: Option<Expr>,
}

// NOTE: When parsing, allow it to behave just like an expression if it is
// the last statement
pub struct ExprStmt {
    pub expr: Expr,
}

pub trait StmtVisitor<T, C> {
    fn visit_stmt(&mut self, stmt: &Stmt, ctx: C) -> T;
    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: C) -> T;
    fn visit_expr_stmt(&mut self, stmt: &ExprStmt, ctx: C) -> T;
}

impl MemAddr for Stmt {}
impl MemAddr for VarStmt {}
