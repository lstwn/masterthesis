use crate::{expr::Expr, util::MemAddr};

pub enum Stmt {
    Var(Box<VarStmt>),
}

pub struct VarStmt {
    pub name: String,
    pub initializer: Option<Expr>,
}

pub trait StmtVisitor<T, C> {
    fn visit_stmt(&mut self, stmt: &Stmt, ctx: C) -> T;
    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: C) -> T;
}

impl MemAddr for Stmt {}
impl MemAddr for VarStmt {}
