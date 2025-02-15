use crate::{expr::Expr, util::MemAddr};

#[derive(Clone)]
pub struct Program {
    pub stmts: Vec<Stmt>,
}

impl From<Vec<Stmt>> for Program {
    fn from(stmts: Vec<Stmt>) -> Self {
        Self { stmts }
    }
}

#[derive(Clone)]
pub enum Stmt {
    Var(Box<VarStmt>),
    Expr(Box<ExprStmt>),
    Block(Box<BlockStmt>),
    Function(Box<FunctionStmt>),
}

#[derive(Clone)]
pub struct VarStmt {
    pub name: String,
    pub initializer: Option<Expr>,
}

// NOTE: When parsing, allow it to behave just like an expression if it is
// the last statement
#[derive(Clone)]
pub struct ExprStmt {
    pub expr: Expr,
}

#[derive(Clone)]
pub struct BlockStmt {
    pub stmts: Vec<Stmt>,
}

// TODO: Get rid of function statements as functions are first-class citizens
// with FunctionExprs.
#[derive(Clone)]
pub struct FunctionStmt {
    pub name: String,
    pub parameters: Vec<String>,
    pub body: BlockStmt,
}

pub trait StmtVisitor<T, C> {
    fn visit_stmt(&mut self, stmt: &Stmt, ctx: C) -> T {
        match stmt {
            Stmt::Var(stmt) => self.visit_var_stmt(stmt, ctx),
            Stmt::Expr(stmt) => self.visit_expr_stmt(stmt, ctx),
            Stmt::Block(stmt) => self.visit_block_stmt(stmt, ctx),
            Stmt::Function(stmt) => self.visit_function_stmt(stmt, ctx),
        }
    }
    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: C) -> T;
    fn visit_expr_stmt(&mut self, stmt: &ExprStmt, ctx: C) -> T;
    fn visit_block_stmt(&mut self, stmt: &BlockStmt, ctx: C) -> T;
    fn visit_function_stmt(&mut self, stmt: &FunctionStmt, ctx: C) -> T;
}

impl MemAddr for Stmt {}
impl MemAddr for VarStmt {}
impl MemAddr for ExprStmt {}
impl MemAddr for BlockStmt {}
impl MemAddr for FunctionStmt {}
