use crate::{expr::Expr, util::MemAddr};

pub type Code = Vec<Stmt>;

#[derive(Clone, Debug)]
pub struct Program {
    stmts: Code,
    /// Index from which unexecuted code is stored in the program.
    partition_index: usize,
}

impl From<Code> for Program {
    fn from(stmts: Code) -> Self {
        Self {
            stmts,
            partition_index: 0,
        }
    }
}

impl Program {
    pub fn empty() -> Self {
        Self {
            stmts: Vec::new(),
            partition_index: 0,
        }
    }
    pub fn unexecuted_code(&self) -> impl Iterator<Item = &Stmt> {
        self.stmts.iter().skip(self.partition_index)
    }
    pub fn extend_program(&mut self, code: impl Iterator<Item = Stmt>) {
        self.partition_index = self.stmts.len();
        self.stmts.extend(code);
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    // TODO: control flow: IfStmt, WhileStmt, Return?, Print?
    Var(Box<VarStmt>),
    Expr(Box<ExprStmt>),
    Block(Box<BlockStmt>),
}

#[derive(Clone, Debug)]
pub struct VarStmt {
    pub name: String,
    pub initializer: Option<Expr>,
}

// NOTE: When parsing, allow it to behave just like an expression if it is
// the last statement
#[derive(Clone, Debug)]
pub struct ExprStmt {
    pub expr: Expr,
}

#[derive(Clone, Debug)]
pub struct BlockStmt {
    pub stmts: Vec<Stmt>,
}

pub trait StmtVisitor<T, C> {
    fn visit_stmt(&mut self, stmt: &Stmt, ctx: C) -> T {
        match stmt {
            Stmt::Var(stmt) => self.visit_var_stmt(stmt, ctx),
            Stmt::Expr(stmt) => self.visit_expr_stmt(stmt, ctx),
            Stmt::Block(stmt) => self.visit_block_stmt(stmt, ctx),
        }
    }
    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: C) -> T;
    fn visit_expr_stmt(&mut self, stmt: &ExprStmt, ctx: C) -> T;
    fn visit_block_stmt(&mut self, stmt: &BlockStmt, ctx: C) -> T;
}

impl MemAddr for Stmt {}
impl MemAddr for VarStmt {}
impl MemAddr for ExprStmt {}
impl MemAddr for BlockStmt {}
