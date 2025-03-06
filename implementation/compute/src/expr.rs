use super::operator::Operator;
use crate::{
    relation::Relation,
    stmt::BlockStmt,
    util::{MemAddr, Named, Resolvable},
    variable::VariableSlot,
};
use std::fmt::{self, Display};

#[derive(Clone, Debug)]
pub enum Expr {
    // TODO: factor out LogicalExpr from BinaryExpr
    Ternary(Box<TernaryExpr>),
    Binary(Box<BinaryExpr>),
    Unary(Box<UnaryExpr>),
    Grouping(Box<GroupingExpr>),
    Var(Box<VarExpr>),
    Assign(Box<AssignExpr>),
    Literal(Box<LiteralExpr>),
    Call(Box<CallExpr>),
    Function(Box<FunctionExpr>),
    Selection(Box<SelectionExpr>),
}

#[derive(Clone, Debug)]
pub struct TernaryExpr {
    pub operator: Operator,
    pub left: Expr,
    pub mid: Expr,
    pub right: Expr,
}

#[derive(Clone, Debug)]
pub struct BinaryExpr {
    pub operator: Operator,
    pub left: Expr,
    pub right: Expr,
}

#[derive(Clone, Debug)]
pub struct UnaryExpr {
    pub operator: Operator,
    pub operand: Expr,
}

#[derive(Clone, Debug)]
pub struct GroupingExpr {
    pub expr: Expr,
}

#[derive(Clone, Debug)]
pub struct VarExpr {
    pub name: String,
    pub resolved: Option<VariableSlot>,
}

impl VarExpr {
    pub fn new(name: String) -> Self {
        Self {
            name,
            resolved: None,
        }
    }
}

impl Resolvable for VarExpr {
    fn set_resolved(&mut self, info: VariableSlot) -> () {
        self.resolved = Some(info);
    }
}

impl Named for VarExpr {
    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Debug)]
pub struct AssignExpr {
    pub name: String,
    pub value: Expr,
    pub resolved: Option<VariableSlot>,
}

impl AssignExpr {
    pub fn new(name: String, value: Expr) -> Self {
        Self {
            name,
            value,
            resolved: None,
        }
    }
}

impl Resolvable for AssignExpr {
    fn set_resolved(&mut self, info: VariableSlot) -> () {
        self.resolved = Some(info);
    }
}

impl Named for AssignExpr {
    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Debug)]
pub struct LiteralExpr {
    pub value: Literal,
}

#[derive(Clone, Debug)]
pub struct FunctionExpr {
    pub parameters: Vec<String>,
    pub body: BlockStmt,
}

#[derive(Clone, Debug)]
pub struct CallExpr {
    pub callee: Expr,
    pub arguments: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub struct SelectionExpr {
    pub relation: Expr,
    pub condition: Expr,
}

#[derive(Clone, Debug)]
pub enum Literal {
    /// String.
    String(String),
    /// Unsigned integer value of 64 bits.
    Uint(u64),
    /// Signed integer value of 64 bits.
    Iint(i64),
    /// Boolean.
    Bool(bool),
    /// Null.
    Null(()),
    /// Relation (an extensional database predicate).
    // With a textual representation of the language, this may be just an
    // identifier and the stream and the schema are elsewhere.
    Relation(Relation),
}

impl Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::String(value) => write!(f, "{}", value),
            Literal::Uint(value) => write!(f, "{}", value),
            Literal::Iint(value) => write!(f, "{}", value),
            Literal::Bool(value) => write!(f, "{}", value),
            Literal::Null(()) => write!(f, "null"),
            Literal::Relation(value) => write!(f, "relation {}", value.name),
        }
    }
}

pub trait ExprVisitor<T, C> {
    fn visit_expr(&mut self, expr: &Expr, ctx: C) -> T {
        match expr {
            Expr::Ternary(expr) => self.visit_ternary_expr(expr, ctx),
            Expr::Binary(expr) => self.visit_binary_expr(expr, ctx),
            Expr::Unary(expr) => self.visit_unary_expr(expr, ctx),
            Expr::Grouping(expr) => self.visit_grouping_expr(expr, ctx),
            Expr::Var(expr) => self.visit_var_expr(expr, ctx),
            Expr::Assign(expr) => self.visit_assign_expr(expr, ctx),
            Expr::Literal(expr) => self.visit_literal_expr(expr, ctx),
            Expr::Function(expr) => self.visit_function_expr(expr, ctx),
            Expr::Call(expr) => self.visit_call_expr(expr, ctx),
            Expr::Selection(expr) => self.visit_selection_expr(expr, ctx),
        }
    }
    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: C) -> T;
    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: C) -> T;
    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: C) -> T;
    fn visit_grouping_expr(&mut self, expr: &GroupingExpr, ctx: C) -> T;
    fn visit_var_expr(&mut self, expr: &VarExpr, ctx: C) -> T;
    fn visit_assign_expr(&mut self, expr: &AssignExpr, ctx: C) -> T;
    fn visit_literal_expr(&mut self, expr: &LiteralExpr, ctx: C) -> T;
    fn visit_function_expr(&mut self, expr: &FunctionExpr, ctx: C) -> T;
    fn visit_call_expr(&mut self, expr: &CallExpr, ctx: C) -> T;
    fn visit_selection_expr(&mut self, expr: &SelectionExpr, ctx: C) -> T;
}

pub trait ExprVisitorMut<T, C> {
    fn visit_expr(&mut self, expr: &mut Expr, ctx: C) -> T {
        match expr {
            Expr::Ternary(expr) => self.visit_ternary_expr(expr, ctx),
            Expr::Binary(expr) => self.visit_binary_expr(expr, ctx),
            Expr::Unary(expr) => self.visit_unary_expr(expr, ctx),
            Expr::Grouping(expr) => self.visit_grouping_expr(expr, ctx),
            Expr::Var(expr) => self.visit_var_expr(expr, ctx),
            Expr::Assign(expr) => self.visit_assign_expr(expr, ctx),
            Expr::Literal(expr) => self.visit_literal_expr(expr, ctx),
            Expr::Function(expr) => self.visit_function_expr(expr, ctx),
            Expr::Call(expr) => self.visit_call_expr(expr, ctx),
            Expr::Selection(expr) => self.visit_selection_expr(expr, ctx),
        }
    }
    fn visit_ternary_expr(&mut self, expr: &mut TernaryExpr, ctx: C) -> T;
    fn visit_binary_expr(&mut self, expr: &mut BinaryExpr, ctx: C) -> T;
    fn visit_unary_expr(&mut self, expr: &mut UnaryExpr, ctx: C) -> T;
    fn visit_grouping_expr(&mut self, expr: &mut GroupingExpr, ctx: C) -> T;
    fn visit_var_expr(&mut self, expr: &mut VarExpr, ctx: C) -> T;
    fn visit_assign_expr(&mut self, expr: &mut AssignExpr, ctx: C) -> T;
    fn visit_literal_expr(&mut self, expr: &mut LiteralExpr, ctx: C) -> T;
    fn visit_function_expr(&mut self, expr: &mut FunctionExpr, ctx: C) -> T;
    fn visit_call_expr(&mut self, expr: &mut CallExpr, ctx: C) -> T;
    fn visit_selection_expr(&mut self, expr: &mut SelectionExpr, ctx: C) -> T;
}

pub trait ExprVisitorOwn<T, C> {
    fn visit_expr(&mut self, expr: Expr, ctx: C) -> T {
        match expr {
            Expr::Ternary(expr) => self.visit_ternary_expr(*expr, ctx),
            Expr::Binary(expr) => self.visit_binary_expr(*expr, ctx),
            Expr::Unary(expr) => self.visit_unary_expr(*expr, ctx),
            Expr::Grouping(expr) => self.visit_grouping_expr(*expr, ctx),
            Expr::Var(expr) => self.visit_var_expr(*expr, ctx),
            Expr::Assign(expr) => self.visit_assign_expr(*expr, ctx),
            Expr::Literal(expr) => self.visit_literal_expr(*expr, ctx),
            Expr::Function(expr) => self.visit_function_expr(*expr, ctx),
            Expr::Call(expr) => self.visit_call_expr(*expr, ctx),
            Expr::Selection(expr) => self.visit_selection_expr(*expr, ctx),
        }
    }
    fn visit_ternary_expr(&mut self, expr: TernaryExpr, ctx: C) -> T;
    fn visit_binary_expr(&mut self, expr: BinaryExpr, ctx: C) -> T;
    fn visit_unary_expr(&mut self, expr: UnaryExpr, ctx: C) -> T;
    fn visit_grouping_expr(&mut self, expr: GroupingExpr, ctx: C) -> T;
    fn visit_var_expr(&mut self, expr: VarExpr, ctx: C) -> T;
    fn visit_assign_expr(&mut self, expr: AssignExpr, ctx: C) -> T;
    fn visit_literal_expr(&mut self, expr: LiteralExpr, ctx: C) -> T;
    fn visit_function_expr(&mut self, expr: FunctionExpr, ctx: C) -> T;
    fn visit_call_expr(&mut self, expr: CallExpr, ctx: C) -> T;
    fn visit_selection_expr(&mut self, expr: SelectionExpr, ctx: C) -> T;
}

impl MemAddr for Expr {}
impl MemAddr for TernaryExpr {}
impl MemAddr for BinaryExpr {}
impl MemAddr for UnaryExpr {}
impl MemAddr for GroupingExpr {}
impl MemAddr for VarExpr {}
impl MemAddr for AssignExpr {}
impl MemAddr for LiteralExpr {}
impl MemAddr for FunctionExpr {}
impl MemAddr for CallExpr {}
impl MemAddr for SelectionExpr {}
