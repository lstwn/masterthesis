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
    Alias(Box<AliasExpr>),
    Union(Box<UnionExpr>),
    Selection(Box<SelectionExpr>),
    Projection(Box<ProjectionExpr>),
    EquiJoin(Box<EquiJoinExpr>),
    ThetaJoin(Box<ThetaJoinExpr>),
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
    pub fn new<T: Into<String>>(name: T) -> Self {
        Self {
            name: name.into(),
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
    pub fn new<T: Into<String>>(name: T, value: Expr) -> Self {
        Self {
            name: name.into(),
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
pub struct AliasExpr {
    pub relation: Expr,
    pub alias: String,
}

#[derive(Clone, Debug)]
pub struct UnionExpr {
    /// All `Expr`s must evaluate to a relation and have a compatible schema,
    /// that is, the same order and arity of attributes with same types, respectively.
    pub relations: Vec<Expr>,
}

#[derive(Clone, Debug)]
pub struct SelectionExpr {
    pub relation: Expr,
    pub condition: Expr,
}

#[derive(Clone, Debug)]
pub struct ProjectionExpr {
    /// Must evaluate to a relation.
    pub relation: Expr,
    /// The attributes to map over. The first element `String` is the name
    /// of the attribute. The second element `Expr` is the expression
    /// which produces the new value of the attribute.
    ///
    /// In case the `Expr` is just a `VarExpr` referencing a **tuple** variable,
    /// the interpreter is not run to evaluate the expression but instead only
    /// the schema is changed.
    pub attributes: Vec<(String, Expr)>,
}

/// An equi join is a join that exclusively uses equality of attribute(s).
/// [More information on join classifications](https://stackoverflow.com/a/7870216).
// TODO
#[derive(Clone, Debug)]
pub struct EquiJoinExpr {
    /// Must evaluate to a relation.
    pub left: Expr,
    /// Must evaluate to a relation.
    pub right: Expr,
    /// The attributes to join on. The first element is the attribute of the
    /// left relation, and the second element is the attribute of the right
    /// relation.
    /// Each attribute pair should produce the same type.
    /// Why not use a `Vec<(Expr, Expr, String)>`?
    pub on: Vec<(String, String)>,
    /// An optional projection step. See documentation of [`ProjectionExpr`].
    pub attributes: Option<Vec<(String, Expr)>>,
}

/// A theta join is a join that uses an arbitrary condition which may be more
/// complicated than just equality of attribute(s).
/// [More information on join classifications](https://stackoverflow.com/a/7870216).
#[derive(Clone, Debug)]
pub struct ThetaJoinExpr {
    /// Must evaluate to a relation.
    pub left: Expr,
    /// Must evaluate to a relation.
    pub right: Expr,
    pub condition: Expr,
    /// An optional projection step. See documentation of [`ProjectionExpr`].
    pub attributes: Option<Vec<(String, Expr)>>,
}

/// Iteration until the condition is met. Should include fixed-point computations.
// TODO
#[derive(Clone, Debug)]
pub struct GenericIterationExpr {
    pub condition: Expr,
    /// Must evaluate to a relation.
    pub body: Expr,
}

/// Evaluates to a relation/stream again.
#[derive(Clone, Debug)]
pub struct FixedPointIterationExpr {
    /// Must evaluate to a circuit.
    pub circuit: Expr,
    /// What to do as a preparation. Runs in the context of the parent circuit.
    pub parent: BlockStmt,
    /// What to do in each iteration. Runs in the context of the child circuit.
    pub child: BlockStmt,
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
            Literal::Relation(value) => write!(f, "relation {}", value.schema.name),
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
            Expr::Alias(expr) => self.visit_alias_expr(expr, ctx),
            Expr::Union(expr) => self.visit_union_expr(expr, ctx),
            Expr::Selection(expr) => self.visit_selection_expr(expr, ctx),
            Expr::Projection(expr) => self.visit_projection_expr(expr, ctx),
            Expr::EquiJoin(expr) => self.visit_equi_join_expr(expr, ctx),
            Expr::ThetaJoin(expr) => self.visit_theta_join_expr(expr, ctx),
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
    fn visit_alias_expr(&mut self, expr: &AliasExpr, ctx: C) -> T;
    fn visit_union_expr(&mut self, expr: &UnionExpr, ctx: C) -> T;
    fn visit_selection_expr(&mut self, expr: &SelectionExpr, ctx: C) -> T;
    fn visit_projection_expr(&mut self, expr: &ProjectionExpr, ctx: C) -> T;
    fn visit_equi_join_expr(&mut self, expr: &EquiJoinExpr, ctx: C) -> T;
    fn visit_theta_join_expr(&mut self, expr: &ThetaJoinExpr, ctx: C) -> T;
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
            Expr::Alias(expr) => self.visit_alias_expr(expr, ctx),
            Expr::Union(expr) => self.visit_union_expr(expr, ctx),
            Expr::Selection(expr) => self.visit_selection_expr(expr, ctx),
            Expr::Projection(expr) => self.visit_projection_expr(expr, ctx),
            Expr::EquiJoin(expr) => self.visit_equi_join_expr(expr, ctx),
            Expr::ThetaJoin(expr) => self.visit_theta_join_expr(expr, ctx),
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
    fn visit_alias_expr(&mut self, expr: &mut AliasExpr, ctx: C) -> T;
    fn visit_union_expr(&mut self, expr: &mut UnionExpr, ctx: C) -> T;
    fn visit_selection_expr(&mut self, expr: &mut SelectionExpr, ctx: C) -> T;
    fn visit_projection_expr(&mut self, expr: &mut ProjectionExpr, ctx: C) -> T;
    fn visit_equi_join_expr(&mut self, expr: &mut EquiJoinExpr, ctx: C) -> T;
    fn visit_theta_join_expr(&mut self, expr: &mut ThetaJoinExpr, ctx: C) -> T;
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
            Expr::Alias(expr) => self.visit_alias_expr(*expr, ctx),
            Expr::Union(expr) => self.visit_union_expr(*expr, ctx),
            Expr::Selection(expr) => self.visit_selection_expr(*expr, ctx),
            Expr::Projection(expr) => self.visit_projection_expr(*expr, ctx),
            Expr::EquiJoin(expr) => self.visit_equi_join_expr(*expr, ctx),
            Expr::ThetaJoin(expr) => self.visit_theta_join_expr(*expr, ctx),
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
    fn visit_alias_expr(&mut self, expr: AliasExpr, ctx: C) -> T;
    fn visit_union_expr(&mut self, expr: UnionExpr, ctx: C) -> T;
    fn visit_selection_expr(&mut self, expr: SelectionExpr, ctx: C) -> T;
    fn visit_projection_expr(&mut self, expr: ProjectionExpr, ctx: C) -> T;
    fn visit_equi_join_expr(&mut self, expr: EquiJoinExpr, ctx: C) -> T;
    fn visit_theta_join_expr(&mut self, expr: ThetaJoinExpr, ctx: C) -> T;
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
impl MemAddr for AliasExpr {}
impl MemAddr for UnionExpr {}
impl MemAddr for SelectionExpr {}
impl MemAddr for ProjectionExpr {}
impl MemAddr for EquiJoinExpr {}
impl MemAddr for ThetaJoinExpr {}
impl MemAddr for GenericIterationExpr {}
impl MemAddr for FixedPointIterationExpr {}
