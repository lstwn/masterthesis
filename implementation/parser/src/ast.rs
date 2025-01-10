pub struct Program {
    instructions: Vec<Statement>,
}

pub enum Statement {
    Query(QueryOperator),
    Mutation,
    Schema,
}

pub enum QueryOperator {
    // Binary
    // Unary
}

pub struct Intersection {}

pub struct Union {}

pub struct SetDifference {}

pub struct Projection {
    fields: Vec<()>,
}

pub struct Selection {
    condition: Expr,
}

pub struct CrossProduct {}

pub struct NaturalJoin {
    field: (),
}

pub struct ThetaJoin {
    condition: Expr,
}

/// Represents an iteration until a fix point is reached.
pub struct Iteration {
    body: QueryOperator,
}

pub enum Expr {
    Nary(Box<NaryExpr>),
    Ternary(Box<TernaryExpr>),
    Binary(Box<BinaryExpr>),
    Unary(Box<UnaryExpr>),
    Variable(Box<VarExpr>),
    Literal(Box<LitExpr>),
}

// https://ns.inria.fr/ast/sql/index.html
pub enum Operator {
    Comparison,
    Logical,
    Arithmetic,
    Set,
}

pub struct NaryExpr {
    operator: Operator,
    operands: Vec<Expr>,
}

pub struct TernaryExpr {
    operator: Operator,
    left: Expr,
    mid: Expr,
    right: Expr,
}

pub struct BinaryExpr {
    operator: Operator,
    left: Expr,
    right: Expr,
}

pub struct UnaryExpr {
    operator: Operator,
    operand: Expr,
}

pub struct VarExpr {}

pub struct LitExpr {}
