#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Operator {
    // Comparison operations.
    /// Eagerly-evaluated binary operation.
    Equal,
    /// Eagerly-evaluated binary operation.
    NotEqual,
    /// Eagerly-evaluated binary operation.
    Less,
    /// Eagerly-evaluated binary operation.
    LessThan,
    /// Eagerly-evaluated binary operation.
    Greater,
    /// Eagerly-evaluated binary operation.
    GreaterThan,

    // Logical operations on booleans (and values coerced into booleans).
    /// Lazily-evaluated binary operation.
    And,
    /// Lazily-evaluated binary operation.
    Or,
    /// Eagerly-evaluated unary operation.
    Not,

    // Arithmetic operations on numbers.
    /// Eagerly-evaluated binary operation.
    Addition,
    /// Eagerly-evaluated binary operation or eagerly-evaluated unary operation.
    Subtraction,
    /// Eagerly-evaluated binary operation.
    Multiplication,
    /// Eagerly-evaluated binary operation.
    Division,

    // Operations on sets/relations.
    /// Eagerly-evaluated binary operation.
    Union,
    /// Eagerly-evaluated binary operation.
    Intersection,
    /// Eagerly-evaluated binary operation.
    Difference,

    // Operations on relations.
    /// Eagerly-evaluated binary operation.
    Projection,
    /// Eagerly-evaluated binary operation.
    Selection,
    /// Eagerly-evaluated binary operation.
    CrossProduct,
    /// Eagerly-evaluated ternary operation.
    NaturalJoin,
    /// Eagerly-evaluated ternary operation.
    ThetaJoin,
    /// Eagerly-evaluated unary operation.
    Iteration,
}
