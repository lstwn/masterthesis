#[derive(Debug, Copy, Clone)]
pub enum Scalar<String, Uint, Iint, Bool, Null> {
    /// String.
    String(String),
    /// Unsigned integer value of 64 bits.
    Uint(Uint),
    /// Signed integer value of 64 bits.
    Iint(Iint),
    /// Boolean.
    Bool(Bool),
    /// Null.
    Null(Null),
}

/// An enum of all possible scalar types.
pub type ScalarType = Scalar<(), (), (), (), ()>;

/// Prefixed with `D` to distinguish from Rust's built-in types.
pub type DString = String;
/// Prefixed with `D` to distinguish from Rust's built-in types.
pub type DUint = u64;
/// Prefixed with `D` to distinguish from Rust's built-in types.
pub type DIint = i64;
/// Prefixed with `D` to distinguish from Rust's built-in types.
pub type DBool = bool;
/// Prefixed with `D` to distinguish from Rust's built-in types.
pub type DNull = ();

/// In contrast to `ScalarTypedValue` it only stores a value but not its type.
/// Currently unused.
union ScalarValue {
    string: std::mem::ManuallyDrop<DString>,
    uint: DUint,
    iint: DIint,
    bool: DBool,
    null: DNull,
}

/// Stores a value plus its type.
pub type ScalarTypedValue = Scalar<DString, DUint, DIint, DBool, DNull>;

impl Default for ScalarTypedValue {
    fn default() -> Self {
        Scalar::Null(())
    }
}
