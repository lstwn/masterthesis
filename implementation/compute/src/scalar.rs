use std::fmt::{self, Display, Formatter};

/// Stores a value plus its type.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    size_of::SizeOf,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[archive_attr(derive(Ord, Eq, PartialEq, PartialOrd))]
pub enum ScalarTypedValue {
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
    // QUESTION: Add `TupleValue` variant? Or push its attributes as scalars?
}

impl Default for ScalarTypedValue {
    fn default() -> Self {
        ScalarTypedValue::Null(())
    }
}

impl Display for ScalarTypedValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ScalarTypedValue::String(value) => write!(f, "{}", value),
            ScalarTypedValue::Uint(value) => write!(f, "{}", value),
            ScalarTypedValue::Iint(value) => write!(f, "{}", value),
            ScalarTypedValue::Bool(value) => write!(f, "{}", value),
            ScalarTypedValue::Null(()) => write!(f, "null"),
        }
    }
}

/// An enum of all possible scalar types.
// pub type ScalarType = Scalar<(), (), (), (), ()>;

/// Stores a value plus its type.
// pub type ScalarTypedValue = Scalar<DString, DUint, DIint, DBool, DNull>;

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
/// Currently unused, as it has the same binary size as `ScalarTypedValue` due
/// to some smartness of the Rust compiler.
#[deprecated(note = "currently unused")]
union ScalarValue {
    string: std::mem::ManuallyDrop<DString>,
    uint: DUint,
    iint: DIint,
    bool: DBool,
    null: DNull,
}
