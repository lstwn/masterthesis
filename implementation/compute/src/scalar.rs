use crate::variable::Value;
use std::fmt::{self, Display, Formatter};

/// Stores a scalar value plus its type. These are the kinds of values that
/// can be assumed in a relation's field.
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
}

impl Default for ScalarTypedValue {
    fn default() -> Self {
        ScalarTypedValue::Null(())
    }
}

impl TryFrom<Value> for ScalarTypedValue {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(value) => Ok(ScalarTypedValue::String(value)),
            Value::Uint(value) => Ok(ScalarTypedValue::Uint(value)),
            Value::Iint(value) => Ok(ScalarTypedValue::Iint(value)),
            Value::Bool(value) => Ok(ScalarTypedValue::Bool(value)),
            Value::Null(()) => Ok(ScalarTypedValue::Null(())),
            _ => Err(()),
        }
    }
}

impl From<String> for ScalarTypedValue {
    fn from(value: String) -> Self {
        ScalarTypedValue::String(value)
    }
}

impl From<&str> for ScalarTypedValue {
    fn from(value: &str) -> Self {
        ScalarTypedValue::String(value.to_owned())
    }
}

impl From<u64> for ScalarTypedValue {
    fn from(value: u64) -> Self {
        ScalarTypedValue::Uint(value)
    }
}

impl From<i64> for ScalarTypedValue {
    fn from(value: i64) -> Self {
        ScalarTypedValue::Iint(value)
    }
}

impl From<bool> for ScalarTypedValue {
    fn from(value: bool) -> Self {
        ScalarTypedValue::Bool(value)
    }
}

impl From<()> for ScalarTypedValue {
    fn from(value: ()) -> Self {
        ScalarTypedValue::Null(value)
    }
}

impl Display for ScalarTypedValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ScalarTypedValue::String(value) => write!(f, "{value}"),
            ScalarTypedValue::Uint(value) => write!(f, "{value}"),
            ScalarTypedValue::Iint(value) => write!(f, "{value}"),
            ScalarTypedValue::Bool(value) => write!(f, "{value}"),
            ScalarTypedValue::Null(()) => write!(f, "null"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    String,
    Uint,
    Iint,
    Bool,
    Null,
}

// /// An enum of all possible scalar types.
// pub type ScalarType = Scalar<(), (), (), (), ()>;

// /// Stores a value plus its type.
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
