use super::scalar::ScalarTypedValue;
use crate::dbsp::OrdIndexedStream;
use dbsp::{ChildCircuit, OrdIndexedZSet, Stream};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{self, Debug, Display},
    rc::Rc,
};

pub trait Tuple {
    fn data(&self, index: usize) -> &ScalarTypedValue;
}

#[derive(
    Clone,
    Default,
    Debug,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    size_of::SizeOf,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[archive_attr(derive(Ord, Eq, PartialEq, PartialOrd))]
pub struct TupleValue {
    /// The data of the tuple which can be accessed by the index.
    /// Currently, the fields store their types alongside the data. However,
    /// this is redundant and could be removed to save space. Interestingly,
    /// `ScalarValue`, which is a union without a type tag (contrast it
    /// with `ScalarTypedValue`), has the same size, hence, the extra type tag
    /// does not increase the size currently.
    pub data: Vec<ScalarTypedValue>,
}

impl Tuple for TupleValue {
    fn data(&self, index: usize) -> &ScalarTypedValue {
        &self.data[index]
    }
}

#[derive(
    Clone,
    Default,
    Debug,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    size_of::SizeOf,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[archive_attr(derive(Ord, Eq, PartialEq, PartialOrd))]
pub struct TupleKey {
    data: Vec<ScalarTypedValue>,
}

impl Tuple for TupleKey {
    fn data(&self, index: usize) -> &ScalarTypedValue {
        &self.data[index]
    }
}

/// Currently unused.
#[derive(Debug, Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
struct Identifier {
    name: String,
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// Convenience type alias for a reference to a [`Relation`].
pub type RelationRef<Circuit = ChildCircuit<()>> = Rc<RefCell<Relation<Circuit>>>;

pub fn new_relation(name: String, schema: Schema, inner: OrdIndexedStream) -> RelationRef {
    Rc::new(RefCell::new(Relation::new(name, schema, inner)))
}

/// A [`Relation`]'s schema is a set of attributes and we store the index of each.
#[derive(Clone)]
pub struct Schema {
    pub key_attributes: HashMap<String, usize>,
    pub all_attributes: HashMap<String, usize>,
}

#[derive(Clone)]
pub struct Relation<Circuit = ChildCircuit<()>> {
    pub name: String,
    /// The schema of the relation. We need to track it on a per-relation basis
    /// because it may change during execution.
    pub schema: Schema,
    pub inner: Stream<Circuit, OrdIndexedZSet<TupleKey, TupleValue>>,
}

impl Relation {
    pub fn new(name: String, schema: Schema, inner: OrdIndexedStream) -> Self {
        Self {
            name,
            schema,
            inner,
        }
    }
    pub fn to_string_helper(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<relation {}>", self.name)
    }
}

impl Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string_helper(f)
    }
}

impl Debug for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string_helper(f)
    }
}
