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
    fn data_at(&self, index: usize) -> &ScalarTypedValue;
    fn data_iter(&self) -> impl Iterator<Item = &ScalarTypedValue>;
    fn to_string_helper(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fields = self
            .data_iter()
            .map(|field| field.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        write!(f, "| {} |", fields)
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
    fn data_at(&self, index: usize) -> &ScalarTypedValue {
        &self.data[index]
    }
    fn data_iter(&self) -> impl Iterator<Item = &ScalarTypedValue> {
        self.data.iter()
    }
}

impl Display for TupleValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string_helper(f)
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
    pub data: Vec<ScalarTypedValue>,
}

impl Tuple for TupleKey {
    fn data_at(&self, index: usize) -> &ScalarTypedValue {
        &self.data[index]
    }
    fn data_iter(&self) -> impl Iterator<Item = &ScalarTypedValue> {
        self.data.iter()
    }
}

impl Display for TupleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_string_helper(f)
    }
}

/// Currently unused.
#[derive(Debug, Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
struct Identifier {
    name: String,
}

impl Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl Schema {
    // TODO: smarter schema creation.
    pub fn new(key_attributes: Vec<(String, usize)>, all_attributes: Vec<(String, usize)>) -> Self {
        Self {
            key_attributes: key_attributes.into_iter().collect(),
            all_attributes: all_attributes.into_iter().collect(),
        }
    }
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
