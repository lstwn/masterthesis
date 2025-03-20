use super::scalar::ScalarTypedValue;
use crate::{dbsp::OrdIndexedStream, error::SyntaxError};
use dbsp::{ChildCircuit, OrdIndexedZSet, Stream};
use std::{
    cell::RefCell,
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

impl FromIterator<ScalarTypedValue> for TupleValue {
    fn from_iter<I: IntoIterator<Item = ScalarTypedValue>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
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

#[derive(Clone, Debug)]
pub struct AttributeInfo {
    /// The attribute's name.
    name: String,
    /// Whether the attribute is active, that is, not eliminated by, e.g., a projection.
    active: bool,
    // Maybe add type information here, too.
}

impl AttributeInfo {
    fn new(name: String) -> Self {
        Self { name, active: true }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl PartialEq for AttributeInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

type Index = usize;

/// A [`Relation`]'s schema is a set of attributes and we store the index of each.
#[derive(Clone, Debug)]
pub struct Schema {
    pub key_attributes: Vec<AttributeInfo>,
    pub all_attributes: Vec<AttributeInfo>,
}

impl Schema {
    pub fn new<T: Into<String>>(
        all_attributes: impl IntoIterator<Item = T>,
        key_attributes: impl IntoIterator<Item = T>,
    ) -> Result<Self, SyntaxError> {
        let all_attributes = all_attributes
            .into_iter()
            .map(|name| AttributeInfo::new(name.into()))
            .collect::<Vec<AttributeInfo>>();
        let key_attributes = key_attributes
            .into_iter()
            .map(|name| AttributeInfo::new(name.into()))
            .collect::<Vec<AttributeInfo>>();
        Ok(Self {
            key_attributes,
            all_attributes,
        })
    }
    pub fn project(&self, attributes: &Vec<&String>) -> Self {
        let mapper = |info: &AttributeInfo| {
            let mut info = info.clone();
            if !attributes.contains(&&info.name) {
                info.active = false;
            }
            info
        };
        Self {
            all_attributes: self.all_attributes.iter().map(mapper).collect(),
            key_attributes: self.key_attributes.iter().map(mapper).collect(),
        }
    }
    pub fn join(&self, other: &Self) -> Self {
        // TODO: how to handle name clashes?
        Self {
            all_attributes: self
                .all_attributes
                .iter()
                .chain(other.all_attributes.iter())
                .cloned()
                .collect(),
            key_attributes: self
                .key_attributes
                .iter()
                .chain(other.key_attributes.iter())
                .cloned()
                .collect(),
        }
    }
    pub fn active_key_fields(&self) -> impl Iterator<Item = (Index, &AttributeInfo)> {
        self.key_attributes
            .iter()
            .enumerate()
            .filter(|(_index, info)| info.active)
    }
    fn all_key_fields(&self) -> impl Iterator<Item = (Index, &AttributeInfo)> {
        self.key_attributes.iter().enumerate()
    }
    pub fn active_value_fields(&self) -> impl Iterator<Item = (Index, &AttributeInfo)> {
        self.all_attributes
            .iter()
            .enumerate()
            .filter(|(_index, info)| info.active)
    }
    pub fn value_schema_to_string(&self) -> String {
        let fields = self
            .active_value_fields()
            .map(|(_, info)| info.name.clone())
            .collect::<Vec<_>>()
            .join(" | ");
        format!("| {} |", fields)
    }
    pub fn tuple_attributes<'tuple>(
        &self,
        tuple: &'tuple TupleValue,
    ) -> impl Iterator<Item = &'tuple ScalarTypedValue> {
        self.active_value_fields()
            .map(|(index, info)| tuple.data_at(index))
    }
    pub fn tuple_to_string(&self, tuple: &TupleValue) -> String {
        let fields = self
            .tuple_attributes(tuple)
            .map(|attribute| attribute.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        format!("| {} |", fields)
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
    fn to_string_helper(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
