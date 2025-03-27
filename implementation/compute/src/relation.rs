use super::scalar::ScalarTypedValue;
use crate::{dbsp::OrdIndexedStream, error::SyntaxError};
use dbsp::{ChildCircuit, OrdIndexedZSet, Stream};
use std::{
    cell::RefCell,
    fmt::{self, Debug, Display},
    rc::Rc,
};

pub trait Tuple: FromIterator<ScalarTypedValue> {
    fn data_at(&self, index: usize) -> &ScalarTypedValue;
    /// Iterates over _all_ stored fields of the tuple,
    /// regardless if they are part of the current schema.
    fn data(&self) -> impl Iterator<Item = &ScalarTypedValue>;
    /// Assumes that the passed indexes are valid for the tuple.
    fn data_to_string(&self) -> String {
        let fields = self
            .data()
            .map(|field| field.to_string())
            .collect::<Vec<_>>()
            .join(" | ");
        format!("| {} |", fields)
    }
}

pub struct SchemaTuple<'a, T> {
    schema: &'a TupleSchema,
    tuple: &'a T,
}

impl<'a, T: Tuple> SchemaTuple<'a, T> {
    pub fn new(schema: &'a TupleSchema, tuple: &'a T) -> Self {
        Self { schema, tuple }
    }
    pub fn fields(&self) -> impl Iterator<Item = &'a ScalarTypedValue> {
        self.schema
            .active_fields()
            .map(|(index, info)| self.tuple.data_at(index))
    }
    pub fn named_fields(
        &self,
        alias: &Option<String>,
    ) -> impl Iterator<Item = (String, ScalarTypedValue)> {
        self.schema
            .active_fields()
            .map(|(index, info)| (info.name(alias), self.tuple.data_at(index).clone()))
    }
    pub fn pick(&self, fields: &Vec<String>) -> impl Iterator<Item = ScalarTypedValue> {
        self.schema.active_fields().filter_map(|(index, info)| {
            if fields.contains(&info.name) {
                Some(self.tuple.data_at(index).clone())
            } else {
                None
            }
        })
    }
    pub fn join(&self, other: &Self) -> impl Iterator<Item = ScalarTypedValue> {
        self.fields().chain(other.fields()).cloned()
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
    fn data(&self) -> impl Iterator<Item = &ScalarTypedValue> {
        self.data.iter()
    }
}

impl Display for TupleValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data_to_string())
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

impl FromIterator<ScalarTypedValue> for TupleKey {
    fn from_iter<I: IntoIterator<Item = ScalarTypedValue>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
}

impl Tuple for TupleKey {
    fn data_at(&self, index: usize) -> &ScalarTypedValue {
        &self.data[index]
    }
    fn data(&self) -> impl Iterator<Item = &ScalarTypedValue> {
        self.data.iter()
    }
}

impl Display for TupleKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data_to_string())
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

pub fn new_relation(schema: RelationSchema, inner: OrdIndexedStream) -> RelationRef {
    Rc::new(RefCell::new(Relation::new(schema, inner)))
}

#[derive(Clone, Debug)]
pub struct FieldInfo {
    /// The field's name.
    name: String,
    /// Whether the field is active, that is, not eliminated by, e.g., a projection.
    active: bool,
    // Maybe add type information here, too.
}

impl FieldInfo {
    fn new(name: String) -> Self {
        Self { name, active: true }
    }
    fn name(&self, alias: &Option<String>) -> String {
        alias
            .as_ref()
            .map(|alias| format!("{}.{}", alias, self.name))
            .unwrap_or_else(|| self.name.clone())
    }
}

impl PartialEq for FieldInfo {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

type Index = usize;

#[derive(Clone)]
pub struct TupleSchema {
    fields: Vec<FieldInfo>,
}

impl TupleSchema {
    pub fn new<T: Into<String>>(fields: impl IntoIterator<Item = T>) -> Self {
        Self {
            fields: fields
                .into_iter()
                .map(|name| FieldInfo::new(name.into()))
                .collect(),
        }
    }
    fn active_fields(&self) -> impl Iterator<Item = (Index, &FieldInfo)> {
        self.fields
            .iter()
            .enumerate()
            .filter(|(_index, info)| info.active)
    }
    fn all_fields(&self) -> impl Iterator<Item = (Index, &FieldInfo)> {
        self.fields.iter().enumerate()
    }
    pub fn field_names(&self, alias: &Option<String>) -> impl Iterator<Item = String> {
        self.active_fields().map(|(_index, info)| info.name(alias))
    }
    pub fn select(&self) -> Self {
        self.clone()
    }
    /// In contrast to the `project` method, this method does not remove fields
    /// from the schema but marks them as inactive, thereby not coalescing the
    /// schema and the order of fields.
    pub fn pick(&self, fields: &Vec<String>) -> Self {
        self.all_fields()
            .map(|(_index, info)| {
                let mut info = info.clone();
                if info.active {
                    if fields.contains(&info.name) {
                        info.active = true;
                    } else {
                        info.active = false;
                    }
                }
                info
            })
            .collect()
    }
    /// In case of a full projection, we coalesce the schema and remove inactive
    /// fields. The order is also redefined according to the projection.
    pub fn project(&self, fields: Vec<String>) -> Self {
        fields.into_iter().collect()
    }
    pub fn join(&self, other: &Self) -> Self {
        self.active_fields()
            .chain(other.active_fields())
            .map(|(_index, info)| info)
            .cloned()
            .collect()
    }
    fn fields_to_string<'a>(
        &self,
        fields: impl Iterator<Item = (Index, &'a FieldInfo)>,
        with_extra: bool,
    ) -> String {
        let fields = fields
            .map(|(_, info)| {
                if with_extra {
                    if info.active {
                        info.name.clone()
                    } else {
                        format!("[{}]", info.name)
                    }
                } else {
                    info.name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!("| {} |", fields)
    }
}

impl FromIterator<FieldInfo> for TupleSchema {
    fn from_iter<I: IntoIterator<Item = FieldInfo>>(iter: I) -> Self {
        Self {
            fields: iter.into_iter().collect(),
        }
    }
}

impl FromIterator<String> for TupleSchema {
    fn from_iter<I: IntoIterator<Item = String>>(iter: I) -> Self {
        Self {
            fields: iter.into_iter().map(|name| FieldInfo::new(name)).collect(),
        }
    }
}

impl Debug for TupleSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.fields_to_string(self.all_fields(), true))
    }
}

impl Display for TupleSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.fields_to_string(self.active_fields(), false))
    }
}

/// A [`Relation`]'s schema is a set of fields and we store the index of each.
#[derive(Clone, Debug)]
pub struct RelationSchema {
    /// Not a real name to reference the relation but more like a transformation trace.
    /// Real names are handled by variable names.
    pub name: String,
    pub key: TupleSchema,
    pub tuple: TupleSchema,
}

impl RelationSchema {
    pub fn new<T: Into<String>>(
        name: T,
        tuple_fields: impl IntoIterator<Item = T>,
        key_fields: impl IntoIterator<Item = T>,
    ) -> Result<Self, SyntaxError> {
        Ok(Self {
            name: name.into(),
            key: TupleSchema::new(key_fields),
            tuple: TupleSchema::new(tuple_fields),
        })
    }
    /// Just clones the current schema, as selections do not alter the schema.
    pub fn select(&self) -> Self {
        Self {
            name: format!("{}-selected", self.name),
            key: self.key.clone(),
            tuple: self.tuple.clone(),
        }
    }
    pub fn pick(&self, fields: &Vec<String>) -> Self {
        Self {
            name: format!("{}-picked", self.name),
            // We leave the keys as they are.
            key: self.key.clone(),
            tuple: self.tuple.pick(fields),
        }
    }
    pub fn project(&self, fields: Vec<String>) -> Self {
        Self {
            name: format!("{}-projected", self.name),
            // We leave the keys as they are.
            key: self.key.clone(),
            tuple: self.tuple.project(fields),
        }
    }
    pub fn join(&self, other: &Self, key_fields: impl IntoIterator<Item = String>) -> Self {
        Self {
            name: format!("{}-{}-joined", self.name, other.name),
            key: key_fields.into_iter().collect(),
            tuple: self.tuple.join(&other.tuple),
        }
    }
}

impl Display for RelationSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<relation {}>", self.name)
    }
}

#[derive(Clone)]
pub struct Relation<Circuit = ChildCircuit<()>> {
    /// The schema of the relation. We need to track it on a per-relation basis
    /// because it may change during execution.
    pub schema: RelationSchema,
    pub inner: Stream<Circuit, OrdIndexedZSet<TupleKey, TupleValue>>,
}

impl Relation {
    pub fn new(schema: RelationSchema, inner: OrdIndexedStream) -> Self {
        Self { schema, inner }
    }
}

impl Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.schema)
    }
}

impl Debug for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.schema)
    }
}
