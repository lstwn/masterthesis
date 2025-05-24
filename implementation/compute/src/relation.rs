use super::scalar::ScalarTypedValue;
use crate::{dbsp::StreamWrapper, error::SyntaxError};
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::{self, Debug, Display},
    rc::Rc,
};

pub trait Tuple: FromIterator<ScalarTypedValue> {
    fn empty() -> Self {
        Self::from_iter(vec![])
    }
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
    pub fn all_fields(&self) -> impl Iterator<Item = &'a ScalarTypedValue> {
        self.schema
            .all_fields()
            .map(|(index, _info)| self.tuple.data_at(index))
    }
    pub fn named_fields(
        &self,
        alias: &Option<String>,
    ) -> impl Iterator<Item = (String, ScalarTypedValue)> {
        self.schema
            .active_fields()
            .map(|(index, info)| (info.name(alias), self.tuple.data_at(index).clone()))
    }
    pub fn coalesce(&self) -> impl Iterator<Item = ScalarTypedValue> {
        self.schema
            .active_fields()
            .map(|(index, info)| self.tuple.data_at(index).clone())
    }
    pub fn pick(&self, fields: &[String]) -> impl Iterator<Item = ScalarTypedValue> {
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

#[macro_export]
macro_rules! tuple {
    ( $( $key:expr ),* $(,)?) => {{
        let tuple = [$( ScalarTypedValue::from($key) ),*];
        TupleValue {
            data: tuple.to_vec(),
        }
    }};
}

impl<T: Into<ScalarTypedValue>> FromIterator<T> for TupleValue {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().map(|v| v.into()).collect(),
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

impl<T: Into<ScalarTypedValue>> FromIterator<T> for TupleKey {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().map(|v| v.into()).collect(),
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
pub type RelationRef = Rc<RefCell<Relation>>;

pub fn new_relation<T: Into<StreamWrapper>>(schema: RelationSchema, inner: T) -> RelationRef {
    Rc::new(RefCell::new(Relation::new(schema, inner)))
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
        let name = alias
            .as_ref()
            .map(|alias| format!("{}.{}", alias, self.name))
            .unwrap_or_else(|| self.name.clone());
        if self.active {
            name
        } else {
            format!("{}*", name)
        }
    }
}

type Index = usize;

#[derive(Clone, PartialEq, Eq)]
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
    pub fn empty() -> Self {
        Self { fields: vec![] }
    }
    /// Only the active fields are included in the count.
    pub fn len(&self) -> usize {
        self.fields.iter().filter(|info| info.active).count()
    }
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    /// Includes the active and inactive fields in the count.
    pub fn full_len(&self) -> usize {
        self.fields.len()
    }
    fn is_coalesced(&self) -> bool {
        !self.fields.iter().any(|info| !info.active)
    }
    fn coalesce(&self) -> Self {
        self.fields
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect()
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
    pub fn all_field_names(&self, alias: &Option<String>) -> impl Iterator<Item = String> {
        self.all_fields().map(|(_index, info)| info.name(alias))
    }
    fn select(&self) -> Self {
        self.clone()
    }
    /// We mark all fields as inactive, that is, we forget about them.
    fn forget(&self) -> Self {
        self.fields
            .iter()
            .map(|info| FieldInfo {
                name: info.name.clone(),
                active: false,
            })
            .collect()
    }
    /// In contrast to the `project` method, this method does not remove fields
    /// from the schema but marks them as inactive, thereby not coalescing the
    /// schema and the order of fields. Optionally, you can rename a field by
    /// providing an alias/new name/target name as a second element.
    fn pick(&self, fields: &Vec<(&String, Option<&String>)>) -> Self {
        // For keeping track of duplicated field names.
        let mut active = HashSet::with_capacity(fields.len());
        // Don't use active_fields() here because the tuple is not coalesced
        // but we only allow to pick from the set of active fields though.
        self.all_fields()
            .map(|(_index, info)| {
                // We do not reactivate already inactive fields.
                if !info.active {
                    return info.clone();
                }
                if let Some((source_name, target_name)) =
                    fields.iter().find(|field| *field.0 == info.name)
                {
                    let name = target_name.cloned().unwrap_or_else(|| info.name.clone());
                    if !active.contains(&name) {
                        active.insert(name.clone());
                        return FieldInfo::new(name); // Field is active by constructor.
                    } else {
                        // We have a duplicated field name, so we mark it as inactive.
                        return FieldInfo {
                            name,
                            active: false,
                        };
                    }
                }
                // Field is not in the list of fields to pick, so we mark it as inactive.
                FieldInfo {
                    name: info.name.clone(),
                    active: false,
                }
            })
            .collect()
    }
    /// In case of a full projection, we coalesce the schema and remove inactive
    /// fields. The order is also redefined according to the projection.
    fn project(&self, fields: Vec<String>) -> Self {
        fields.into_iter().collect()
    }
    fn join(&self, other: &Self) -> Self {
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
            .map(|(_, info)| info.name(&None))
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
            fields: iter.into_iter().map(FieldInfo::new).collect(),
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
    pub fn is_coalesced(&self) -> bool {
        self.key.is_coalesced() && self.tuple.is_coalesced()
    }
    pub fn coalesce(&self) -> Self {
        Self {
            name: format!("{}-coalesced", self.name),
            key: self.key.coalesce(),
            tuple: self.tuple.coalesce(),
        }
    }
    /// Just clones the current schema, as selections do not alter the schema.
    pub fn select(&self) -> Self {
        Self {
            name: format!("{}-selected", self.name),
            key: self.key.clone(),
            tuple: self.tuple.clone(),
        }
    }
    pub fn pick(&self, fields: &Vec<(&String, Option<&String>)>) -> Self {
        Self {
            name: format!("{}-picked", self.name),
            // To keep the `ProjectionExpr`'s semantics consistent,
            // we erase the key here, too, as we do for the full projection below.
            key: self.key.forget(),
            tuple: self.tuple.pick(fields),
        }
    }
    pub fn project(&self, fields: Vec<String>) -> Self {
        Self {
            name: format!("{}-projected", self.name),
            key: TupleSchema::empty(),
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

impl PartialEq for RelationSchema {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.tuple == other.tuple
    }
}

impl Eq for RelationSchema {}

#[derive(Clone)]
pub struct Relation {
    /// The schema of the relation. We need to track it on a per-relation basis
    /// because it may change during execution.
    pub schema: RelationSchema,
    pub inner: StreamWrapper,
}

impl Relation {
    pub fn new<T: Into<StreamWrapper>>(schema: RelationSchema, inner: T) -> Self {
        Self {
            schema,
            inner: inner.into(),
        }
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
