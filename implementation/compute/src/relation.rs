use super::scalar::ScalarTypedValue;
use std::fmt::Display;

#[derive(Debug, Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
struct Identifier {
    name: String,
}
impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
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

// /// An enum of all possible types a field of a relation can assume.
// #[derive(Debug, Copy, Clone, PartialEq, Eq)]
// pub enum ScalarType {
//     /// String.
//     String,
//     /// Unsigned integer value of 64 bits.
//     Uint,
//     /// Signed integer value of 64 bits.
//     Iint,
//     /// Boolean.
//     Bool,
// }

// /// The type of a field of a relation. May be null.
// pub struct RelationScalarType {
//     scalar_type: ScalarType,
//     nullable: bool,
// }

// struct RelationType {
//     name: String,
//     // TODO: How to allow nullable fields?
//     info: HashMap<Identifier, (ScalarType, Index)>,
// }

// impl RelationType {
//     pub fn new(name: String, mut fields: Vec<(Identifier, ScalarType)>) -> Self {
//         // NOTE: If the same identifier is used multiple times, the first
//         // occurrence will be used.
//         fields.dedup_by_key(|(identifier, _)| identifier.clone());
//         let info = fields
//             .into_iter()
//             .enumerate()
//             .map(|(index, (identifier, scalar_type))| (identifier, (scalar_type, index)))
//             .collect();
//         Self { name, info }
//     }
//     pub fn name(&self) -> &str {
//         &self.name
//     }
//     pub fn get(&self, identifier: &Identifier) -> Option<(ScalarType, Index)> {
//         self.info.get(identifier).copied()
//     }
//     pub fn fields(&self) -> impl Iterator<Item = (&Identifier, ScalarType, Index)> {
//         self.info
//             .iter()
//             .map(|(identifier, (scalar_type, index))| (identifier, *scalar_type, *index))
//     }
// }
