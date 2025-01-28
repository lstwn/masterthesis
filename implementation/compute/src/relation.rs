use super::scalar::ScalarTypedValue;
use crate::error::RuntimeError;
use std::{collections::HashMap, fmt::Display};

#[derive(Debug, Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
struct Identifier {
    name: String,
}
impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

type Index = usize;

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

struct TupleValue {
    /// The data of the tuple which can be accessed by the index.
    /// Currently, the fields store their types alongside the data. However,
    /// this is redundant and could be removed to save space. Interestingly,
    /// `ScalarValue`, which is a union without a type tag (contrast it
    /// with `ScalarTypedValue`), has the same size, hence, the extra type tag
    /// does not increase the size currently.
    data: HashMap<Identifier, ScalarTypedValue>,
}

impl TupleValue {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    fn with_capacity(capacity: Index) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
        }
    }
    pub fn get(&self, attr: &Identifier) -> Result<&ScalarTypedValue, RuntimeError> {
        self.data.get(attr).ok_or(RuntimeError::new(format!(
            "Relation does not have attribute {}",
            attr
        )))
    }
    pub fn get_mut(&mut self, attr: &Identifier) -> Result<&mut ScalarTypedValue, RuntimeError> {
        self.data.get_mut(attr).ok_or(RuntimeError::new(format!(
            "Relation does not have attribute {}",
            attr
        )))
    }
    pub fn set(&mut self, attr: &Identifier, value: ScalarTypedValue) -> Result<(), RuntimeError> {
        self.data.insert(attr.clone(), value);
        Ok(())
    }
}
