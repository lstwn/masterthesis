use std::collections::HashMap;

use super::scalar::{ScalarType, ScalarTypedValue};

#[derive(Debug, Hash, Eq, PartialEq, Clone, PartialOrd, Ord)]
struct Identifier {
    name: String,
}

type Index = usize;

struct TupleType {
    // TODO: How to allow nullable fields?
    info: HashMap<Identifier, (ScalarType, Index)>,
}

impl TupleType {
    pub fn new(mut fields: Vec<(Identifier, ScalarType)>) -> Self {
        // NOTE: If the same identifier is used multiple times, the first
        // occurrence will be used.
        fields.dedup_by_key(|(identifier, _)| identifier.clone());
        let info = fields
            .into_iter()
            .enumerate()
            .map(|(index, (identifier, scalar_type))| (identifier, (scalar_type, index)))
            .collect();
        Self { info }
    }
    pub fn get(&self, identifier: &Identifier) -> Option<(ScalarType, Index)> {
        self.info.get(identifier).copied()
    }
}

struct TupleValue {
    /// The data of the tuple which can be accessed by the index.
    /// Currently, the fields store their types alongside the data. However,
    /// this is redundant and could be removed to save space. Interestingly,
    /// `ScalarValue` which is a union without a type tag (contrast it
    /// with `ScalarTypedValue`) has the same size, hence, the extra type tag
    /// does not increase the size currently.
    data: Vec<ScalarTypedValue>,
}

impl TupleValue {
    // TODO: Create with data. How to handle `null`?
    fn new() -> Self {
        Self { data: Vec::new() }
    }
    fn with_capacity(capacity: Index) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
    fn get(&self, index: Index) -> &ScalarTypedValue {
        &self.data[index]
    }
    fn get_mut(&mut self, index: Index) -> &mut ScalarTypedValue {
        &mut self.data[index]
    }
    fn set(&mut self, index: Index, value: ScalarTypedValue) {
        self.data[index] = value;
    }
}
