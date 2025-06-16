//! This module is only available if compiling with the `test` feature.
//! It provides helpers for testing and benchmarking purposes.

use crate::{
    IncDataLog,
    relation::{RelationSchema, TupleKey, TupleValue},
    scalar::ScalarTypedValue,
};
use std::{collections::HashSet, fmt::Debug, num::NonZeroUsize};

// TODO: test multithreaded runtime
pub fn setup_inc_data_log() -> IncDataLog {
    IncDataLog::new(NonZeroUsize::try_from(4).unwrap(), true)
}

pub trait InputEntity: Into<TupleKey> + Into<TupleValue> + Clone + Debug {
    fn schema() -> RelationSchema;
}

#[derive(Clone, Debug)]
pub struct Person {
    person_id: u64,
    name: String,
    age: u64,
    profession_id: u64,
}

impl Person {
    pub fn data() -> impl IntoIterator<Item = Person> {
        [
            Person {
                person_id: 0,
                name: "Alice".to_string(),
                age: 20,
                profession_id: 0,
            },
            Person {
                person_id: 1,
                name: "Bob".to_string(),
                age: 30,
                profession_id: 1,
            },
            Person {
                person_id: 2,
                name: "Charlie".to_string(),
                age: 40,
                profession_id: 0,
            },
        ]
    }
}

impl InputEntity for Person {
    fn schema() -> RelationSchema {
        RelationSchema::new(
            "person",
            ["person_id", "name", "age", "profession_id"],
            ["person_id"],
        )
        .expect("Correct schema definition")
    }
}

impl From<Person> for TupleKey {
    fn from(person: Person) -> Self {
        TupleKey {
            data: vec![ScalarTypedValue::Uint(person.person_id)],
        }
    }
}

impl From<Person> for TupleValue {
    fn from(person: Person) -> Self {
        TupleValue {
            data: vec![
                ScalarTypedValue::Uint(person.person_id),
                ScalarTypedValue::String(person.name),
                ScalarTypedValue::Uint(person.age),
                ScalarTypedValue::Uint(person.profession_id),
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Profession {
    profession_id: u64,
    name: String,
}

impl Profession {
    pub fn data() -> impl IntoIterator<Item = Profession> {
        [
            Profession {
                profession_id: 0,
                name: "Engineer".to_string(),
            },
            Profession {
                profession_id: 1,
                name: "Doctor".to_string(),
            },
        ]
    }
}

impl InputEntity for Profession {
    fn schema() -> RelationSchema {
        RelationSchema::new("profession", ["profession_id", "name"], ["profession_id"])
            .expect("Correct schema definition")
    }
}

impl From<Profession> for TupleKey {
    fn from(profession: Profession) -> Self {
        TupleKey {
            data: vec![ScalarTypedValue::Uint(profession.profession_id)],
        }
    }
}

impl From<Profession> for TupleValue {
    fn from(profession: Profession) -> Self {
        TupleValue {
            data: vec![
                ScalarTypedValue::Uint(profession.profession_id),
                ScalarTypedValue::String(profession.name),
            ],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Edge {
    from: u64,
    to: u64,
    weight: u64,
    active: bool,
}

impl Edge {
    pub fn new(from: u64, to: u64, weight: u64) -> Self {
        Self {
            from,
            to,
            weight,
            active: true,
        }
    }
}

impl InputEntity for Edge {
    fn schema() -> RelationSchema {
        RelationSchema::new("edges", ["from", "to", "weight", "active"], ["from", "to"])
            .expect("Correct schema definition")
    }
}

impl From<Edge> for TupleKey {
    fn from(edge: Edge) -> Self {
        TupleKey {
            data: vec![
                ScalarTypedValue::Uint(edge.from),
                ScalarTypedValue::Uint(edge.to),
            ],
        }
    }
}

impl From<Edge> for TupleValue {
    fn from(edge: Edge) -> Self {
        TupleValue {
            data: vec![
                ScalarTypedValue::Uint(edge.from),
                ScalarTypedValue::Uint(edge.to),
                ScalarTypedValue::Uint(edge.weight),
                ScalarTypedValue::Bool(edge.active),
            ],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SetOp {
    node_id: u64,
    counter: u64,
    key: u64,
    value: u64,
}

impl SetOp {
    pub fn new(node_id: u64, counter: u64, key: u64, value: u64) -> Self {
        Self {
            node_id,
            counter,
            key,
            value,
        }
    }
}

impl InputEntity for SetOp {
    fn schema() -> RelationSchema {
        RelationSchema::new(
            "set",
            ["NodeId", "Counter", "Key", "Value"],
            ["NodeId", "Counter"],
        )
        .expect("Correct schema definition")
    }
}

impl From<SetOp> for TupleKey {
    fn from(set_op: SetOp) -> Self {
        TupleKey::from_iter([set_op.node_id, set_op.counter])
    }
}

impl From<SetOp> for TupleValue {
    fn from(set_op: SetOp) -> Self {
        TupleValue::from_iter([set_op.node_id, set_op.counter, set_op.key, set_op.value])
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PredRel {
    from_node_id: u64,
    from_counter: u64,
    to_node_id: u64,
    to_counter: u64,
}

impl PredRel {
    pub fn new(from_node_id: u64, from_counter: u64, to_node_id: u64, to_counter: u64) -> Self {
        Self {
            from_node_id,
            from_counter,
            to_node_id,
            to_counter,
        }
    }
}

impl InputEntity for PredRel {
    fn schema() -> RelationSchema {
        RelationSchema::new(
            "pred",
            ["FromNodeId", "FromCounter", "ToNodeId", "ToCounter"],
            ["FromNodeId", "FromCounter", "ToNodeId", "ToCounter"],
        )
        .expect("Correct schema definition")
    }
}

impl From<PredRel> for TupleKey {
    fn from(pred_rel: PredRel) -> Self {
        TupleKey::from_iter([
            pred_rel.from_node_id,
            pred_rel.from_counter,
            pred_rel.to_node_id,
            pred_rel.to_counter,
        ])
    }
}

impl From<PredRel> for TupleValue {
    fn from(pred_rel: PredRel) -> Self {
        TupleValue::from_iter([
            pred_rel.from_node_id,
            pred_rel.from_counter,
            pred_rel.to_node_id,
            pred_rel.to_counter,
        ])
    }
}

pub struct Replica {
    node_id: u64,
    /// Always points to the next unused counter value.
    counter: u64,
    heads: HashSet<(u64, u64)>,
}

impl Replica {
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id,
            counter: 0,
            heads: HashSet::new(),
        }
    }
    /// Warning: This method assumes that the `set_op` and its causal predecessors
    /// are all causally ready for this replica. There is no buffering due to
    /// missing causal predecessors for now.
    pub fn ack_remote_set_op(&mut self, set_op: &SetOp, pred_rels: &[PredRel]) {
        // We ensure that the set_op is not from this replica.
        debug_assert!(set_op.node_id != self.node_id);
        // We ensure that the pred_rels are all related to the set_op.
        debug_assert!(
            pred_rels
                .iter()
                .all(|pred_rel| pred_rel.to_node_id == set_op.node_id
                    && pred_rel.to_counter == set_op.counter)
        );
        for pred_rel in pred_rels {
            let value = (pred_rel.from_node_id, pred_rel.from_counter);
            // We remove all heads that are now subsumed by the new set_op.
            if self.heads.contains(&value) {
                self.heads.remove(&value);
            }
        }
        self.heads.insert((set_op.node_id, set_op.counter));
        // Advance the lamport clock if the new set_op has a higher counter.
        if self.counter < set_op.counter {
            self.counter = set_op.counter + 1;
        }
    }
    pub fn new_local_set_op(&mut self, key: u64, value: u64) -> (SetOp, Vec<PredRel>) {
        let set_op = SetOp::new(self.node_id, self.counter, key, value);
        let pred_rels = self
            .heads
            .iter()
            .map(|&(from_node_id, from_counter)| {
                let to_node_id = set_op.node_id;
                let to_counter = set_op.counter;
                PredRel::new(from_node_id, from_counter, to_node_id, to_counter)
            })
            .collect::<Vec<_>>();
        self.heads.clear();
        self.heads.insert((set_op.node_id, set_op.counter));
        self.counter += 1;
        (set_op, pred_rels)
    }
}
