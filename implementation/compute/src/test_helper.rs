//! This module is only available if compiling with the `test` feature.
//! It provides helpers for testing and benchmarking purposes.

use crate::{
    IncDataLog,
    relation::{RelationSchema, TupleKey, TupleValue},
    scalar::ScalarTypedValue,
    tuple,
};
use std::{collections::HashSet, fmt::Debug, num::NonZeroUsize};

pub fn setup_inc_data_log() -> IncDataLog {
    IncDataLog::new(NonZeroUsize::try_from(1).unwrap(), true)
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

pub fn person_profession_data() -> [(Vec<Person>, Vec<Profession>); 1] {
    [(
        vec![
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
        ],
        vec![
            Profession {
                profession_id: 0,
                name: "Engineer".to_string(),
            },
            Profession {
                profession_id: 1,
                name: "Doctor".to_string(),
            },
        ],
    )]
}

#[derive(Copy, Clone, Debug)]
pub struct PlainRelation {
    a: u64,
    b: u64,
    c: u64,
}

impl PlainRelation {
    pub fn new(a: u64, b: u64, c: u64) -> Self {
        Self { a, b, c }
    }
    const STEPS: usize = 1;
    pub fn test_data_1() -> [Vec<PlainRelation>; Self::STEPS] {
        [vec![
            PlainRelation::new(1, 2, 3),
            PlainRelation::new(4, 5, 6),
            PlainRelation::new(7, 8, 9),
        ]]
    }
    pub fn test_data_2() -> [Vec<PlainRelation>; Self::STEPS] {
        [vec![PlainRelation::new(1, 2, 3)]]
    }
    pub fn test_data_3() -> [Vec<PlainRelation>; Self::STEPS] {
        [vec![PlainRelation::new(4, 5, 6)]]
    }
}

impl InputEntity for PlainRelation {
    fn schema() -> RelationSchema {
        RelationSchema::new("plain", ["a", "b", "c"], []).expect("Correct schema definition")
    }
}

impl From<PlainRelation> for TupleKey {
    fn from(fact: PlainRelation) -> Self {
        TupleKey { data: vec![] }
    }
}

impl From<PlainRelation> for TupleValue {
    fn from(fact: PlainRelation) -> Self {
        TupleValue {
            data: vec![
                ScalarTypedValue::Uint(fact.a),
                ScalarTypedValue::Uint(fact.b),
                ScalarTypedValue::Uint(fact.c),
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

/// This relation represents set operations on a key-value store used in the
/// [`parser::crdts::mvr_crdt_store_datalog`] and [`parser::crdts::mvr_store_datalog`]
/// Datalog programs.
#[derive(Copy, Clone, Debug)]
pub struct SetOp {
    rep_id: u64,
    ctr: u64,
    key: u64,
    value: u64,
}

impl SetOp {
    pub fn new(rep_id: u64, ctr: u64, key: u64, value: u64) -> Self {
        Self {
            rep_id,
            ctr,
            key,
            value,
        }
    }
}

impl InputEntity for SetOp {
    fn schema() -> RelationSchema {
        RelationSchema::new("set", ["RepId", "Ctr", "Key", "Value"], ["RepId", "Ctr"])
            .expect("Correct schema definition")
    }
}

impl From<SetOp> for TupleKey {
    fn from(set_op: SetOp) -> Self {
        TupleKey::from_iter([set_op.rep_id, set_op.ctr])
    }
}

impl From<SetOp> for TupleValue {
    fn from(set_op: SetOp) -> Self {
        TupleValue::from_iter([set_op.rep_id, set_op.ctr, set_op.key, set_op.value])
    }
}

/// This relation represents a causal predecessor relationship used in the
/// [`parser::crdts::mvr_crdt_store_datalog`] and [`parser::crdts::mvr_store_datalog`]
/// Datalog programs.
#[derive(Copy, Clone, Debug)]
pub struct PredRel {
    from_rep_id: u64,
    from_ctr: u64,
    to_rep_id: u64,
    to_ctr: u64,
}

impl PredRel {
    pub fn new(from_rep_id: u64, from_ctr: u64, to_rep_id: u64, to_ctr: u64) -> Self {
        Self {
            from_rep_id,
            from_ctr,
            to_rep_id,
            to_ctr,
        }
    }
}

impl InputEntity for PredRel {
    fn schema() -> RelationSchema {
        RelationSchema::new(
            "pred",
            ["FromRepId", "FromCtr", "ToRepId", "ToCtr"],
            ["FromRepId", "FromCtr", "ToRepId", "ToCtr"],
        )
        .expect("Correct schema definition")
    }
}

impl From<PredRel> for TupleKey {
    fn from(pred_rel: PredRel) -> Self {
        TupleKey::from_iter([
            pred_rel.from_rep_id,
            pred_rel.from_ctr,
            pred_rel.to_rep_id,
            pred_rel.to_ctr,
        ])
    }
}

impl From<PredRel> for TupleValue {
    fn from(pred_rel: PredRel) -> Self {
        TupleValue::from_iter([
            pred_rel.from_rep_id,
            pred_rel.from_ctr,
            pred_rel.to_rep_id,
            pred_rel.to_ctr,
        ])
    }
}

/// This function returns test data for an operation history of the MVR CRDT store.
/// The history is as follows.
/// The notation is `set_<replica_id>_<counter>(<key>, <value>)`.
///
/// 1. step (just one root operation setting register with key 1 to value 1):
///
/// ```text
/// set_0_0(1, 1)
/// ```
///
/// 2. step (concurrent writes by replica 0 and 1):
///
/// ```text
///               ---> set_0_1(1, 2)
/// set_0_0(1, 1)
///               ---> set_1_0(1, 3)
/// ```
///
/// 3. step (replica 1 does a "merge" operation overwriting the previous conflict):
///
/// ```text
///               ---> set_0_1(1, 2)
/// set_0_0(1, 1)                    ---> set_1_2(1, 4)
///               ---> set_1_0(1, 3)
/// ```
///
/// 4. step (replica 0 overwrites a not-yet delivered operation):
///
/// ```text
///               ---> set_0_1(1, 2)
/// set_0_0(1, 1)                    ---> set_1_2(1, 4) ---> missing ---> set_0_4(1, 6)
///               ---> set_1_0(1, 3)
/// ```
///
/// 5. step (replica 0's missing operation arrives):
///
/// ```text
///               ---> set_0_1(1, 2)
/// set_0_0(1, 1)                    ---> set_1_2(1, 4) ---> set_0_3(1, 5) ---> set_0_4(1, 6)
///               ---> set_1_0(1, 3)
/// ```
///
pub fn mvr_store_operation_history() -> [(Vec<PredRel>, Vec<SetOp>); 5] {
    [
        (vec![], vec![SetOp::new(0, 0, 1, 1)]),
        (
            vec![PredRel::new(0, 0, 0, 1), PredRel::new(0, 0, 1, 0)],
            vec![SetOp::new(0, 1, 1, 2), SetOp::new(1, 0, 1, 3)],
        ),
        (
            vec![PredRel::new(0, 1, 1, 2), PredRel::new(1, 0, 1, 2)],
            vec![SetOp::new(1, 2, 1, 4)],
        ),
        (vec![PredRel::new(0, 3, 0, 4)], vec![SetOp::new(0, 4, 1, 6)]),
        (vec![PredRel::new(1, 2, 0, 3)], vec![SetOp::new(0, 3, 1, 5)]),
    ]
}

/// This relation represents an insertion operation in a list and is used in the
/// [`parser::crdts::list_crdt_datalog`] Datalog program.
#[derive(Copy, Clone, Debug)]
pub struct InsertOp {
    rep_id: u64,
    ctr: u64,
    parent_rep_id: u64,
    parent_ctr: u64,
}

impl InsertOp {
    fn new(rep_id: u64, ctr: u64, parent_rep_id: u64, parent_ctr: u64) -> Self {
        Self {
            rep_id,
            ctr,
            parent_rep_id,
            parent_ctr,
        }
    }
}

impl InputEntity for InsertOp {
    fn schema() -> RelationSchema {
        RelationSchema::new(
            "insert_op",
            ["RepId", "Ctr", "ParentRepId", "ParentCtr"],
            ["RepId", "Ctr", "ParentRepId", "ParentCtr"],
        )
        .expect("Correct schema definition")
    }
}

impl From<InsertOp> for TupleKey {
    fn from(insert_op: InsertOp) -> Self {
        TupleKey::from_iter([
            insert_op.rep_id,
            insert_op.ctr,
            insert_op.parent_rep_id,
            insert_op.parent_ctr,
        ])
    }
}

impl From<InsertOp> for TupleValue {
    fn from(insert_op: InsertOp) -> Self {
        TupleValue::from_iter([
            insert_op.rep_id,
            insert_op.ctr,
            insert_op.parent_rep_id,
            insert_op.parent_ctr,
        ])
    }
}

/// This relation represents an assignment operation of a value to a position in
/// in a list and is used in the [`parser::crdts::list_crdt_datalog`] Datalog program.
#[derive(Copy, Clone, Debug)]
pub struct AssignOp {
    rep_id: u64,
    ctr: u64,
    value: char,
}

impl AssignOp {
    fn new(rep_id: u64, ctr: u64, value: char) -> Self {
        Self { rep_id, ctr, value }
    }
}

impl InputEntity for AssignOp {
    fn schema() -> RelationSchema {
        RelationSchema::new("assign_op", ["RepId", "Ctr", "Value"], ["RepId", "Ctr"])
            .expect("Correct schema definition")
    }
}

impl From<AssignOp> for TupleKey {
    fn from(assign_op: AssignOp) -> Self {
        TupleKey::from_iter([assign_op.rep_id, assign_op.ctr])
    }
}

impl From<AssignOp> for TupleValue {
    fn from(assign_op: AssignOp) -> Self {
        tuple!(assign_op.rep_id, assign_op.ctr, assign_op.value)
    }
}

/// This relation a deletion of an element in a list and is used in the
/// [`parser::crdts::list_crdt_datalog`] Datalog program.
#[derive(Copy, Clone, Debug)]
pub struct RemoveOp {
    rep_id: u64,
    ctr: u64,
}

impl RemoveOp {
    pub fn new(rep_id: u64, ctr: u64) -> Self {
        Self { rep_id, ctr }
    }
}

impl InputEntity for RemoveOp {
    fn schema() -> RelationSchema {
        RelationSchema::new("remove_op", ["RepId", "Ctr"], ["RepId", "Ctr"])
            .expect("Correct schema definition")
    }
}

impl From<RemoveOp> for TupleKey {
    fn from(remove_op: RemoveOp) -> Self {
        TupleKey::from_iter([remove_op.rep_id, remove_op.ctr])
    }
}

impl From<RemoveOp> for TupleValue {
    fn from(remove_op: RemoveOp) -> Self {
        TupleValue::from_iter([remove_op.rep_id, remove_op.ctr])
    }
}

/// Example tree, encoded as `insert(Child, Parent)` facts. Below, the example
/// only shows counters but no replica ids because all updates emanate from
/// the same replica.
///
/// ```text
///        0 <- sentinel element
///      /   \
///     2     1
///   / | \   |
///  6  5  3  4
/// ```
pub fn list_crdt_operation_history_martin() -> [(Vec<InsertOp>, Vec<AssignOp>, Vec<RemoveOp>); 1] {
    [(
        vec![
            InsertOp::new(0, 1, 0, 0),
            InsertOp::new(0, 2, 0, 0),
            InsertOp::new(0, 3, 0, 2),
            InsertOp::new(0, 4, 0, 1),
            InsertOp::new(0, 5, 0, 2),
            InsertOp::new(0, 6, 0, 2),
        ],
        vec![
            AssignOp::new(0, 2, 'H'),
            AssignOp::new(0, 6, 'E'),
            AssignOp::new(0, 5, 'L'),
            AssignOp::new(0, 3, 'L'),
            AssignOp::new(0, 1, 'O'),
            AssignOp::new(0, 4, '!'),
        ],
        vec![],
    )]
}

/// Example tree, encoded as `insert(ChildRepId, ChildCtr, ParentRepId, ParentCtr)`
/// facts. Below, a node depicts `(RepId, Ctr)`:
///
/// ```text
///             (0,0) <- sentinel element
///         /           \
///       (2,1)        (1,1)
///     /   |   \        |
/// (2,3) (1,3) (3,2)  (2,2)
/// ```
pub fn list_crdt_operation_history_multi_replicas()
-> [(Vec<InsertOp>, Vec<AssignOp>, Vec<RemoveOp>); 1] {
    [(
        vec![
            InsertOp::new(2, 1, 0, 0),
            InsertOp::new(1, 1, 0, 0),
            InsertOp::new(2, 2, 1, 1),
            InsertOp::new(2, 3, 2, 1),
            InsertOp::new(1, 3, 2, 1),
            InsertOp::new(3, 2, 2, 1),
        ],
        vec![
            AssignOp::new(2, 1, 'H'),
            AssignOp::new(2, 3, 'E'),
            AssignOp::new(1, 3, 'L'),
            AssignOp::new(3, 2, 'L'),
            AssignOp::new(1, 1, 'O'),
            AssignOp::new(2, 2, '!'),
        ],
        vec![RemoveOp::new(2, 2)],
    )]
}

// For benchmarking purposes.
pub struct Replica {
    rep_id: u64,
    /// Always points to the next unused counter value.
    ctr: u64,
    heads: HashSet<(u64, u64)>,
}

impl Replica {
    pub fn new(rep_id: u64) -> Self {
        Self {
            rep_id,
            ctr: 0,
            heads: HashSet::new(),
        }
    }
    pub fn rep_id(&self) -> u64 {
        self.rep_id
    }
    pub fn ctr(&self) -> u64 {
        self.ctr
    }
    /// Warning: This method assumes that the `set_op` and its causal predecessors
    /// are all causally ready for this replica. There is no buffering due to
    /// missing causal predecessors for now.
    pub fn ack_remote_set_op(&mut self, set_op: &SetOp, pred_rels: &[PredRel]) {
        // We ensure that the set_op is not from this replica.
        debug_assert!(set_op.rep_id != self.rep_id);
        // We ensure that the pred_rels are all related to the set_op.
        debug_assert!(
            pred_rels.iter().all(
                |pred_rel| pred_rel.to_rep_id == set_op.rep_id && pred_rel.to_ctr == set_op.ctr
            )
        );
        for pred_rel in pred_rels {
            let value = (pred_rel.from_rep_id, pred_rel.from_ctr);
            // We remove all heads that are now subsumed by the new set_op.
            if self.heads.contains(&value) {
                self.heads.remove(&value);
            }
        }
        self.heads.insert((set_op.rep_id, set_op.ctr));
        // Advance the Lamport clock if the new set_op has a higher counter.
        if self.ctr < set_op.ctr {
            self.ctr = set_op.ctr + 1;
        }
    }
    pub fn new_local_set_op(&mut self, key: u64, value: u64) -> (SetOp, Vec<PredRel>) {
        let set_op = SetOp::new(self.rep_id, self.ctr, key, value);
        let pred_rels = self
            .heads
            .iter()
            .map(|&(from_rep_id, from_ctr)| {
                let to_rep_id = set_op.rep_id;
                let to_ctr = set_op.ctr;
                PredRel::new(from_rep_id, from_ctr, to_rep_id, to_ctr)
            })
            .collect::<Vec<_>>();
        self.heads.clear();
        self.heads.insert((set_op.rep_id, set_op.ctr));
        self.ctr += 1;
        (set_op, pred_rels)
    }
}
