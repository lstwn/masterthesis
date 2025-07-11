//! This module is only available if compiling with the `test` feature.
//! It provides helpers for testing and benchmarking purposes.

use crate::{
    IncDataLog,
    relation::{RelationSchema, TupleKey, TupleValue},
    scalar::ScalarTypedValue,
    tuple,
};
use std::{
    collections::{BTreeMap, HashSet},
    fmt::Debug,
    iter::Peekable,
    num::NonZeroUsize,
};

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
    pub fn new(rep_id: u64, ctr: u64, parent_rep_id: u64, parent_ctr: u64) -> Self {
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
    elem_id: u64,
    elem_ctr: u64,
    value: char,
}

impl AssignOp {
    pub fn new(rep_id: u64, ctr: u64, elem_id: u64, elem_ctr: u64, value: char) -> Self {
        Self {
            rep_id,
            ctr,
            elem_id,
            elem_ctr,
            value,
        }
    }
}

impl InputEntity for AssignOp {
    fn schema() -> RelationSchema {
        RelationSchema::new(
            "assign_op",
            ["RepId", "Ctr", "ElemId", "ElemCtr", "Value"],
            ["RepId", "Ctr"],
        )
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
        tuple!(
            assign_op.rep_id,
            assign_op.ctr,
            assign_op.elem_id,
            assign_op.elem_ctr,
            assign_op.value
        )
    }
}

/// This relation a deletion of an element in a list and is used in the
/// [`parser::crdts::list_crdt_datalog`] Datalog program.
/// The `rep_id` and `ctr` relate to an assign operation's `rep_id` and `ctr`.
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
            // Oddly, another replica assigns the values to the list elements.
            // I think it should be alternating between the insert ops and assign
            // ops but this would break Martin's example.
            AssignOp::new(0, 0, 0, 0, '#'), // dummy element
            AssignOp::new(1, 2, 0, 2, 'H'),
            AssignOp::new(1, 3, 0, 6, 'E'),
            AssignOp::new(1, 4, 0, 5, 'L'),
            AssignOp::new(1, 5, 0, 3, 'L'),
            AssignOp::new(1, 6, 0, 1, 'O'),
            AssignOp::new(1, 7, 0, 4, '!'),
        ],
        // No removals here.
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
/// (2,3) (1,3) (3,2)  (2,5)
/// ```
pub fn list_crdt_operation_history_multi_replicas()
-> [(Vec<InsertOp>, Vec<AssignOp>, Vec<RemoveOp>); 1] {
    [(
        vec![
            InsertOp::new(2, 1, 0, 0),
            InsertOp::new(1, 1, 0, 0),
            InsertOp::new(2, 5, 1, 1),
            InsertOp::new(2, 3, 2, 1),
            InsertOp::new(1, 3, 2, 1),
            InsertOp::new(3, 2, 2, 1),
        ],
        vec![
            AssignOp::new(0, 0, 0, 0, '#'), // dummy element
            AssignOp::new(2, 2, 2, 1, 'H'),
            AssignOp::new(2, 4, 2, 3, 'E'),
            AssignOp::new(1, 4, 1, 3, 'L'),
            AssignOp::new(3, 3, 3, 2, 'L'),
            AssignOp::new(1, 2, 1, 1, 'O'),
            AssignOp::new(2, 6, 2, 5, '!'),
        ],
        // We can remove the 'E' in the *middle* of the list! :)
        vec![RemoveOp::new(2, 4)],
    )]
}

#[derive(Debug, Clone, Copy)]
pub enum ListOperation {
    InsertAt(usize, char),
    DeleteAt(usize),
}

type PrevRepId = u64;
type PrevCtr = u64;
type PrevId = (PrevRepId, PrevCtr);
type NextRepId = u64;
type NextCtr = u64;
type NextId = (NextRepId, NextCtr);
type ElemId = (u64, u64);
type Value = char;

pub struct ListReplica {
    rep_id: u64,
    ctr: u64,
    lookup_table: BTreeMap<PrevId, (NextId, Value)>,
    cached_list_order: Vec<(PrevId, ElemId, Value)>,
}

impl ListReplica {
    const SENTINEL_ID: PrevId = (0, 0);

    pub fn new(rep_id: u64) -> Self {
        // TODO: Sentinel element should be there already.
        Self {
            rep_id,
            ctr: 0,
            lookup_table: BTreeMap::new(),
            cached_list_order: Vec::new(),
        }
    }
    fn get_parent_op_id_from_idx(&self, idx: usize) -> Option<PrevId> {
        let list_order = &self.cached_list_order;
        list_order
            .get(idx)
            .map(|(prev_id, _elem_id, _value)| *prev_id)
            .or(match idx {
                0 if self.lookup_table.is_empty() => {
                    // If the list is empty, we return the sentinel ID.
                    Some(Self::SENTINEL_ID)
                }
                idx if list_order.len() == idx => {
                    // If the index is at the end of the list,
                    // we return the last element's ID.
                    list_order
                        .last()
                        .map(|(_prev_id, elem_id, _value)| *elem_id)
                }
                _ => None,
            })
    }
    fn get_elem_op_id_from_idx(&self, idx: usize) -> Option<ElemId> {
        let list_order = &self.cached_list_order;
        list_order
            .get(idx)
            .map(|(_prev_id, elem_id, _value)| *elem_id)
    }
    pub fn feed_ops<'a, Stream: Iterator<Item = &'a ListOperation>>(
        &mut self,
        mut list_ops: Peekable<Stream>,
    ) -> Option<(
        Peekable<Stream>,
        Vec<InsertOp>,
        Vec<AssignOp>,
        Vec<RemoveOp>,
    )> {
        // Have a stack of last insert CRDT ops to support deletion?
        let mut last_op = None;
        let mut insert_ops = Vec::new();
        let mut assign_ops = Vec::new();
        let mut remove_ops = Vec::new();
        while let Some(op) = list_ops.peek() {
            let is_consecutive = last_op
                .map(|last_op| match (last_op, op) {
                    (ListOperation::InsertAt(last_idx, _), ListOperation::InsertAt(idx, _)) => {
                        last_idx == idx - 1
                    }
                    (ListOperation::InsertAt(last_idx, _), ListOperation::DeleteAt(idx))
                    | (ListOperation::DeleteAt(last_idx), ListOperation::InsertAt(idx, _)) => {
                        last_idx == *idx
                    }
                    (ListOperation::DeleteAt(last_idx), ListOperation::DeleteAt(idx)) => {
                        last_idx - 1 == *idx
                    }
                })
                .unwrap_or(true);
            if !is_consecutive {
                // We yield due to a cursor jump.
                return Some((list_ops, insert_ops, assign_ops, remove_ops));
            }
            let op = list_ops.next().unwrap(); // Safe to call due to peek.
            match op {
                ListOperation::InsertAt(idx, value) => {
                    let parent_id = last_op.map_or_else(
                        || {
                            self.get_parent_op_id_from_idx(*idx)
                                .unwrap_or_else(|| panic!("Invalid index for insertion {idx}"))
                        },
                        |last_op| (self.rep_id, self.ctr - 2),
                    );
                    let insert_op =
                        InsertOp::new(self.rep_id, self.consume_ctr(), parent_id.0, parent_id.1);
                    insert_ops.push(insert_op);
                    let assign_op = AssignOp::new(
                        self.rep_id,
                        self.consume_ctr(),
                        insert_op.rep_id,
                        insert_op.ctr,
                        *value,
                    );
                    assign_ops.push(assign_op);
                }
                ListOperation::DeleteAt(idx) => {
                    let elem_id = last_op.map_or_else(
                        || {
                            self.get_elem_op_id_from_idx(*idx)
                                .unwrap_or_else(|| panic!("Invalid index for deletion {idx}"))
                        },
                        |last_op| (self.rep_id, self.ctr - 1),
                    );
                    let remove_op = RemoveOp::new(elem_id.0, elem_id.1);
                    remove_ops.push(remove_op);
                }
            }
            last_op = Some(*op);
        }
        // The stream is exhausted.
        if last_op.is_some() {
            // We have found new ops in this run.
            Some((list_ops, insert_ops, assign_ops, remove_ops))
        } else {
            None
        }
    }
    fn consume_ctr(&mut self) -> u64 {
        let ctr = self.ctr;
        self.ctr += 1;
        ctr
    }
    pub fn to_list_order(&self) -> Vec<(PrevId, ElemId, Value)> {
        let mut current = ListReplica::SENTINEL_ID;
        let mut result = Vec::with_capacity(self.lookup_table.len());
        while let Some((next_id, value)) = self.lookup_table.get(&current) {
            current = *next_id;
            result.push((current, *next_id, *value));
        }
        result
    }
    pub fn materialize_string(&self) -> String {
        self.to_list_order()
            .iter()
            .map(|(prev_id, elem_id, value)| value)
            .collect()
    }
    pub fn apply_output_delta<'a>(
        &mut self,
        output_delta: impl Iterator<Item = (i64, &'a TupleValue)>,
    ) {
        for (zweight, tuple_value) in output_delta {
            let prev_rep_id = tuple_value.data[0].unwrap_into_uint();
            let prev_ctr = tuple_value.data[1].unwrap_into_uint();
            let value = tuple_value.data[2].unwrap_into_char();
            let next_rep_id = tuple_value.data[3].unwrap_into_uint();
            let next_ctr = tuple_value.data[4].unwrap_into_uint();

            let prev_id = (prev_rep_id, prev_ctr);
            let next_id = (next_rep_id, next_ctr);
            match zweight {
                1 => {
                    self.lookup_table.insert(prev_id, (next_id, value));
                }
                -1 => {
                    self.lookup_table.remove(&prev_id);
                }
                _ => panic!("Unexpected zweight: {zweight}"),
            }
        }
        self.cached_list_order = self.to_list_order();
    }
}

// For benchmarking purposes.
pub struct KeyValueStoreReplica {
    rep_id: u64,
    /// Always points to the next unused counter value.
    ctr: u64,
    heads: HashSet<(u64, u64)>,
}

impl KeyValueStoreReplica {
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
