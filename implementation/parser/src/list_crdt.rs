use crate::Parser;
use compute::{
    IncDataLog,
    dbsp::{CircuitHandle, DbspInput, DbspOutput},
    relation::{TupleKey, TupleValue},
    scalar::ScalarTypedValue,
    tuple,
};
use std::{cmp::max, collections::BTreeMap, iter::Peekable, num::NonZeroUsize};

/// This is an implementation of a list CRDT in Datalog.
///
/// Credits are due to Martin Kleppmann.
pub const LIST_CRDT_DATALOG: &str = r#"
        // These are extensional database predicates (EDBPs).
        insert(RepId, Ctr, ParentRepId, ParentCtr, Value) :- .
        remove(ElemId, ElemCtr) :- .

        // These are intensional database predicates (IDBPs).
        laterChild(ParentRepId, ParentCtr, ChildRepId, ChildCtr) :-
          insert(SiblingRepId = RepId, SiblingCtr = Ctr, ParentRepId, ParentCtr),
          insert(ChildRepId = RepId, ChildCtr = Ctr, ParentRepId, ParentCtr),
          (SiblingCtr > ChildCtr; (SiblingCtr == ChildCtr, SiblingRepId > ChildRepId)).

        firstChild(ParentRepId, ParentCtr, ChildRepId, ChildCtr) :-
          insert(ChildRepId = RepId, ChildCtr = Ctr, ParentRepId, ParentCtr),
          not laterChild(ParentRepId, ParentCtr, ChildRepId, ChildCtr).

        sibling(Child1RepId, Child1Ctr, Child2RepId, Child2Ctr) :-
          insert(Child1RepId = RepId, Child1Ctr = Ctr, ParentRepId, ParentCtr),
          insert(Child2RepId = RepId, Child2Ctr = Ctr, ParentRepId, ParentCtr).

        laterSibling(Child1RepId, Child1Ctr, Child2RepId, Child2Ctr) :-
          sibling(Child1RepId, Child1Ctr, Child2RepId, Child2Ctr),
          (Child1Ctr > Child2Ctr; (Child1Ctr == Child2Ctr, Child1RepId > Child2RepId)).

        laterIndirectSibling(Child1RepId, Child1Ctr, Child3RepId, Child3Ctr) :-
          sibling(Child1RepId, Child1Ctr, Child2RepId, Child2Ctr),
          sibling(Child1RepId, Child1Ctr, Child3RepId = Child2RepId, Child3Ctr = Child2Ctr),
          (Child1Ctr > Child2Ctr; (Child1Ctr == Child2Ctr, Child1RepId > Child2RepId)),
          (Child2Ctr > Child3Ctr; (Child2Ctr == Child3Ctr, Child2RepId > Child3RepId)).

        nextSibling(Child1RepId, Child1Ctr, Child2RepId, Child2Ctr) :-
          laterSibling(Child1RepId, Child1Ctr, Child2RepId, Child2Ctr),
          not laterIndirectSibling(Child1RepId, Child1Ctr, Child2RepId = Child3RepId, Child2Ctr = Child3Ctr).

        distinct hasNextSibling(ChildRepId, ChildCtr) :-
          nextSibling(ChildRepId = Child1RepId, ChildCtr = Child1Ctr).

        nextSiblingAnc(ChildRepId, ChildCtr, AncRepId, AncCtr) :-
          nextSibling(ChildRepId = Child1RepId, ChildCtr = Child1Ctr, AncRepId = Child2RepId, AncCtr = Child2Ctr).
        nextSiblingAnc(ChildRepId, ChildCtr, AncRepId, AncCtr) :-
          insert(ChildRepId = RepId, ChildCtr = Ctr, ParentRepId, ParentCtr),
          not hasNextSibling(ChildRepId, ChildCtr),
          nextSiblingAnc(ParentRepId = ChildRepId, ParentCtr = ChildCtr, AncRepId, AncCtr).

        distinct hasChild(ParentRepId, ParentCtr) :-
          insert(ParentRepId, ParentCtr).

        nextElem(PrevRepId, PrevCtr, NextRepId, NextCtr) :-
          firstChild(PrevRepId = ParentRepId, PrevCtr = ParentCtr, NextRepId = ChildRepId, NextCtr = ChildCtr).
        nextElem(PrevRepId, PrevCtr, NextRepId, NextCtr) :-
          not hasChild(PrevRepId = ParentRepId, PrevCtr = ParentCtr),
          nextSiblingAnc(PrevRepId = ChildRepId, PrevCtr = ChildCtr, NextRepId = AncRepId, NextCtr = AncCtr).

        distinct hasValue(ElemId, ElemCtr) :-
          // Fix for not being able to assign anything to the sentinel element.
          insert(ElemId = ParentRepId, ElemCtr = ParentCtr), ElemId == 0, ElemCtr == 0.
        distinct hasValue(ElemId, ElemCtr) :-
          insert(ElemId = RepId, ElemCtr = Ctr),
          not remove(ElemId, ElemCtr).

        nextElemSkipTombstones(PrevRepId, PrevCtr, NextRepId, NextCtr) :-
          nextElem(PrevRepId, PrevCtr, NextRepId, NextCtr).
        nextElemSkipTombstones(PrevRepId, PrevCtr, NextRepId, NextCtr) :-
          nextElem(PrevRepId, PrevCtr, ViaRepId = NextRepId, ViaCtr = NextCtr),
          not hasValue(ViaRepId = ElemId, ViaCtr = ElemCtr),
          nextElemSkipTombstones(ViaRepId = PrevRepId, ViaCtr = PrevCtr, NextRepId, NextCtr).

        nextVisible(PrevRepId, PrevCtr, NextRepId, NextCtr) :-
          hasValue(PrevRepId = ElemId, PrevCtr = ElemCtr),
          nextElemSkipTombstones(PrevRepId, PrevCtr, NextRepId, NextCtr),
          hasValue(NextRepId = ElemId, NextCtr = ElemCtr).

        listElem(PrevRepId, PrevCtr, Value, NextRepId, NextCtr) :-
          nextVisible(PrevRepId, PrevCtr, NextCtr, NextRepId),
          insert(NextRepId = RepId, NextCtr = Ctr, Value).
    "#;

/// This relation represents an insertion operation in a list.
///
/// It is an input in the [`LIST_CRDT_DATALOG`] Datalog program.
#[derive(Copy, Clone, Debug)]
pub struct InsertOp {
    rep_id: u64,
    ctr: u64,
    parent_rep_id: u64,
    parent_ctr: u64,
    value: char,
}

impl InsertOp {
    pub fn new(rep_id: u64, ctr: u64, parent_rep_id: u64, parent_ctr: u64, value: char) -> Self {
        Self {
            rep_id,
            ctr,
            parent_rep_id,
            parent_ctr,
            value,
        }
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
        tuple!(
            insert_op.rep_id,
            insert_op.ctr,
            insert_op.parent_rep_id,
            insert_op.parent_ctr,
            insert_op.value,
        )
    }
}

/// This relation represents a deletion of an element in a list.
/// The `rep_id` and `ctr` relate to an assign operation's `rep_id` and `ctr`.
///
/// It is an input in the [`LIST_CRDT_DATALOG`] Datalog program.
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

/// Basic operations on a list which are compatible with the outside world and
/// not related to CRDTs' operations. Indexes are plain array indexes.
#[derive(Debug, Clone, Copy)]
pub enum ListOperation {
    /// Inserts a character at the given index in the list.
    InsertAt(usize, char),
    /// Deletes the character at the given index in the list.
    DeleteAt(usize),
}

type PrevRepId = u64;
type PrevCtr = u64;
type PrevId = (PrevRepId, PrevCtr);
type NextRepId = u64;
type NextCtr = u64;
type NextId = (NextRepId, NextCtr);
type ElemRepId = u64;
type ElemCtr = u64;
type ElemId = (ElemRepId, ElemCtr);
type Value = char;

pub struct ListReplica {
    rep_id: u64,
    /// Points to the next unused counter value.
    ctr: u64,
    /// Whether the replica has seen any new updates since the last evaluation.
    is_dirty: bool,
    /// For walking the Datalog program's output of a linked list.
    lookup_table: BTreeMap<PrevId, (Value, NextId)>,
    /// Cached list order for fast repeated access if there are no updates.
    cached_list_order: Vec<(PrevId, ElemId, Value)>,
    /// Cached string for fast repeated access if there are no updates.
    cached_string: String,
    handle: CircuitHandle,
    output: DbspOutput,
    insert_input: DbspInput,
    remove_input: DbspInput,
}

impl ListReplica {
    const SENTINEL_ID: PrevId = (0, 0);

    pub fn new(rep_id: u64) -> Self {
        let threads = NonZeroUsize::new(1).expect("At least one thread is required");
        let (handle, mut inputs, output) = IncDataLog::new(threads, true)
            .build_circuit_from_parser(|root_circuit| {
                Parser::new(root_circuit).parse(LIST_CRDT_DATALOG)
            })
            .expect("Correct Datalog program");
        let insert_input = inputs
            .take("insert")
            .expect("Insert input should be present");
        let remove_input = inputs
            .take("remove")
            .expect("Remove input should be present");
        // Assign a dummy value to the sentinel element.
        // assign_input.insert_with_same_weight([&AssignOp::new(0, 0, 0, 0, '#')], 1);
        Self {
            rep_id,
            ctr: 0,
            lookup_table: BTreeMap::new(),
            cached_list_order: Vec::new(),
            cached_string: String::new(),
            handle,
            output,
            insert_input,
            remove_input,
            is_dirty: false,
        }
    }
    fn consume_ctr(&mut self) -> u64 {
        let ctr = self.ctr;
        self.ctr += 1;
        ctr
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
    /// Generates CRDT list operations, either an [`InsertOp`], an [`AssignOp`],
    /// and/or a [`RemoveOp`] from a local event defined by a [`ListOperation`].
    pub fn generate_ops(
        &mut self,
        local_event: &ListOperation,
    ) -> (Option<InsertOp>, Option<RemoveOp>) {
        match local_event {
            ListOperation::InsertAt(idx, char) => {
                let parent_id = self.get_parent_op_id_from_idx(*idx).unwrap_or_else(|| {
                    panic!("Invalid index for insertion {idx}, op {local_event:#?}")
                });
                let insert_op = InsertOp::new(
                    self.rep_id,
                    self.consume_ctr(),
                    parent_id.0,
                    parent_id.1,
                    *char,
                );
                (Some(insert_op), None)
            }
            ListOperation::DeleteAt(idx) => {
                let elem_op_id = self
                    .get_elem_op_id_from_idx(*idx)
                    .unwrap_or_else(|| panic!("Invalid index for deletion {idx}"));
                let remove_op = RemoveOp::new(elem_op_id.0, elem_op_id.1);
                (None, Some(remove_op))
            }
        }
    }
    pub fn feed_ops<'a>(
        &mut self,
        insert_ops: impl IntoIterator<Item = &'a InsertOp>,
        remove_ops: impl IntoIterator<Item = &'a RemoveOp>,
    ) {
        let mut insert_ops = insert_ops.into_iter().peekable();
        let mut remove_ops = remove_ops.into_iter().peekable();
        self.is_dirty = insert_ops.peek().is_some() || remove_ops.peek().is_some();
        self.insert_input.insert_with_same_weight(insert_ops, 1);
        self.remove_input.insert_with_same_weight(remove_ops, 1);
    }
    pub fn derive_state(&mut self) -> &String {
        if !self.is_dirty {
            // If the replica is not dirty, we return the cached string.
            return &self.cached_string;
        }
        // We derive the state by stepping the circuit.
        self.handle.step().expect("Circuit step should succeed");
        self.apply_output_delta();
        self.is_dirty = false;
        &self.cached_string
    }
    pub fn list_order(&self) -> &[(PrevId, ElemId, Value)] {
        &self.cached_list_order
    }
    pub fn string(&self) -> &String {
        &self.cached_string
    }
    fn apply_output_delta(&mut self) {
        for (zweight, tuple_value) in self.output.to_batch().as_data() {
            let prev_rep_id = tuple_value.data[0].unwrap_into_uint();
            let prev_ctr = tuple_value.data[1].unwrap_into_uint();
            let value = tuple_value.data[2].unwrap_into_char();
            let next_rep_id = tuple_value.data[3].unwrap_into_uint();
            let next_ctr = tuple_value.data[4].unwrap_into_uint();

            let prev_id = (prev_rep_id, prev_ctr);
            let next_id = (next_rep_id, next_ctr);
            match zweight {
                1 => {
                    self.lookup_table.insert(prev_id, (value, next_id));
                }
                -1 => {
                    self.lookup_table.remove(&prev_id);
                }
                _ => panic!("Unexpected zweight: {zweight}"),
            }
        }
        self.update_caches();
    }
    fn update_caches(&mut self) {
        let mut current = ListReplica::SENTINEL_ID;
        let expected_len = self.lookup_table.len();
        self.cached_list_order
            .reserve(max(expected_len - self.cached_list_order.len(), 0));
        self.cached_string
            .reserve(max(expected_len - self.cached_string.len(), 0));
        self.cached_string.clear();
        self.cached_list_order.clear();
        while let Some((value, next_id)) = self.lookup_table.get(&current) {
            current = *next_id;
            self.cached_string.push(*value);
            self.cached_list_order.push((current, *next_id, *value));
        }
    }
}

impl ListReplica {
    /// Just a sketch for handling bursts of local events.
    #[allow(clippy::type_complexity)]
    pub fn generate_burst<'a, Stream: Iterator<Item = &'a ListOperation>>(
        &mut self,
        mut list_ops: Peekable<Stream>,
    ) -> Option<(Peekable<Stream>, Vec<InsertOp>, Vec<RemoveOp>)> {
        // Have a stack of last insert CRDT ops to support deletion?
        let mut last_op = None;
        // Pointer into the insert_ops and assign_ops stack of this burst.
        let mut stack_ptr = -1_i64;
        let mut insert_ops: Vec<InsertOp> = Vec::new();
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
                return Some((list_ops, insert_ops, remove_ops));
            }
            let op = list_ops.next().unwrap(); // Safe to call due to peek.
            match op {
                ListOperation::InsertAt(idx, value) => {
                    let parent_id = if stack_ptr < 0 {
                        self.get_parent_op_id_from_idx(*idx).unwrap_or_else(|| {
                            println!(
                                "{:#?}",
                                self.cached_list_order
                                    .iter()
                                    .enumerate()
                                    .collect::<Vec<_>>()
                            );
                            panic!("Invalid index for insertion {idx}, last op {last_op:#?}")
                        })
                    } else {
                        insert_ops
                            .get(stack_ptr as usize)
                            .map(|insert_op| (insert_op.rep_id, insert_op.ctr))
                            .unwrap()
                    };
                    let insert_op = InsertOp::new(
                        self.rep_id,
                        self.consume_ctr(),
                        parent_id.0,
                        parent_id.1,
                        *value,
                    );
                    insert_ops.push(insert_op);
                    // Reset stack_ptr to point to last insertion.
                    stack_ptr = (insert_ops.len() as i64) - 1;
                }
                ListOperation::DeleteAt(idx) => {
                    let elem_id = if stack_ptr < 0 {
                        self.get_elem_op_id_from_idx(*idx)
                            .unwrap_or_else(|| panic!("Invalid index for deletion {idx}"))
                    } else {
                        insert_ops
                            .get(stack_ptr as usize)
                            .map(|insert_op| (insert_op.rep_id, insert_op.ctr))
                            .unwrap()
                    };
                    let remove_op = RemoveOp::new(elem_id.0, elem_id.1);
                    remove_ops.push(remove_op);
                    stack_ptr -= 1;
                }
            }
            last_op = Some(*op);
        }
        // The stream is exhausted.
        if last_op.is_some() {
            // We have found new ops in this run.
            Some((list_ops, insert_ops, remove_ops))
        } else {
            None
        }
    }
}
