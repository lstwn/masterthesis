use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
};

use compute::{
    IncDataLog,
    dbsp::{CircuitHandle, DbspInput, DbspOutput},
    relation::{TupleKey, TupleValue},
};

use crate::Parser;

/// This is an implementation of a key value store with multi-value registers
/// in Datalog.
/// It also includes a causal broadcast mechanism to ensure that operations
/// respect their causal dependencies, unlike [`mvr_kv_store_datalog`].
pub const MVR_KV_STORE_CRDT_DATALOG: &str = r#"
        // These are extensional database predicates (EDBPs).
        pred(FromRepId, FromCtr, ToRepId, ToCtr) :- .
        set(RepId, Ctr, Key, Value)              :- .

        // These are intensional database predicates (IDBPs).
        distinct overwritten(RepId, Ctr)         :- pred(RepId = FromRepId, Ctr = FromCtr, _ToRepId, _ToCtr).
        distinct overwrites(RepId, Ctr)          :- pred(_FromRepId, _FromCtr, RepId = ToRepId, Ctr = ToCtr).

        isRoot(RepId, Ctr)                       :- set(RepId, Ctr, _Key, _Value),
                                                    not overwrites(RepId, Ctr).

        isLeaf(RepId, Ctr)                       :- set(RepId, Ctr, _Key, _Value),
                                                    not overwritten(RepId, Ctr).

        isCausallyReady(RepId, Ctr)              :- isRoot(RepId, Ctr). // Trailing comment.
        isCausallyReady(RepId, Ctr)              :- isCausallyReady(FromRepId = RepId, FromCtr = Ctr),
                                                    pred(FromRepId, FromCtr, RepId = ToRepId, Ctr = ToCtr).

        mvrStore(Key, Value)                     :- // Comments before atoms are fine.
                                                    isLeaf(RepId, Ctr),
                                                    // Alternative version: Do not use `isLeaf` but
                                                    // inline the logic of `isLeaf` here by using instead:
                                                    // not overwritten(RepId, Ctr),
                                                    isCausallyReady(RepId, Ctr),
                                                    // Comments between atoms are also fine.
                                                    // Spanning multiple lines.
                                                    set(RepId, Ctr, Key, Value).
                                                    // Trailing comments are fine.
    "#;

/// This is an implementation of a key value store with multi-value registers
/// in Datalog.
/// It does _not_ include a causal broadcast mechanism and assumes its inputs
/// are fed in in causal order.
///
/// [`mvr_kv_store_crdt_datalog`] extends this with a causal broadcast mechanism.
pub const MVR_KV_STORE_DATALOG: &str = r#"
        // These are extensional database predicates (EDBPs).
        pred(FromRepId, FromCtr, ToRepId, ToCtr) :- .
        set(RepId, Ctr, Key, Value)              :- .

        // These are intensional database predicates (IDBPs).
        distinct overwritten(RepId, Ctr)         :- pred(RepId = FromRepId, Ctr = FromCtr, _ToRepId, _ToCtr).
        mvrStore(Key, Value)                     :- set(RepId, Ctr, Key, Value),
                                                    not overwritten(RepId, Ctr).
    "#;

/// This relation represents set operations on a key-value store and is used in the
/// [`MVR_KV_STORE_CRDT_DATALOG`] and [`MVR_KV_STORE_DATALOG`] Datalog programs.
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

/// This relation represents a causal predecessor relationship and is used in the
/// [`MVR_KV_STORE_CRDT_DATALOG`] and [`MVR_KV_STORE_DATALOG`] Datalog programs.
#[derive(Copy, Clone, Debug)]
pub struct PredOp {
    from_rep_id: u64,
    from_ctr: u64,
    to_rep_id: u64,
    to_ctr: u64,
}

impl PredOp {
    pub fn new(from_rep_id: u64, from_ctr: u64, to_rep_id: u64, to_ctr: u64) -> Self {
        Self {
            from_rep_id,
            from_ctr,
            to_rep_id,
            to_ctr,
        }
    }
}

impl From<PredOp> for TupleKey {
    fn from(pred_rel: PredOp) -> Self {
        TupleKey::from_iter([
            pred_rel.from_rep_id,
            pred_rel.from_ctr,
            pred_rel.to_rep_id,
            pred_rel.to_ctr,
        ])
    }
}

impl From<PredOp> for TupleValue {
    fn from(pred_rel: PredOp) -> Self {
        TupleValue::from_iter([
            pred_rel.from_rep_id,
            pred_rel.from_ctr,
            pred_rel.to_rep_id,
            pred_rel.to_ctr,
        ])
    }
}

/// Basic operations on a key-value store compatible with the outside world and
/// not related to CRDTs. Indexes are plain array indexes.
#[derive(Debug, Clone, Copy)]
pub enum KeyValueStoreOperation {
    /// Sets the value at the given key to the given value.
    Set(u64, u64),
    // No delete operation defined for now. It could also be implemented as
    // a set operation with a `None` value.
}

pub struct KeyValueStoreReplica {
    rep_id: u64,
    /// Points to the next unused counter value.
    ctr: u64,
    /// Whether the replica has seen any new updates since the last evaluation.
    is_dirty: bool,
    /// The current key value store data. Both keys and values are u64s.
    store: HashMap<u64, HashSet<u64>>,
    /// The heads (leafs) of the causal broadcast graph. Would be nice to have
    /// them be obtained from their Datalog definition but this requires to support
    /// specifying multiple predicates of interest first.
    ///
    /// So for now, we resort to maintaining them manually.
    heads: HashSet<(u64, u64)>,
    handle: CircuitHandle,
    output: DbspOutput,
    set_input: DbspInput,
    pred_input: DbspInput,
}

impl KeyValueStoreReplica {
    pub fn new(rep_id: u64, datalog_impl: &'static str) -> Self {
        let threads = NonZeroUsize::new(1).expect("At least one thread is required");
        let (handle, mut inputs, output) = IncDataLog::new(threads, true)
            .build_circuit_from_parser(|root_circuit| Parser::new(root_circuit).parse(datalog_impl))
            .expect("Correct Datalog program");
        let set_input = inputs.take("set").expect("Set input should be present");
        let pred_input = inputs.take("pred").expect("Pred input should be present");
        Self {
            rep_id,
            ctr: 0,
            is_dirty: false,
            store: HashMap::new(),
            heads: HashSet::new(),
            handle,
            output,
            set_input,
            pred_input,
        }
    }
    pub fn ctr(&self) -> u64 {
        self.ctr
    }
    pub fn consume_ctr(&mut self) -> u64 {
        let ctr = self.ctr;
        self.ctr += 1;
        ctr
    }
    pub fn generate_ops(&mut self, local_event: &KeyValueStoreOperation) -> (SetOp, Vec<PredOp>) {
        match local_event {
            KeyValueStoreOperation::Set(key, value) => {
                let set_op = SetOp::new(self.rep_id, self.consume_ctr(), *key, *value);
                let pred_rels = self
                    .heads
                    .iter()
                    .map(|&(from_rep_id, from_ctr)| {
                        let to_rep_id = set_op.rep_id;
                        let to_ctr = set_op.ctr;
                        PredOp::new(from_rep_id, from_ctr, to_rep_id, to_ctr)
                    })
                    .collect::<Vec<_>>();
                self.heads.clear();
                self.heads.insert((set_op.rep_id, set_op.ctr));
                (set_op, pred_rels)
            }
        }
    }
    pub fn feed_ops<'a>(
        &mut self,
        set_ops: impl IntoIterator<Item = &'a SetOp>,
        pred_ops: impl IntoIterator<Item = &'a PredOp>,
    ) {
        let mut set_ops = set_ops.into_iter().peekable();
        let mut pred_ops = pred_ops.into_iter().peekable();
        self.is_dirty = set_ops.peek().is_some() || pred_ops.peek().is_some();
        self.set_input.insert_with_same_weight(set_ops, 1);
        self.pred_input.insert_with_same_weight(pred_ops, 1);
    }
    pub fn derive_state(&mut self) -> &HashMap<u64, HashSet<u64>> {
        if !self.is_dirty {
            return &self.store;
        }
        // We derive the state by stepping the circuit.
        self.handle.step().expect("Circuit step should succeed");
        self.apply_output_delta();
        self.is_dirty = false;
        &self.store
    }
    fn apply_output_delta(&mut self) {
        // The output is a batch of tuples, each representing a key-value pair.
        for (zweight, tuple_value) in self.output.to_batch().as_data() {
            let key = tuple_value.data[0].unwrap_into_uint();
            let value = tuple_value.data[1].unwrap_into_uint();

            match zweight {
                1 => {
                    // Insert or update the key-value pair.
                    self.store
                        .entry(key)
                        .and_modify(|values| {
                            values.insert(value);
                        })
                        .or_insert(HashSet::from([value]));
                }
                -1 => {
                    // Remove the key if it exists.
                    self.store.entry(key).and_modify(|values| {
                        values.remove(&value);
                    });
                }
                _ => {
                    panic!("Unexpected zweight: {zweight}");
                }
            }
        }
    }
}
