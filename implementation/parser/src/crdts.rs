//! This module contains definitions of CRDTs in Datalog.

/// This represents a key value store whose registers are multi-valued registers
/// powered by a causal history of operations. This is a full-blown CRDT.
pub fn mvr_crdt_store_datalog() -> &'static str {
    r#"
        // These are extensional database predicates (EDBPs).
        pred(FromNodeId, FromCounter, ToNodeId, ToCounter)  :- .
        set(NodeId, Counter, Key, Value)                    :- .

        // These are intensional database predicates (IDBPs).
        distinct overwritten(NodeId, Counter)     :- pred(NodeId = FromNodeId, Counter = FromCounter, _ToNodeId, _ToCounter).
        distinct overwrites(NodeId, Counter)      :- pred(_FromNodeId, _FromCounter, NodeId = ToNodeId, Counter = ToCounter).

        isRoot(NodeId, Counter)                   :- set(NodeId, Counter, _Key, _Value),
                                                     not overwrites(NodeId, Counter).

        isLeaf(NodeId, Counter)                   :- set(NodeId, Counter, _Key, _Value),
                                                     not overwritten(NodeId, Counter).

        isCausallyReady(NodeId, Counter)          :- isRoot(NodeId, Counter). // Trailing comment.
        isCausallyReady(NodeId, Counter)          :- isCausallyReady(FromNodeId = NodeId, FromCounter = Counter),
                                                     pred(FromNodeId, FromCounter, NodeId = ToNodeId, Counter = ToCounter).

        mvrStore(Key, Value)                      :- // Comments before atoms are fine.
                                                     isLeaf(NodeId, Counter),
                                                     // Comments between atoms are also fine.
                                                     // Spanning multiple lines.
                                                     isCausallyReady(NodeId, Counter),
                                                     set(NodeId, Counter, Key, Value).
        // Trailing comments are fine.
    "#
}

/// This is a simplified version of the key value with multi-valued registers
/// that assumes a causal broadcast on the application layer.
pub fn mvr_store_datalog() -> &'static str {
    r#"
        // These are extensional database predicates (EDBPs).
        pred(FromNodeId, FromCounter, ToNodeId, ToCounter)  :- .
        set(NodeId, Counter, Key, Value)                    :- .

        // These are intensional database predicates (IDBPs).
        distinct overwritten(NodeId, Counter)     :- pred(NodeId = FromNodeId, Counter = FromCounter, _ToNodeId, _ToCounter).
        // TODO: This will be an error as the schema does not align.
        mvrStore(Key, Value)                      :- set(NodeId, Counter, Key, Value),
                                                     not overwritten(NodeId, Counter).
    "#
}
