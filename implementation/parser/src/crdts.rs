//! This module contains definitions of CRDTs in Datalog.

/// This represents a key value store whose registers are multi-valued registers
/// powered by a causal history of operations. This is a full-blown CRDT.
pub fn mvr_crdt_store_datalog() -> &'static str {
    r#"
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
                                                    // Comments between atoms are also fine.
                                                    // Spanning multiple lines.
                                                    isCausallyReady(RepId, Ctr),
                                                    set(RepId, Ctr, Key, Value).
                                                    // Trailing comments are fine.
    "#
}

/// This is a simplified version of the key value with multi-valued registers
/// that assumes a causal broadcast on the application layer.
pub fn mvr_store_datalog() -> &'static str {
    r#"
        // These are extensional database predicates (EDBPs).
        pred(FromRepId, FromCtr, ToRepId, ToCtr) :- .
        set(RepId, Ctr, Key, Value)              :- .

        // These are intensional database predicates (IDBPs).
        distinct overwritten(RepId, Ctr)         :- pred(RepId = FromRepId, Ctr = FromCtr, _ToRepId, _ToCtr).
        mvrStore(Key, Value)                     :- set(RepId, Ctr, Key, Value),
                                                    not overwritten(RepId, Ctr).
    "#
}
