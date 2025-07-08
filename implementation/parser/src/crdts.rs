//! This module contains definitions of CRDTs in Datalog.

/// This represents a key value store whose registers are multi-valued registers
/// powered by a causal history of operations. This is a full-blown CRDT.
pub const fn mvr_crdt_store_datalog() -> &'static str {
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
                                                    isCausallyReady(RepId, Ctr),
                                                    // Comments between atoms are also fine.
                                                    // Spanning multiple lines.
                                                    set(RepId, Ctr, Key, Value).
                                                    // Trailing comments are fine.
    "#
}

/// This is a simplified version of the key value with multi-valued registers
/// that assumes a causal broadcast on the application layer.
pub const fn mvr_store_datalog() -> &'static str {
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

/// This is an implementation of the replicated growable array (RGA) CRDT in Datalog.
/// Credits are due to Martin Kleppmann.
pub const fn list_crdt_datalog() -> &'static str {
    r#"
        // These are extensional database predicates (EDBPs).
        insert(RepId, Ctr, ParentRepId, ParentCtr) :- .
        assign(RepId, Ctr, Value) :- .
        remove(RepId, Ctr) :- .

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

        listElem(PrevRepId, PrevCtr, Value, NextRepId, NextCtr) :-
            assign(NextRepId = RepId, NextCtr = Ctr, Value),
            not remove(NextRepId = RepId, NextCtr = Ctr),
            nextElem(PrevRepId, PrevCtr, NextRepId, NextCtr).
    "#
}
