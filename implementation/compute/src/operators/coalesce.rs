use crate::relation::{RelationRef, SchemaTuple, TupleKey, TupleValue, new_relation};
use std::rc::Rc;

pub fn coalesce_helper(relation: RelationRef) -> RelationRef {
    let relation_ref = relation.borrow();

    if relation_ref.schema.is_coalesced() {
        drop(relation_ref);
        return relation;
    }

    let schema = relation_ref.schema.coalesce();
    let coalesced = relation_ref.inner.map_index({
        let relation = Rc::clone(&relation);
        move |(key, tuple)| {
            let relation_ref = relation.borrow();
            let key: TupleKey = SchemaTuple::new(&relation_ref.schema.key, key)
                .coalesce()
                .collect();
            let tuple: TupleValue = SchemaTuple::new(&relation_ref.schema.tuple, tuple)
                .coalesce()
                .collect();
            (key, tuple)
        }
    });

    new_relation(schema, coalesced)
}
