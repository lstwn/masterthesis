use crate::relation::{TupleKey, TupleValue};
use dbsp::{ChildCircuit, OrdIndexedZSet, OrdZSet, Stream};

type OrdStream = Stream<ChildCircuit<()>, OrdZSet<TupleValue>>;

struct OrderedTestStream {
    inner: OrdStream,
    other: OrdStream,
}

impl OrderedTestStream {
    fn try_out(&self) {
        let selected: OrdStream = self.inner.filter(move |tuple| {
            // execute filter expression
            // return result
            true
        });

        // Issue: Apparently, there is no way to select a join key here,
        // the entire tuple will be matched, hence, the OrdZSet is not a good
        // choice here!
        let joined: OrdStream = selected.join_generic(&self.other, |k, _, _| {
            // merge left and right tuple
            Some((k.clone(), ()))
        });

        let projected: OrdStream = joined.map(|k| {
            // project tuple
            k.clone()
            // (k.clone(), ())
        });
    }
}

pub type OrdIndexedStream = Stream<ChildCircuit<()>, OrdIndexedZSet<TupleKey, TupleValue>>;

struct IndexedTestStream {
    inner: OrdIndexedStream,
    other: OrdIndexedStream,
}

impl IndexedTestStream {
    fn try_out(&self) {
        let copy: usize = 12;
        let non_copy = String::from("hi");

        // TODO: How to move interpreter state into the closure?

        let selected: OrdIndexedStream = self.inner.filter(move |tuple| {
            // Put attributes into scope. How?
            // - Define vars uninitialized beforehand and just assign them here?
            //   This would allow to run the resolver just once before the
            //   interpreter starts.
            // - Define vars with a var declaration stmt with an initializer?
            //   But this requires running the resolver for each tuple.
            // - Hacky through the interpreter?
            // - Through an (anonymous) function call?
            // execute filter expression
            // return result
            let access_copy = copy;
            let access_non_copy = &non_copy;
            true
        });

        let joined: OrdIndexedStream = selected.join_index(&self.other, |k, left, right| {
            // merge left and right tuple
            Some((k.clone(), right.clone()))
        });

        let projected: OrdIndexedStream = joined.map_index(|(k, tuple)| {
            // project tuple
            (k.clone(), tuple.clone())
        });
    }
}
