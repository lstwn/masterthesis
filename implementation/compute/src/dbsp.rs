use crate::{
    expr::{Literal, LiteralExpr},
    relation::{Relation, RelationSchema, SchemaTuple, TupleKey, TupleValue},
};
use cli_table::{Cell, Style, Table, format::Justify};
use dbsp::{
    ChildCircuit, IndexedZSetHandle, OrdIndexedZSet, OrdZSet, OutputHandle, RootCircuit, Stream,
    ZWeight,
};
use std::{collections::HashMap, fmt::Display, iter};

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

struct IndexedTestStream {
    a: Relation,
    b: Relation,
}

impl IndexedTestStream {
    fn try_out(&self) {
        let copy: usize = 12;
        let non_copy = String::from("hi");

        // TODO: How to move interpreter state into the closure?

        let selected: OrdIndexedStream = self.a.inner.filter(move |tuple| {
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

        let joined: OrdIndexedStream = selected.join_index(&self.b.inner, |k, left, right| {
            // merge left and right tuple
            Some((k.clone(), right.clone()))
        });

        let projected: OrdIndexedStream = joined.map_index(|(k, tuple)| {
            // project tuple
            (k.clone(), tuple.clone())
        });
    }
}

pub fn new_ord_indexed_stream(
    circuit: &mut RootCircuit,
) -> (OrdIndexedStream, OrdIndexedStreamInputHandle) {
    circuit.add_input_indexed_zset::<TupleKey, TupleValue>()
}

pub type OrdIndexedStreamInputHandle = IndexedZSetHandle<TupleKey, TupleValue>;

pub type OrdIndexedStreamOutputHandle = OutputHandle<OrdIndexedZSet<TupleKey, TupleValue>>;

pub type OrdIndexedStream<Circuit = ChildCircuit<()>> =
    Stream<Circuit, OrdIndexedZSet<TupleKey, TupleValue>>;

pub struct DbspInputs {
    inputs: HashMap<String, DbspInput>,
}

impl DbspInputs {
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
        }
    }
    fn insert(&mut self, name: String, input: DbspInput) {
        self.inputs.insert(name, input);
    }
    pub fn get(&self, name: &str) -> Option<&DbspInput> {
        self.inputs.get(name)
    }
    pub fn iter(&self) -> impl Iterator<Item = &DbspInput> {
        self.inputs.values()
    }
}

pub struct DbspInput {
    schema: RelationSchema,
    handle: OrdIndexedStreamInputHandle,
}

impl DbspInput {
    pub fn new<T: Into<String>>(
        name: T,
        schema: RelationSchema,
        circuit: &mut RootCircuit,
        inputs: &mut DbspInputs,
    ) -> LiteralExpr {
        let (stream, handle) = new_ord_indexed_stream(circuit);
        let name = name.into();
        let input = Self {
            schema: schema.clone(),
            handle,
        };
        inputs.insert(name.clone(), input);
        LiteralExpr {
            value: Literal::Relation(Relation::new(name, schema, stream)),
        }
    }
    pub fn handle(&self) -> &OrdIndexedStreamInputHandle {
        &self.handle
    }
    pub fn insert<'a, T: Into<TupleKey> + Into<TupleValue> + Clone + 'a>(
        &self,
        tuples: impl IntoIterator<Item = (&'a T, ZWeight)>,
    ) -> () {
        tuples.into_iter().for_each(|(tuple, z_weight)| {
            self.handle
                .push(tuple.clone().into(), (tuple.clone().into(), z_weight))
        })
    }
    pub fn insert_with_same_weight<'a, T: Into<TupleKey> + Into<TupleValue> + Clone + 'a>(
        &self,
        tuples: impl IntoIterator<Item = &'a T>,
        z_weight: ZWeight,
    ) -> () {
        self.insert(tuples.into_iter().map(|tuple| (tuple, z_weight)));
    }
}

pub struct DbspOutput {
    handle: OrdIndexedStreamOutputHandle,
    schema: RelationSchema,
}

impl DbspOutput {
    pub fn new(schema: RelationSchema, handle: OrdIndexedStreamOutputHandle) -> Self {
        Self { schema, handle }
    }
    pub fn to_table(&self) -> impl Display {
        const JUSTIFICATION: Justify = Justify::Right;
        let table = self
            .handle
            .consolidate()
            .iter()
            .map(|(key, tuple, weight)| {
                iter::once(weight.to_string().cell().justify(JUSTIFICATION)).chain(
                    SchemaTuple::new(&self.schema.tuple, &tuple)
                        .fields()
                        .map(|attribute| attribute.to_string().cell().justify(JUSTIFICATION))
                        .collect::<Vec<_>>(),
                )
            })
            .table()
            .title(
                iter::once("z-weight".cell()).chain(
                    self.schema
                        .tuple
                        .active_fields()
                        .map(|(_, info)| info.name().cell()),
                ),
            )
            .bold(true);
        table.display().expect("table error")
    }
}
