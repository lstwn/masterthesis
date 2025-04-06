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
    pub fn new(
        schema: RelationSchema,
        circuit: &mut RootCircuit,
        inputs: &mut DbspInputs,
    ) -> LiteralExpr {
        let (stream, handle) = new_ord_indexed_stream(circuit);
        let input = Self {
            schema: schema.clone(),
            handle,
        };
        inputs.insert(schema.name.clone(), input);
        LiteralExpr {
            value: Literal::Relation(Relation::new(schema, stream)),
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
                iter::once("z-weight".cell())
                    .chain(self.schema.tuple.field_names(&None).map(|name| name.cell())),
            )
            .bold(true);
        table.display().expect("table error")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use dbsp::{
        Circuit,
        operator::{Generator, Z1},
    };
    use std::{cell::RefCell, rc::Rc};

    // TODO: Provide a PR to DBSP because their example is buggy and panics.
    #[test]
    fn test_iterate() -> Result<(), anyhow::Error> {
        let (circuit, output) = RootCircuit::build(|circuit| {
            // Generate sequence 0, 1, 2, ...
            let mut n: usize = 0;
            let source = circuit.add_source(Generator::new(move || {
                let result = n;
                n = n + 1;
                result
            }));
            // Compute factorial of each number in the sequence.
            let fact = circuit.iterate(|child| {
                let counter = Rc::new(RefCell::new(1));
                let counter_clone = Rc::clone(&counter);
                let countdown = source.delta0(child).apply(move |parent_val| {
                    let mut counter_local = counter_clone.borrow_mut();
                    *counter_local += *parent_val;
                    let res = *counter_local;
                    *counter_local -= 1;
                    res
                });
                let (z1_output, z1_feedback) = child.add_feedback_with_export(Z1::new(1));
                let multiplication =
                    countdown.apply2(&z1_output.local, |n1: &usize, n2: &usize| n1 * n2);
                z1_feedback.connect(&multiplication);
                // Stop iterating once the counter reached 0.
                Ok((move || Ok(*counter.borrow() == 0), z1_output.export))
            })?;
            Ok(fact.output())
        })?;

        let factorial = |n: usize| (1..n).fold(1, |acc, cur| acc * cur);
        let iterations = 10;
        for i in 1..=iterations {
            circuit.step()?;
            let result = output.take_from_all();
            let result = result.first().unwrap();
            println!("Step {:3}: {:3}! = {}", i, i - 1, result);
            assert_eq!(*result, factorial(i));
        }

        Ok(())
    }

    #[test]
    fn test_add_feedback() -> Result<(), anyhow::Error> {
        let (circuit, output) = RootCircuit::build(|circuit| {
            // Create a data source.
            let mut n: usize = 1;
            let source = circuit.add_source(Generator::new(move || {
                let result = n;
                n = n + 1;
                result
            }));
            // Create z1.  `z1_output` will contain the output stream of `z1`; `z1_feedback`
            // is a placeholder where we can later plug the input to `z1`.
            let (z1_output, z1_feedback) = circuit.add_feedback(Z1::new(0));
            // Connect outputs of `source` and `z1` to the plus operator.
            let plus = source.apply2(&z1_output, |n1: &usize, n2: &usize| n1 + n2);
            // Connect the output of `+` as input to `z1`.
            z1_feedback.connect(&plus);

            Ok(z1_output.output())
        })?;

        let sum_first_natural_numbers = |n: usize| (n * (n + 1)) / 2;
        let iterations = 10;
        for i in 1..=iterations {
            circuit.step()?;
            let result = output.take_from_all();
            let result = result.first().unwrap();
            println!("Step {:3}: \\sum_{{i=0}}^{} i = {}", i, i - 1, result);
            assert_eq!(*result, sum_first_natural_numbers(i - 1));
        }

        Ok(())
    }
}
