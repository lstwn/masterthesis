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
        utils::{Tup2, Tup3, Tup4},
        zset, zset_set,
    };
    use std::{cell::RefCell, rc::Rc};

    // TODO: Provide a PR to DBSP with this example.
    #[test]
    fn test_transitive_closure() -> Result<(), anyhow::Error> {
        let (circuit, output_handle) = RootCircuit::build(move |root_circuit| {
            let mut edges_data = vec![
                // The first clock cycle adds a graph of four nodes:
                // |0| -1-> |1| -1-> |2| -2-> |3| -2-> |4|
                zset_set! { Tup3(0 as usize, 1 as usize, 1 as usize), Tup3(1, 2, 1), Tup3(2, 3, 2), Tup3(3, 4, 2) },
                // The second clock cycle removes the edge |1| -1-> |2|.
                zset! { Tup3(1, 2, 1) => -1 },
            ]
            .into_iter();

            let edges = root_circuit.add_source(Generator::new(move || edges_data.next().unwrap()));

            // Create a base relation with all paths of length 1.
            let len_1 = edges
                .map(|Tup3(from, to, weight)| Tup4(from.clone(), to.clone(), weight.clone(), 1));

            let closure = root_circuit.recursive(
                |child_circuit, len_n_minus_1: Stream<_, OrdZSet<Tup4<usize, usize, usize, usize>>>| {
                    // Import the `edges` and `len_1` relation from the parent circuit.
                    let edges = edges.delta0(child_circuit);
                    let len_1 = len_1.delta0(child_circuit);

                    // Perform an iterative step (n-1 to n) through joining the
                    // paths of length n-1 with the edges.
                    let len_n = len_n_minus_1
                        .map_index(|Tup4(start, end, cum_weight, hopcnt)| {
                            (
                                end.clone(),
                                Tup4(start.clone(), end.clone(), cum_weight.clone(), hopcnt.clone()),
                            )
                        })
                        .join(
                            &edges.map_index(|Tup3(from, to, weight)| {
                                (from.clone(), Tup3(from.clone(), to.clone(), weight.clone()))
                            }),
                            |_end_from,
                             Tup4(start, _end, cum_weight, hopcnt),
                             Tup3(_from, to, weight)| {
                                Tup4(start.clone(), to.clone(), cum_weight + weight, hopcnt + 1)
                            },
                        )
                        .plus(&len_1);

                    Ok(len_n)
                },
            )?;

            let mut expected_outputs = vec![
                // We expect the full transitive closure in the first clock cycle.
                zset! {
                    Tup4(0, 1, 1, 1) => 1,
                    Tup4(0, 2, 2, 2) => 1,
                    Tup4(0, 3, 4, 3) => 1,
                    Tup4(0, 4, 6, 4) => 1,
                    Tup4(1, 2, 1, 1) => 1,
                    Tup4(1, 3, 3, 2) => 1,
                    Tup4(1, 4, 5, 3) => 1,
                    Tup4(2, 3, 2, 1) => 1,
                    Tup4(2, 4, 4, 2) => 1,
                    Tup4(3, 4, 2, 1) => 1,
                },
                // These paths are removed in the second clock cycle.
                zset! {
                    Tup4(0, 2, 2, 2) => -1,
                    Tup4(0, 3, 4, 3) => -1,
                    Tup4(0, 4, 6, 4) => -1,
                    Tup4(1, 2, 1, 1) => -1,
                    Tup4(1, 3, 3, 2) => -1,
                    Tup4(1, 4, 5, 3) => -1,
                },
            ]
            .into_iter();

            closure.inspect(move |output| {
                assert_eq!(*output, expected_outputs.next().unwrap());
            });

            Ok(closure.output())
        })?;

        for _ in 0..2 {
            circuit.step()?;
        }

        Ok(())
    }

    // Taken from the [DBSP docs](https://docs.rs/dbsp/latest/dbsp/circuit/circuit_builder/struct.ChildCircuit.html#method.recursive).
    #[test]
    fn test_recursive() -> Result<(), anyhow::Error> {
        // Propagate labels along graph edges.
        let (circuit, output_handle) = RootCircuit::build(move |root_circuit| {
            // Graph topology.
            let mut edges = vec![
                // Start with four nodes connected in a cycle.
                zset_set! { Tup2(1, 2), Tup2(2, 3), Tup2(3, 4), Tup2(4, 1) },
                // Add an edge.
                zset_set! { Tup2(4, 5) },
                // Remove an edge, breaking the cycle.
                zset! { Tup2(1, 2) => -1 },
            ]
            .into_iter();

            let edges = root_circuit.add_source(Generator::new(move || edges.next().unwrap()));

            // Initial labeling of the graph.
            let mut init_labels = vec![
                // Start with a single label on node 1.
                zset_set! { Tup2(1, "l1".to_string()) },
                // Add a label to node 2.
                zset_set! { Tup2(2, "l2".to_string()) },
                zset! {},
            ]
            .into_iter();

            let init_labels =
                root_circuit.add_source(Generator::new(move || init_labels.next().unwrap()));

            let labels = root_circuit
                .recursive(
                    |child_circuit, labels: Stream<_, OrdZSet<Tup2<u64, String>>>| {
                        // Import `edges` and `init_labels` relations from the parent circuit.
                        let edges = edges.delta0(child_circuit);
                        let init_labels = init_labels.delta0(child_circuit);

                        // Given an edge `from -> to` where the `from` node is labeled with `l`,
                        // propagate `l` to node `to`.
                        let result = labels
                            .map_index(|Tup2(x, y)| (x.clone(), y.clone()))
                            .join(
                                &edges.map_index(|Tup2(x, y)| (x.clone(), y.clone())),
                                |_from, l, to| Tup2(*to, l.clone()),
                            )
                            .plus(&init_labels);
                        Ok(result)
                    },
                )
                .unwrap();

            // Expected _changes_ to the output graph labeling after each clock cycle.
            let mut expected_outputs = vec![
                zset! { Tup2(1, "l1".to_string()) => 1, Tup2(2, "l1".to_string()) => 1, Tup2(3, "l1".to_string()) => 1, Tup2(4, "l1".to_string()) => 1 },
                zset! { Tup2(1, "l2".to_string()) => 1, Tup2(2, "l2".to_string()) => 1, Tup2(3, "l2".to_string()) => 1, Tup2(4, "l2".to_string()) => 1, Tup2(5, "l1".to_string()) => 1, Tup2(5, "l2".to_string()) => 1 },
                zset! { Tup2(2, "l1".to_string()) => -1, Tup2(3, "l1".to_string()) => -1, Tup2(4, "l1".to_string()) => -1, Tup2(5, "l1".to_string()) => -1 },
            ]
            .into_iter();

            labels.inspect(move |ls| {
                assert_eq!(*ls, expected_outputs.next().unwrap());
            });
            Ok(labels.output())
        })?;

        for _ in 0..3 {
            circuit.step()?;
            let x = output_handle.consolidate().iter().collect::<Vec<_>>();
            println!("Output: {:?}", x);
        }

        Ok(())
    }

    // TODO: Provide a PR to DBSP because their example is buggy and panics.
    // Computes the factorial of the first 10 numbers.
    #[test]
    fn test_factorial_with_iterate() -> Result<(), anyhow::Error> {
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

    // Computes the sum of the first n natural numbers.
    #[test]
    fn test_sum_n_natural_numbers_with_add_feedback() -> Result<(), anyhow::Error> {
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
