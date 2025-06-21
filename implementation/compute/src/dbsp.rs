use crate::{
    expr::{Literal, LiteralExpr},
    relation::{Relation, RelationSchema, SchemaTuple, TupleKey, TupleValue},
};
use cli_table::{Cell, Style, Table, format::Justify};
pub use dbsp::{
    CircuitHandle, DBSPHandle as DbspHandle, Error as DbspError, NestedCircuit, RootCircuit,
    Runtime, ZWeight, zset,
};
use dbsp::{IndexedZSetHandle, OrdIndexedZSet, OrdZSet, OutputHandle, Stream, utils::Tup2};
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    iter,
};

type OrdStream = Stream<RootCircuit, OrdZSet<TupleValue>>;

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

        let selected = self.a.inner.filter(move |tuple| {
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

        let joined = selected.join_index(&self.b.inner, |k, left, right| {
            // merge left and right tuple
            Some((k.clone(), right.clone()))
        });

        let projected = joined.map_index(|(k, tuple)| {
            // project tuple
            (k.clone(), tuple.clone())
        });
    }
}

pub fn new_ord_indexed_stream(
    circuit: &mut RootCircuit,
) -> (OrdIndexedRootStream, OrdIndexedStreamInputHandle) {
    circuit.add_input_indexed_zset::<TupleKey, TupleValue>()
}

pub type OrdIndexedStreamInputHandle = IndexedZSetHandle<TupleKey, TupleValue>;

pub type OrdIndexedStreamOutputHandle = OutputHandle<OrdIndexedZSet<TupleKey, TupleValue>>;

pub type OrdIndexedStream<Circuit> = Stream<Circuit, OrdIndexedZSet<TupleKey, TupleValue>>;

pub type OrdIndexedRootStream = OrdIndexedStream<RootCircuit>;
pub type OrdIndexedNestedStream = OrdIndexedStream<NestedCircuit>;

/// A wrapper of DBSP's streams carrying [`dbsp::OrdIndexedZSet`] but generic-free
/// over the circuit type. This limits the nesting level to one level but this
/// does not matter for practical applications.
#[derive(Clone)]
pub enum StreamWrapper {
    Root(OrdIndexedRootStream),
    Nested(OrdIndexedNestedStream),
}

impl StreamWrapper {
    pub fn distinct(&self) -> StreamWrapper {
        match self {
            Self::Root(stream) => Self::Root(stream.distinct()),
            Self::Nested(stream) => Self::Nested(stream.distinct()),
        }
    }

    pub fn sum<'a, I>(&'a self, streams: I) -> StreamWrapper
    where
        I: IntoIterator<Item = &'a Self>,
    {
        match self {
            Self::Root(stream) => {
                Self::Root(stream.sum(streams.into_iter().map(|s| s.expect_root())))
            }
            Self::Nested(stream) => {
                Self::Nested(stream.sum(streams.into_iter().map(|s| s.expect_nested())))
            }
        }
    }

    pub fn minus(&self, other: &Self) -> Self {
        match self {
            Self::Root(stream) => Self::Root(stream.minus(other.expect_root())),
            Self::Nested(stream) => Self::Nested(stream.minus(other.expect_nested())),
        }
    }

    pub fn map_index<F>(&self, map_func: F) -> StreamWrapper
    where
        F: Fn((&TupleKey, &TupleValue)) -> (TupleKey, TupleValue) + 'static,
    {
        match self {
            Self::Root(stream) => Self::Root(stream.map_index(map_func)),
            Self::Nested(stream) => Self::Nested(stream.map_index(map_func)),
        }
    }

    pub fn filter<F>(&self, filter_func: F) -> Self
    where
        F: Fn((&TupleKey, &TupleValue)) -> bool + 'static,
    {
        match self {
            Self::Root(stream) => Self::Root(stream.filter(filter_func)),
            Self::Nested(stream) => Self::Nested(stream.filter(filter_func)),
        }
    }

    pub fn join_index<F, It>(&self, other: &Self, join: F) -> Self
    where
        F: Fn(&TupleKey, &TupleValue, &TupleValue) -> It + Clone + 'static,
        It: IntoIterator<Item = (TupleKey, TupleValue)> + 'static,
    {
        match self {
            Self::Root(stream) => Self::Root(stream.join_index(other.expect_root(), join)),
            Self::Nested(stream) => Self::Nested(stream.join_index(other.expect_nested(), join)),
        }
    }

    pub fn anti_join_index(&self, other: &Self) -> Self {
        match self {
            Self::Root(stream) => Self::Root(stream.antijoin(other.expect_root())),
            Self::Nested(stream) => Self::Nested(stream.antijoin(other.expect_nested())),
        }
    }

    /// The delta0 operator imports a stream from the parent circuit into the
    /// child circuit.
    pub fn delta0(&self, child_circuit: &NestedCircuit) -> Self {
        match self {
            // Transitions from RootStream to NestedStream
            Self::Root(stream) => Self::Nested(stream.delta0(child_circuit)),
            Self::Nested(stream) => panic!("No further nesting for beyond NestedStreams"),
        }
    }

    pub fn output(&self) -> OrdIndexedStreamOutputHandle {
        match self {
            Self::Root(stream) => stream.output(),
            Self::Nested(stream) => panic!("Nested streams do not support output()"),
        }
    }

    pub fn expect_root(&self) -> &OrdIndexedRootStream {
        if let Self::Root(stream) = self {
            stream
        } else {
            panic!("Expected RootStream")
        }
    }

    pub fn expect_nested(&self) -> &OrdIndexedNestedStream {
        if let Self::Nested(stream) = self {
            stream
        } else {
            panic!("Expected NestedStream")
        }
    }
}

impl From<OrdIndexedRootStream> for StreamWrapper {
    fn from(stream: OrdIndexedRootStream) -> Self {
        Self::Root(stream)
    }
}

impl From<OrdIndexedNestedStream> for StreamWrapper {
    fn from(stream: OrdIndexedNestedStream) -> Self {
        Self::Nested(stream)
    }
}

impl IntoIterator for &'_ StreamWrapper {
    type Item = Self;
    type IntoIter = std::iter::Once<Self>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self)
    }
}

#[derive(Default, Debug, Clone)]
pub struct DbspInputs {
    inputs: HashMap<String, DbspInput>,
}

impl DbspInputs {
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

#[derive(Clone)]
pub struct DbspInput {
    schema: RelationSchema,
    handle: OrdIndexedStreamInputHandle,
}

impl DbspInput {
    pub fn add(
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
    ) {
        tuples.into_iter().for_each(|(tuple, z_weight)| {
            self.handle
                .push(tuple.clone().into(), (tuple.clone().into(), z_weight))
        })
    }
    pub fn insert_with_same_weight<'a, T: Into<TupleKey> + Into<TupleValue> + Clone + 'a>(
        &self,
        tuples: impl IntoIterator<Item = &'a T>,
        z_weight: ZWeight,
    ) {
        self.insert(tuples.into_iter().map(|tuple| (tuple, z_weight)));
    }
}

impl Debug for DbspInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DbspInput")
            .field("schema", &self.schema)
            .finish()
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
    pub fn to_batch(&self) -> DbspOutputBatch {
        let inner = self.handle.consolidate().iter().collect::<Vec<_>>();
        DbspOutputBatch {
            schema: &self.schema,
            inner,
        }
    }
}

pub struct DbspOutputBatch<'a> {
    schema: &'a RelationSchema,
    inner: Vec<(TupleKey, TupleValue, ZWeight)>,
}

impl DbspOutputBatch<'_> {
    const JUSTIFICATION: Justify = Justify::Right;

    pub fn as_table(&self) -> impl Display {
        self.inner
            .iter()
            .map(|(key, tuple, weight)| {
                iter::once(weight.to_string().cell().justify(Self::JUSTIFICATION)).chain(
                    SchemaTuple::new(&self.schema.tuple, tuple)
                        .fields()
                        .map(|attribute| attribute.to_string().cell().justify(Self::JUSTIFICATION))
                        .collect::<Vec<_>>(),
                )
            })
            .table()
            .title(
                iter::once("z-weight".cell())
                    .chain(self.schema.tuple.field_names(&None).map(|name| name.cell())),
            )
            .bold(true)
            .display()
            .expect("Table error")
    }
    pub fn as_debug_table(&self) -> impl Display {
        self.inner
            .iter()
            .map(|(key, tuple, weight)| {
                // We ensure that the key and tuple data lengths match the
                // respective schema field lengths.
                debug_assert!(key.data.len() == self.schema.key.full_len());
                debug_assert!(tuple.data.len() == self.schema.tuple.full_len());
                iter::once(weight.to_string().cell().justify(Self::JUSTIFICATION))
                    .chain(
                        SchemaTuple::new(&self.schema.key, key)
                            .all_fields()
                            .map(|attribute| {
                                attribute.to_string().cell().justify(Self::JUSTIFICATION)
                            })
                            .collect::<Vec<_>>(),
                    )
                    .chain(
                        SchemaTuple::new(&self.schema.tuple, tuple)
                            .all_fields()
                            .map(|attribute| {
                                attribute.to_string().cell().justify(Self::JUSTIFICATION)
                            })
                            .collect::<Vec<_>>(),
                    )
            })
            .table()
            .title(
                iter::once("z-weight".cell())
                    .chain(
                        self.schema
                            .key
                            .all_field_names(&None)
                            .map(|name| format!("[key] {}", name).cell()),
                    )
                    .chain(
                        self.schema
                            .tuple
                            .all_field_names(&None)
                            .map(|name| format!("[value] {}", name).cell()),
                    ),
            )
            .bold(true)
            .display()
            .expect("Table error")
    }
    pub fn as_data(&self) -> impl Iterator<Item = (ZWeight, &TupleValue)> {
        self.inner
            .iter()
            .map(|(_key, tuple, weight)| (*weight, tuple))
    }
    pub fn as_zset(&self) -> OrdZSet<TupleValue> {
        let keys = self
            .inner
            .iter()
            .map(|(_key, tuple, weight)| {
                let tuple: TupleValue = SchemaTuple::new(&self.schema.tuple, tuple)
                    .fields()
                    .cloned()
                    .collect();
                Tup2(tuple, *weight)
            })
            .collect::<Vec<_>>();
        OrdZSet::from_keys((), keys)
    }
    pub fn as_debug_zset(&self) -> OrdZSet<TupleValue> {
        let keys = self
            .inner
            .iter()
            .map(|(_key, tuple, weight)| Tup2(tuple.clone(), *weight))
            .collect::<Vec<_>>();
        OrdZSet::from_keys((), keys)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use dbsp::{
        Circuit, indexed_zset,
        operator::{Generator, Z1},
        utils::{Tup2, Tup3, Tup4},
        zset, zset_set,
    };
    use std::{cell::RefCell, rc::Rc};

    // TODO: Provide a PR to DBSP with this example.
    // Note that this example only works with acyclic graphs.
    #[test]
    fn test_transitive_closure() -> Result<(), anyhow::Error> {
        const STEPS: usize = 2;

        let (circuit_handle, output_handle) = RootCircuit::build(move |root_circuit| {
            let mut edges_data = ([
                // The first clock cycle adds a graph of four nodes:
                // |0| -1-> |1| -1-> |2| -2-> |3| -2-> |4|
                zset_set! { Tup3(0_usize, 1_usize, 1_usize), Tup3(1, 2, 1), Tup3(2, 3, 2), Tup3(3, 4, 2) },
                // The second clock cycle removes the edge |1| -1-> |2|.
                zset! { Tup3(1, 2, 1) => -1 },
                // The third clock cycle would introduce a cycle but that would
                // cause the fixpoint computation to never terminate.
                // In total, we have the following graph:
                // |0| -1-> |1| -1-> |2| -2-> |3| -2-> |4|
                //  ^                                   |
                //  |                                   |
                //  ------------------3------------------
                // zset_set! { Tup3(1,2,1), Tup3(4, 0, 3)}
            ] as [_; STEPS])
            .into_iter();

            let edges = root_circuit.add_source(Generator::new(move || edges_data.next().unwrap()));

            // Create a base relation with all paths of length 1.
            let len_1 = edges.map(|Tup3(from, to, weight)| Tup4(*from, *to, *weight, 1));

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
                                *end,
                                Tup4(*start, *end, *cum_weight, *hopcnt),
                            )
                        })
                        .join(
                            &edges.map_index(|Tup3(from, to, weight)| {
                                (*from, Tup3(*from, *to, *weight))
                            }),
                            |_end_from,
                             Tup4(start, _end, cum_weight, hopcnt),
                             Tup3(_from, to, weight)| {
                                Tup4(*start, *to, cum_weight + weight, hopcnt + 1)
                            },
                        ).plus(&len_1);

                    Ok(len_n)
                },
            )?;

            let mut expected_outputs = ([
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
                // This does not matter anymore, as the computation does not
                // terminate anymore due to the cycle.
                // zset! {},
            ] as [_; STEPS])
                .into_iter();

            closure.inspect(move |output| {
                assert_eq!(*output, expected_outputs.next().unwrap());
            });

            Ok(closure.output())
        })?;

        for _ in 0..STEPS {
            circuit_handle.step()?;
        }

        Ok(())
    }

    // Taken from the [DBSP docs](https://docs.rs/dbsp/latest/dbsp/circuit/circuit_builder/struct.ChildCircuit.html#method.recursive).
    #[test]
    fn test_recursive() -> Result<(), anyhow::Error> {
        const STEPS: usize = 3;

        // Propagate labels along graph edges.
        let (circuit, output_handle) = RootCircuit::build(move |root_circuit| {
            // Graph topology.
            let mut edges = ([
                // Start with four nodes connected in a cycle.
                zset_set! { Tup2(1, 2), Tup2(2, 3), Tup2(3, 4), Tup2(4, 1) },
                // Add an edge.
                zset_set! { Tup2(4, 5) },
                // Remove an edge, breaking the cycle.
                zset! { Tup2(1, 2) => -1 },
            ] as [_; STEPS])
                .into_iter();

            let edges = root_circuit.add_source(Generator::new(move || edges.next().unwrap()));

            // Initial labeling of the graph.
            let mut init_labels = ([
                // Start with a single label on node 1.
                zset_set! { Tup2(1, "l1".to_string()) },
                // Add a label to node 2.
                zset_set! { Tup2(2, "l2".to_string()) },
                zset! {},
            ] as [_; STEPS])
                .into_iter();

            let init_labels =
                root_circuit.add_source(Generator::new(move || init_labels.next().unwrap()));

            let labels = root_circuit.recursive(
                |child_circuit, labels: Stream<_, OrdZSet<Tup2<u64, String>>>| {
                    // Import `edges` and `init_labels` relations from the parent circuit.
                    let edges = edges.delta0(child_circuit);
                    let init_labels = init_labels.delta0(child_circuit);

                    // Given an edge `from -> to` where the `from` node is labeled with `l`,
                    // propagate `l` to node `to`.
                    let result = labels
                        .map_index(|Tup2(x, y)| (*x, y.clone()))
                        .join(&edges.map_index(|Tup2(x, y)| (*x, *y)), |_from, l, to| {
                            Tup2(*to, l.clone())
                        })
                        .plus(&init_labels);
                    Ok(result)
                },
            )?;

            // Expected _changes_ to the output graph labeling after each clock cycle.
            let mut expected_outputs = ([
                zset! { Tup2(1, "l1".to_string()) => 1, Tup2(2, "l1".to_string()) => 1, Tup2(3, "l1".to_string()) => 1, Tup2(4, "l1".to_string()) => 1 },
                zset! { Tup2(1, "l2".to_string()) => 1, Tup2(2, "l2".to_string()) => 1, Tup2(3, "l2".to_string()) => 1, Tup2(4, "l2".to_string()) => 1, Tup2(5, "l1".to_string()) => 1, Tup2(5, "l2".to_string()) => 1 },
                zset! { Tup2(2, "l1".to_string()) => -1, Tup2(3, "l1".to_string()) => -1, Tup2(4, "l1".to_string()) => -1, Tup2(5, "l1".to_string()) => -1 },
            ] as [_; STEPS])
            .into_iter();

            labels.inspect(move |ls| {
                assert_eq!(*ls, expected_outputs.next().unwrap());
            });

            Ok(labels.output())
        })?;

        for _ in 0..STEPS {
            circuit.step()?;
            let x = output_handle.consolidate().iter().collect::<Vec<_>>();
            println!("Output: {:?}", x);
        }

        Ok(())
    }

    // Computes the factorial of the first 10 numbers.
    #[test]
    fn test_factorial_with_iterate() -> Result<(), anyhow::Error> {
        let (circuit, output) = RootCircuit::build(|circuit| {
            // Generate sequence 0, 1, 2, ...
            let mut n: usize = 0;
            let source = circuit.add_source(Generator::new(move || {
                let result = n;
                n += 1;
                result
            }));
            // Compute factorial of each number in the sequence.
            let fact = circuit.iterate(|child| {
                let counter = Rc::new(RefCell::new(1));
                let counter_clone = Rc::clone(&counter);
                let countdown = source.delta0(child).apply(move |parent_val| {
                    let mut counter_borrow = counter_clone.borrow_mut();
                    *counter_borrow += *parent_val;
                    let res = *counter_borrow;
                    *counter_borrow -= 1;
                    res
                });
                let (z1_output, z1_feedback) = child.add_feedback_with_export(Z1::new(1));
                let multiplication =
                    countdown.apply2(&z1_output.local, |n1: &usize, n2: &usize| n1 * n2);
                z1_feedback.connect(&multiplication);
                // Stop iterating when the counter reaches 0.
                Ok((move || Ok(*counter.borrow() == 0), z1_output.export))
            })?;
            Ok(fact.output())
        })?;

        let factorial = |n: usize| (1..=n).product::<usize>();
        const ITERATIONS: usize = 10;
        for i in 0..ITERATIONS {
            circuit.step()?;
            let result = output.take_from_all();
            let result = result.first().unwrap();
            println!("Iteration {:3}: {:3}! = {}", i + 1, i, result);
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
                n += 1;
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

    #[test]
    fn test_not_operator() -> Result<(), anyhow::Error> {
        let (circuit, output) = RootCircuit::build(|root_circuit| {
            let mut left = [
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                    Tup2(1, 1) => { Tup2(1, 1) => 1 },
                    Tup2(1, 2) => { Tup2(1, 2) => 1 },
                    Tup2(1, 3) => { Tup2(1, 3) => 1 },
                    Tup2(1, 4) => { Tup2(1, 4) => 1 },
                },
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                },
            ]
            .into_iter();
            let left = root_circuit.add_source(Generator::new(move || left.next().unwrap()));

            let mut right = [
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                    Tup2(1, 2) => { Tup2(1, 2) => 1 },
                    Tup2(1, 3) => { Tup2(1, 3) => 1 },
                },
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                    Tup2(1, 4) => { Tup2(1, 4) => 1 },
                },
            ]
            .into_iter();
            let right = root_circuit.add_source(Generator::new(move || right.next().unwrap()));

            let set_minus = left.minus(&right);

            Ok(set_minus.output())
        })?;

        let iterations = 2;
        for i in 1..=iterations {
            circuit.step()?;
            let result = output.take_from_all();
            let result = result.first().unwrap();
            println!("{:?}", result);
        }

        Ok(())
    }

    #[test]
    fn test_cartesian_product() -> Result<(), anyhow::Error> {
        const STEPS: usize = 2;

        let (circuit, output) = RootCircuit::build(|root_circuit| {
            let mut left = ([
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                    Tup2(1, 1) => { Tup2(1, 1) => 1 },
                    Tup2(1, 1) => { Tup2(1, 1) => 1 }, // duplicate of the above!
                    Tup2(1, 2) => { Tup2(1, 2) => 1 },
                    Tup2(1, 3) => { Tup2(1, 3) => 1 },
                },
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                },
            ] as [_; STEPS])
                .into_iter();
            let left = root_circuit.add_source(Generator::new(move || left.next().unwrap()));
            let left = left.map_index(|(_k, v)| ((), *v));

            let mut right = ([
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                    Tup2(2, 1) => { Tup2(2, 1) => 1 },
                    Tup2(2, 2) => { Tup2(2, 2) => 1 },
                },
                indexed_zset! {Tup2<usize, usize> => Tup2<usize, usize>:
                    Tup2(2, 3) => { Tup2(2, 3) => 1 },
                },
            ] as [_; STEPS])
                .into_iter();
            let right = root_circuit.add_source(Generator::new(move || right.next().unwrap()));
            let right = right.map_index(|(_k, v)| ((), *v));

            let cartesian_product = left.join_index(&right, |_k, Tup2(l1, l2), Tup2(r1, r2)| {
                // Merge left and right tuples.
                Some(((), Tup4(*l1, *l2, *r1, *r2)))
            });

            Ok(cartesian_product.output())
        })?;

        let mut expected_outputs = ([
            indexed_zset! {() => Tup4<usize, usize, usize, usize>:
                () => { Tup4(1, 1, 2, 1) => 2 },
                () => { Tup4(1, 1, 2, 2) => 2 },
                () => { Tup4(1, 2, 2, 1) => 1 },
                () => { Tup4(1, 2, 2, 2) => 1 },
                () => { Tup4(1, 3, 2, 1) => 1 },
                () => { Tup4(1, 3, 2, 2) => 1 },
            },
            indexed_zset! {() => Tup4<usize, usize, usize, usize>:
                () => { Tup4(1, 1, 2, 3) => 2 },
                () => { Tup4(1, 2, 2, 3) => 1 },
                () => { Tup4(1, 3, 2, 3) => 1 },
            },
        ] as [_; STEPS])
            .into_iter();

        for i in 1..=STEPS {
            circuit.step()?;
            let result = output.take_from_all();
            let result = result.first().unwrap();
            println!("{:?}", result);
            assert_eq!(*result, expected_outputs.next().unwrap());
        }

        Ok(())
    }
}
