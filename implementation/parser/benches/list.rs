mod parse_trace;

use compute::{
    IncDataLog,
    dbsp::{CircuitHandle, DbspInputs, DbspOutput},
    test_helper::{AssignOp, InsertOp, KeyValueStoreReplica, ListReplica, PredRel, SetOp},
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parser::{Parser, crdts::list_crdt_datalog};
use std::num::NonZeroUsize;

mod helper {
    use crate::parse_trace::parse_trace;
    use compute::test_helper::ListOperation;
    use std::{fs::File, io::BufReader};

    pub fn read_automerge_paper_trace() -> (String, Vec<ListOperation>) {
        let file = File::open("benches/automerge-paper.json").expect("Failed to open trace file");
        let reader = BufReader::new(file);
        let trace = parse_trace(reader).expect("Failed to parse trace file");
        let result = trace.end_content;
        let list_operations = trace
            .transactions
            .into_iter()
            .flat_map(|tx| {
                tx.patches.into_iter().map(|patch| {
                    if patch.delete_count > 0 {
                        assert!(patch.content.is_empty(), "no replace operations");
                        ListOperation::DeleteAt(patch.start)
                    } else {
                        assert!(patch.content.len() == 1, "only single character inserts");
                        assert!(patch.delete_count == 0, "no replace operations");
                        ListOperation::InsertAt(patch.start, patch.content.chars().next().unwrap())
                    }
                })
            })
            .collect::<Vec<_>>();
        (result, list_operations)
    }
}

const CRDTS: [(&str, &str); 1] = [("list", list_crdt_datalog())];

fn bench_hydration(c: &mut Criterion) {
    println!("Cwd {}", std::env::current_dir().unwrap().display());
    let trace = helper::read_automerge_paper_trace();
    let list_ops = &trace.1[0..60];
    println!("Trace with len {}: {list_ops:#?}", list_ops.len());
    let mut replica = ListReplica::new(1);

    let mut stream = list_ops.iter().peekable();
    let mut bursts = Vec::new();
    while let Some((remaining, insert_ops, assign_ops, remove_ops)) = replica.feed_ops(stream) {
        stream = remaining;
        bursts.push((insert_ops, assign_ops, remove_ops));
    }
    println!("Bursts: {bursts:#?}");
    let (handle, inputs, output) =
        prepare_circuit(NonZeroUsize::try_from(1).unwrap(), list_crdt_datalog());
    let insert_op_input = inputs.get("insert").unwrap();
    let assign_op_input = inputs.get("assign").unwrap();
    let remove_op_input = inputs.get("remove").unwrap();
    // Prepare the sentinel.
    assign_op_input.insert_with_same_weight([&AssignOp::new(0, 0, 0, 0, '#')], 1);
    for (insert_op_burst, assign_op_burst, remove_op_burst) in bursts {
        insert_op_input.insert_with_same_weight(insert_op_burst.iter(), 1);
        assign_op_input.insert_with_same_weight(assign_op_burst.iter(), 1);
        remove_op_input.insert_with_same_weight(remove_op_burst.iter(), 1);
        handle.step().unwrap();
        let result = output.to_batch();
        replica.apply_output_delta(result.as_data());
        println!("STRING:\n{}", replica.materialize_string());
    }

    panic!("Done :)");

    let mut group = c.benchmark_group("list_hydration_setting");

    for (crdt, datalog) in CRDTS {
        let name = format!("CRDT={crdt}");
        let diameter = 0;
        group.throughput(Throughput::Elements(diameter as u64));
        group.bench_with_input(
            BenchmarkId::new(name, diameter),
            &diameter,
            |b, &diameter| {
                let mut replica = KeyValueStoreReplica::new(0);
                let data = generate_operation_history(&mut replica, diameter).collect::<Vec<_>>();

                b.iter_batched(
                    || prepare_circuit(NonZeroUsize::try_from(1).unwrap(), datalog),
                    |(handle, inputs, output)| {
                        let set_op_input = inputs.get("set").unwrap();
                        let pred_rel_input = inputs.get("pred").unwrap();
                        for (set_op_step, pred_rel_step) in data.iter() {
                            pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
                            set_op_input.insert_with_same_weight([set_op_step], 1);
                        }
                        handle.step().unwrap();
                        let _result = output.to_batch();
                        output
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_near_real_time(c: &mut Criterion) {
    let base_diameters = [1000_usize, 2000, 3000, 4000, 5000];
    let delta_diameters = [20_usize, 40, 60, 80, 100];

    let mut group = c.benchmark_group("list_near_real_time_setting");

    for (crdt, datalog) in CRDTS {
        for delta_diameter in delta_diameters {
            let base_diameter = 10;
            let name = format!("CRDT={crdt}_base={base_diameter}");
            group.throughput(Throughput::Elements(delta_diameter as u64));
            group.bench_with_input(
                BenchmarkId::new(name, delta_diameter),
                &delta_diameter,
                |b, delta_diameter| {
                    let mut replica = KeyValueStoreReplica::new(0);
                    let base_data =
                        generate_operation_history(&mut replica, base_diameter).collect::<Vec<_>>();
                    let delta_data = generate_operation_history(&mut replica, *delta_diameter)
                        .collect::<Vec<_>>();

                    b.iter_batched(
                        || {
                            let (handle, inputs, output) =
                                prepare_circuit(NonZeroUsize::try_from(1).unwrap(), datalog);
                            let set_op_input = inputs.get("set").unwrap();
                            let pred_rel_input = inputs.get("pred").unwrap();
                            for (set_op_step, pred_rel_step) in base_data.iter() {
                                pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
                                set_op_input.insert_with_same_weight([set_op_step], 1);
                            }
                            handle.step().unwrap();
                            (handle, inputs, output)
                        },
                        |(handle, inputs, output)| {
                            let set_op_input = inputs.get("set").unwrap();
                            let pred_rel_input = inputs.get("pred").unwrap();
                            for (set_op_step, pred_rel_step) in delta_data.iter() {
                                pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
                                set_op_input.insert_with_same_weight([set_op_step], 1);
                            }
                            handle.step().unwrap();
                            let _result = output.to_batch();
                            output
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

/// Returns set operations and according predecessor relations by setting
/// the key 0 to the value of the replica's counter at the respective time.
fn generate_operation_history(
    replica: &mut KeyValueStoreReplica,
    diameter: usize,
) -> impl Iterator<Item = (SetOp, Vec<PredRel>)> {
    (0..diameter).map(|_i| replica.new_local_set_op(0, replica.ctr()))
}

fn prepare_circuit(
    threads: NonZeroUsize,
    code: &'static str,
) -> (CircuitHandle, DbspInputs, DbspOutput) {
    IncDataLog::new(threads, true)
        .build_circuit_from_parser(|root_circuit| Parser::new(root_circuit).parse(code))
        .unwrap()
}

criterion_group!(benches, bench_hydration, bench_near_real_time);
criterion_main!(benches);
