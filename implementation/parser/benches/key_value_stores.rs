use compute::{
    IncDataLog,
    dbsp::{CircuitHandle, DbspInputs, DbspOutput},
    test_helper::{PredRel, Replica, SetOp},
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parser::{
    Parser,
    crdts::{mvr_crdt_store_datalog, mvr_store_datalog},
};
use std::num::NonZeroUsize;

const CRDTS: [(&str, &str); 2] = [
    ("without_causal_broadcast", mvr_store_datalog()),
    ("with_causal_broadcast", mvr_crdt_store_datalog()),
];

fn bench_hydration(c: &mut Criterion) {
    let diameters = [1000_usize, 2000, 3000, 4000, 5000];

    let mut group = c.benchmark_group("MVR_stores_hydration_setting");

    for (crdt, datalog) in CRDTS {
        for diameter in diameters {
            let name = format!("CRDT={crdt}");
            group.throughput(Throughput::Elements(diameter as u64));
            group.bench_with_input(
                BenchmarkId::new(name, diameter),
                &diameter,
                |b, &diameter| {
                    let mut replica = Replica::new(0);
                    let data =
                        generate_operation_history(&mut replica, diameter).collect::<Vec<_>>();

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
    }

    group.finish();
}

fn bench_near_real_time(c: &mut Criterion) {
    let base_diameters = [1000_usize, 2000, 3000, 4000, 5000];
    let delta_diameters = [20_usize, 40, 60, 80, 100];

    let mut group = c.benchmark_group("MVR_stores_near_real_time_setting");

    for (crdt, datalog) in CRDTS {
        for base_diameter in base_diameters {
            for delta_diameter in delta_diameters {
                let name = format!("CRDT={crdt}_base={base_diameter}");
                group.throughput(Throughput::Elements(delta_diameter as u64));
                group.bench_with_input(
                    BenchmarkId::new(name, delta_diameter),
                    &delta_diameter,
                    |b, delta_diameter| {
                        let mut replica = Replica::new(0);
                        let base_data = generate_operation_history(&mut replica, base_diameter)
                            .collect::<Vec<_>>();
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
    }

    group.finish();
}

/// Returns set operations and according predecessor relations by setting
/// the key 0 to the value of the replica's counter at the respective time.
fn generate_operation_history(
    replica: &mut Replica,
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
