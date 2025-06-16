use compute::{
    dbsp::{DbspHandle, DbspInputs, DbspOutput},
    test_helper::{PredRel, Replica, SetOp},
    IncDataLog,
};
use criterion::{criterion_group, criterion_main, Criterion};
use parser::{crdts::mvr_crdt_store_datalog, Parser};
use std::num::NonZeroUsize;

fn bench_hydration(c: &mut Criterion) {
    let threads = [1, 2, 4, 8]
        .into_iter()
        .map(NonZeroUsize::try_from)
        .collect::<Result<Vec<NonZeroUsize>, _>>()
        .expect("Non-zero thread count");
    let diameters = [100, 10_000];

    let mut group = c.benchmark_group("mvr_crdt_store_datalog: hydration");

    for thread_count in threads {
        for diameter in diameters {
            let name = format!("diameter {diameter}, threads {thread_count:?}");
            group.bench_with_input(name, &diameter, |b, &diameter| {
                let mut replica = Replica::new(0);
                let (set_op_data, pred_rel_data) =
                    generate_operation_history(&mut replica, diameter);

                b.iter(|| {
                    let (mut handle, inputs, _output) = prepare_circuit(thread_count);
                    let set_op_input = inputs.get("set").unwrap();
                    let pred_rel_input = inputs.get("pred").unwrap();
                    for (pred_rel_step, set_op_step) in pred_rel_data.iter().zip(set_op_data.iter())
                    {
                        pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
                        set_op_input.insert_with_same_weight([set_op_step], 1);
                    }
                    handle.step().unwrap();
                    // println!("{}", output.to_batch().as_debug_table());
                });
            });
        }
    }

    group.finish();
}

fn generate_operation_history(
    replica: &mut Replica,
    diameter: usize,
) -> (Vec<SetOp>, Vec<Vec<PredRel>>) {
    (0..diameter)
        .map(|i| {
            let (set_op, pred_rels) = replica.new_local_set_op(0, i as u64);
            (set_op, pred_rels)
        })
        .unzip()
}

fn prepare_circuit(threads: NonZeroUsize) -> (DbspHandle, DbspInputs, DbspOutput) {
    IncDataLog::new(threads, true)
        .build_circuit_from_parser(|root_circuit| {
            Parser::new(root_circuit).parse(mvr_crdt_store_datalog())
        })
        .unwrap()
}

criterion_group!(benches, bench_hydration);
criterion_main!(benches);
