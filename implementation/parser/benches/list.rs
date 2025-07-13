use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parser::list_crdt::{AssignOp, InsertOp, ListReplica, RemoveOp};
use std::ops::Range;
use trace_provider::{
    TraceProvider,
    from_string::{StringTrace, TEST_STRING},
};

mod trace_provider;

const REPLICA_ID: u64 = 1;
const BASE_TEXT_LENS: [usize; 5] = [10_000, 20_000, 30_000, 40_000, 50_000];
const DELTA_TEXT_LENS: [usize; 5] = [20, 40, 60, 80, 100];

fn bench_hydration(c: &mut Criterion) {
    let mut group = c.benchmark_group("list_hydration_setting");

    for base_text_len in BASE_TEXT_LENS {
        let name = format!("base_len={base_text_len}");
        let trace = StringTrace::with_len(TEST_STRING, base_text_len);
        let mut generator_replica = ListReplica::new(REPLICA_ID);
        let (insert_ops, assign_ops, remove_ops) =
            collect_history(&mut generator_replica, &trace, 0..base_text_len);
        group.throughput(Throughput::Elements(base_text_len as u64));
        group.bench_with_input(
            BenchmarkId::new(name, base_text_len),
            &base_text_len,
            |b, &_base_diameter| {
                b.iter_batched(
                    || ListReplica::new(REPLICA_ID),
                    |mut replica| {
                        replica.feed_ops(&insert_ops, &assign_ops, &remove_ops);
                        let result = replica.derive_state();
                        debug_assert_eq!(result, trace.final_result());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_near_real_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("list_near_real_time_setting");

    for base_text_len in BASE_TEXT_LENS {
        for delta_text_len in DELTA_TEXT_LENS {
            let name = format!("base_len={base_text_len} delta_len={delta_text_len}");
            let total_len = base_text_len + delta_text_len;
            let trace = StringTrace::with_len(TEST_STRING, total_len);
            let mut generator_replica = ListReplica::new(REPLICA_ID);
            let (base_insert_ops, base_assign_ops, base_remove_ops) =
                collect_history(&mut generator_replica, &trace, 0..base_text_len);
            let (delta_insert_ops, delta_assign_ops, delta_remove_ops) =
                collect_history(&mut generator_replica, &trace, base_text_len..total_len);
            group.throughput(Throughput::Elements(delta_text_len as u64));
            group.bench_with_input(
                BenchmarkId::new(name, delta_text_len),
                &delta_text_len,
                |b, _delta_diameter| {
                    b.iter_batched(
                        || {
                            let mut replica = ListReplica::new(REPLICA_ID);
                            replica.feed_ops(&base_insert_ops, &base_assign_ops, &base_remove_ops);
                            let result = replica.derive_state();
                            debug_assert_eq!(result, &trace.result_until(base_text_len));
                            replica
                        },
                        |mut replica| {
                            replica.feed_ops(
                                &delta_insert_ops,
                                &delta_assign_ops,
                                &delta_remove_ops,
                            );
                            let result = replica.derive_state();
                            debug_assert_eq!(result, trace.final_result());
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

fn collect_history(
    replica: &mut ListReplica,
    trace: &impl TraceProvider,
    range: Range<usize>,
) -> (Vec<InsertOp>, Vec<AssignOp>, Vec<RemoveOp>) {
    let from = range.start;
    let mut insert_ops = Vec::new();
    let mut assign_ops = Vec::new();
    let mut remove_ops = Vec::new();

    for (idx, local_list_event) in trace.list_ops_range(range).iter().enumerate() {
        let (insert_op, assign_op, remove_op) = replica.generate_ops(local_list_event);
        replica.feed_ops(&insert_op, &assign_op, &remove_op);
        let result = replica.derive_state();
        debug_assert_eq!(result, &trace.result_until(from + idx + 1));
        insert_ops.extend(insert_op);
        assign_ops.extend(assign_op);
        remove_ops.extend(remove_op);
    }
    (insert_ops, assign_ops, remove_ops)
}

criterion_group!(benches, bench_hydration, bench_near_real_time);
criterion_main!(benches);
