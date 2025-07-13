use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parser::key_value_store_crdts::{
    KeyValueStoreOperation, KeyValueStoreReplica, MVR_KV_STORE_CRDT_DATALOG, MVR_KV_STORE_DATALOG,
    PredOp, SetOp,
};
use std::collections::{HashMap, HashSet};

const CRDTS: [(&str, &str); 2] = [
    ("wo_cb", MVR_KV_STORE_DATALOG),
    ("w_cb", MVR_KV_STORE_CRDT_DATALOG),
];

const REPLICA_ID: u64 = 1;
const BASE_DIAMETERS: [usize; 5] = [10_000, 20_000, 30_000, 40_000, 50_000];
const DELTA_DIAMETERS: [usize; 5] = [20, 40, 60, 80, 100];

fn bench_hydration(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_stores_hydration_setting");

    for (crdt, datalog) in CRDTS {
        for diameter in BASE_DIAMETERS {
            let name = format!("crdt={crdt} base_diameter={diameter}");
            let mut generating_replica = KeyValueStoreReplica::new(REPLICA_ID, datalog);
            let (set_ops, pred_ops) = collect_history(&mut generating_replica, diameter);
            group.throughput(Throughput::Elements(diameter as u64));
            group.bench_with_input(
                BenchmarkId::new(name, diameter),
                &diameter,
                |b, &diameter| {
                    b.iter_batched(
                        || KeyValueStoreReplica::new(REPLICA_ID, datalog),
                        |mut replica| {
                            replica.feed_ops(&set_ops, &pred_ops);
                            let result = replica.derive_state();
                            debug_assert_eq!(
                                result,
                                &HashMap::from([(0, HashSet::from([(diameter - 1) as u64]))])
                            )
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
    let mut group = c.benchmark_group("kv_stores_near_real_time_setting");

    for (crdt, datalog) in CRDTS {
        for base_diameter in BASE_DIAMETERS {
            for delta_diameter in DELTA_DIAMETERS {
                let name = format!(
                    "crdt={crdt} base_diameter={base_diameter} delta_diameter={delta_diameter}"
                );
                let mut generating_replica = KeyValueStoreReplica::new(REPLICA_ID, datalog);
                let (base_set_ops, base_pred_ops) =
                    collect_history(&mut generating_replica, base_diameter);
                let (delta_set_ops, delta_pred_ops) =
                    collect_history(&mut generating_replica, delta_diameter);
                group.throughput(Throughput::Elements(delta_diameter as u64));
                group.bench_with_input(
                    BenchmarkId::new(name, delta_diameter),
                    &delta_diameter,
                    |b, delta_diameter| {
                        b.iter_batched(
                            || {
                                let mut replica = KeyValueStoreReplica::new(REPLICA_ID, datalog);
                                replica.feed_ops(&base_set_ops, &base_pred_ops);
                                let result = replica.derive_state();
                                debug_assert_eq!(
                                    result,
                                    &HashMap::from([(
                                        0,
                                        HashSet::from([(base_diameter - 1) as u64])
                                    )])
                                );
                                replica
                            },
                            |mut replica| {
                                replica.feed_ops(&delta_set_ops, &delta_pred_ops);
                                let result = replica.derive_state();
                                debug_assert_eq!(
                                    result,
                                    &HashMap::from([(
                                        0,
                                        HashSet::from(
                                            [(base_diameter + delta_diameter - 1) as u64]
                                        )
                                    )])
                                );
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

/// Generates a causal history of diameter `diameter` by generating set operations
/// that set the key 0 to the value of the replica's counter at the respective time.
fn collect_history(
    replica: &mut KeyValueStoreReplica,
    diameter: usize,
) -> (Vec<SetOp>, Vec<PredOp>) {
    let mut set_ops = Vec::new();
    let mut pred_ops = Vec::new();

    // No need to do a full simulation here, as we can just generate the operations,
    // unlike in a real-time setting or with the list CRDT.
    // This would change once the heads are tracked via Datalog though.
    (0..diameter)
        .map(|_i| replica.generate_ops(&KeyValueStoreOperation::Set(0, replica.ctr())))
        .for_each(|(set_op, pred_rels)| {
            set_ops.push(set_op);
            pred_ops.extend(pred_rels);
        });

    (set_ops, pred_ops)
}

criterion_group!(benches, bench_hydration, bench_near_real_time);
criterion_main!(benches);
