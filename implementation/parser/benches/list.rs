use compute::{
    IncDataLog,
    dbsp::{CircuitHandle, DbspInputs, DbspOutput},
    test_helper::{
        AssignOp, InsertOp, ListReplica, RemoveOp,
        trace_provider::{
            TraceProvider,
            from_string::{StringTrace, TEST_STRING},
        },
    },
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parser::{Parser, crdts::list_crdt_datalog};
use std::{num::NonZeroUsize, ops::Range};

const CRDTS: [(&str, &str); 1] = [("list", list_crdt_datalog())];

const BASE_TEXT_LENS: [usize; 5] = [5000, 10_000, 15_000, 20_000, 25_000];
const DELTA_TEXT_LENS: [usize; 5] = [20, 40, 60, 80, 100];

fn bench_hydration_list(c: &mut Criterion) {
    let mut group = c.benchmark_group("list_hydration_setting");

    for (_crdt, datalog) in CRDTS {
        for base_text_len in BASE_TEXT_LENS {
            let trace = StringTrace::with_len(TEST_STRING, base_text_len);

            let mut replica = ListReplica::new(1);
            let (handle, inputs, output) =
                prepare_circuit(NonZeroUsize::try_from(1).unwrap(), datalog);
            let (insert_ops, assign_ops, remove_ops) = generate_operation_history_list(
                &handle,
                &inputs,
                &output,
                &mut replica,
                &trace,
                base_text_len,
            );

            let name = format!("base={base_text_len}");
            group.throughput(Throughput::Elements(base_text_len as u64));
            group.bench_with_input(
                BenchmarkId::new(name, base_text_len),
                &base_text_len,
                |b, &_base_diameter| {
                    b.iter_batched(
                        || prepare_circuit(NonZeroUsize::try_from(1).unwrap(), datalog),
                        |(handle, inputs, output)| {
                            let insert_op_input = inputs.get("insert").unwrap();
                            let assign_op_input = inputs.get("assign").unwrap();
                            let remove_op_input = inputs.get("remove").unwrap();
                            insert_op_input.insert_with_same_weight(insert_ops.iter(), 1);
                            assign_op_input.insert_with_same_weight(assign_ops.iter(), 1);
                            remove_op_input.insert_with_same_weight(remove_ops.iter(), 1);
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

fn bench_near_real_time_list(c: &mut Criterion) {
    let mut group = c.benchmark_group("list_near_real_time_setting");

    for (crdt, datalog) in CRDTS {
        for base_text_len in BASE_TEXT_LENS {
            for delta_text_len in DELTA_TEXT_LENS {
                let trace = StringTrace::with_len(TEST_STRING, base_text_len + delta_text_len);
                let name = format!("CRDT={crdt}_base={base_text_len}");
                group.throughput(Throughput::Elements(delta_text_len as u64));
                group.bench_with_input(
                    BenchmarkId::new(name, delta_text_len),
                    &delta_text_len,
                    |b, delta_diameter| {
                        b.iter_batched(
                            || {
                                let mut replica = ListReplica::new(1);
                                let (handle, inputs, output) =
                                    prepare_circuit(NonZeroUsize::try_from(1).unwrap(), datalog);
                                let _ = generate_operation_history_list(
                                    &handle,
                                    &inputs,
                                    &output,
                                    &mut replica,
                                    &trace,
                                    base_text_len,
                                );
                                (handle, inputs, output, replica)
                            },
                            |(handle, inputs, output, mut replica)| {
                                simulate_inserts(
                                    &handle,
                                    &inputs,
                                    &output,
                                    &mut replica,
                                    &trace,
                                    base_text_len..(base_text_len + *delta_diameter),
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

#[inline]
fn simulate_inserts(
    handle: &CircuitHandle,
    inputs: &DbspInputs,
    output: &DbspOutput,
    replica: &mut ListReplica,
    trace_provider: &impl TraceProvider,
    range: Range<usize>,
) {
    let insert_op_input = inputs.get("insert").unwrap();
    let assign_op_input = inputs.get("assign").unwrap();
    let remove_op_input = inputs.get("remove").unwrap();

    let until = range.end;
    for list_op in trace_provider.list_ops_range(range) {
        let (insert_ops, assign_ops, remove_ops) = replica.generate_op(list_op);
        insert_op_input.insert_with_same_weight(insert_ops.iter(), 1);
        assign_op_input.insert_with_same_weight(assign_ops.iter(), 1);
        remove_op_input.insert_with_same_weight(remove_ops.iter(), 1);
        handle.step().unwrap();
        let result = output.to_batch();
        replica.apply_output_delta(result.as_data());
    }
    let result = replica.materialize_string();
    debug_assert_eq!(result, trace_provider.result_until(until));
}

fn generate_operation_history_list(
    handle: &CircuitHandle,
    inputs: &DbspInputs,
    output: &DbspOutput,
    replica: &mut ListReplica,
    trace_provider: &impl TraceProvider,
    until: usize,
) -> (Vec<InsertOp>, Vec<AssignOp>, Vec<RemoveOp>) {
    let insert_op_input = inputs.get("insert").unwrap();
    let assign_op_input = inputs.get("assign").unwrap();
    let remove_op_input = inputs.get("remove").unwrap();

    let mut bursts = Vec::new();
    for list_op in trace_provider.list_ops_range(0..until) {
        let (insert_ops, assign_ops, remove_ops) = replica.generate_op(list_op);
        insert_op_input.insert_with_same_weight(insert_ops.iter(), 1);
        assign_op_input.insert_with_same_weight(assign_ops.iter(), 1);
        remove_op_input.insert_with_same_weight(remove_ops.iter(), 1);
        bursts.push((insert_ops, assign_ops, remove_ops));
        handle.step().unwrap();
        let result = output.to_batch();
        replica.apply_output_delta(result.as_data());
    }
    let result = replica.materialize_string();
    debug_assert_eq!(result, trace_provider.result_until(until));

    // We collect all the operations that were generated during the simulation
    // for simulating the hydration setting.
    let mut insert_ops: Vec<InsertOp> = Vec::new();
    let mut assign_ops: Vec<AssignOp> = Vec::new();
    let mut remove_ops: Vec<RemoveOp> = Vec::new();
    bursts
        .iter()
        .for_each(|(insert_op_burst, assign_op_burst, remove_op_burst)| {
            insert_ops.extend(insert_op_burst);
            assign_ops.extend(assign_op_burst);
            remove_ops.extend(remove_op_burst);
        });
    (insert_ops, assign_ops, remove_ops)
}

fn prepare_circuit(
    threads: NonZeroUsize,
    code: &'static str,
) -> (CircuitHandle, DbspInputs, DbspOutput) {
    let (handle, inputs, output) = IncDataLog::new(threads, true)
        .build_circuit_from_parser(|root_circuit| Parser::new(root_circuit).parse(code))
        .unwrap();
    let assign_op_input = inputs.get("assign").unwrap();
    // Insert the sentinel dummy assignment.
    assign_op_input.insert_with_same_weight([&AssignOp::new(0, 0, 0, 0, '#')], 1);
    (handle, inputs, output)
}

criterion_group!(benches, bench_hydration_list, bench_near_real_time_list);
criterion_main!(benches);
