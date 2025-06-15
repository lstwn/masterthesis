use compute::{
    dbsp::{DBSPHandle, DbspError, DbspInputs, DbspOutput, RootCircuit, Runtime},
    error::{RuntimeError, SyntaxError},
    stmt::Code,
    variable::Value,
    IncLog,
};
use optimizer::Optimizer;
use parser::Parser;
use std::num::NonZeroUsize;

fn main() {
    println!("Hello, world!");
}

#[derive(Clone)]
pub struct IncDataLog {
    threads: NonZeroUsize,
    optimize: bool,
}

impl Default for IncDataLog {
    fn default() -> Self {
        Self {
            threads: NonZeroUsize::new(1).unwrap(),
            optimize: true,
        }
    }
}

impl IncDataLog {
    pub fn build_circuit_from_ir<F>(
        &self,
        intermediate_representation: F,
    ) -> Result<(DBSPHandle, DbspInputs, DbspOutput), anyhow::Error>
    where
        F: Fn(&mut RootCircuit, &mut DbspInputs) -> Result<Code, SyntaxError>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let optimizer = self.init_optimizer();
        let (circuit, (inputs, output)) = self.init_dbsp_runtime(move |root_circuit| {
            let mut inputs = DbspInputs::default();
            let naive_program = intermediate_representation(root_circuit, &mut inputs)?;
            let optimized_program = if let Some(optimizer) = optimizer {
                optimizer.optimize(naive_program)?
            } else {
                naive_program
            };
            Self::build_circuit(inputs, optimized_program).map_err(anyhow::Error::from)
        })?;

        Ok((circuit, inputs, output))
    }
    pub fn build_circuit_from_datalog<T: AsRef<str> + Clone + Send + 'static>(
        &self,
        code: T,
    ) -> Result<(DBSPHandle, DbspInputs, DbspOutput), anyhow::Error> {
        let optimizer = self.init_optimizer();
        let (circuit, (inputs, output)) = self.init_dbsp_runtime(move |root_circuit| {
            let (inputs, naive_program) = Parser::new(root_circuit).parse(code.as_ref())?;
            let optimized_program = if let Some(optimizer) = optimizer {
                optimizer.optimize(naive_program)?
            } else {
                naive_program
            };
            Self::build_circuit(inputs, optimized_program).map_err(anyhow::Error::from)
        })?;

        Ok((circuit, inputs, output))
    }
    fn init_dbsp_runtime<F, T>(&self, constructor: F) -> Result<(DBSPHandle, T), DbspError>
    where
        F: FnOnce(&mut RootCircuit) -> Result<T, anyhow::Error> + Clone + Send + 'static,
        T: Send + 'static,
    {
        Runtime::init_circuit(usize::from(self.threads), constructor)
    }
    fn init_optimizer(&self) -> Option<Optimizer> {
        if self.optimize {
            Some(Optimizer::default())
        } else {
            None
        }
    }
    fn build_circuit(
        inputs: DbspInputs,
        program: Code,
    ) -> Result<(DbspInputs, DbspOutput), RuntimeError> {
        let output = IncLog::default().execute(program);

        let output = match output {
            Ok(Some(Value::Relation(relation))) => {
                let relation = relation.borrow();
                let output_handle = relation.inner.output();
                let output_schema = relation.schema.clone();
                DbspOutput::new(output_schema, output_handle)
            }
            result => {
                return Err(RuntimeError::new(format!(
                    "Expected a relation as program's output, got {:?}",
                    result
                )));
            }
        };

        Ok((inputs, output))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use compute::test_helper::{PredRel, SetOp};

    #[test]
    fn test_inc_data_log_end_to_end() -> Result<(), anyhow::Error> {
        let inc_data_log = IncDataLog::default();

        let input = r#"
            // These are extensional database predicates (EDBPs).
            pred(FromNodeId, FromCounter, ToNodeId, ToCounter)  :- .
            set(NodeId, Counter, Key, Value)                    :- .

            // These are intensional database predicates (IDBPs).
            distinct overwritten(NodeId, Counter)     :- pred(NodeId = FromNodeId, Counter = FromCounter, _ToNodeId, _ToCounter).
            distinct overwrites(NodeId, Counter)      :- pred(_FromNodeId, _FromCounter, NodeId = ToNodeId, Counter = ToCounter).

            isRoot(NodeId, Counter)                   :- set(NodeId, Counter, _Key, _Value),
                                                         not overwrites(NodeId, Counter).

            isLeaf(NodeId, Counter)                   :- set(NodeId, Counter, _Key, _Value),
                                                         not overwritten(NodeId, Counter).

            isCausallyReady(NodeId, Counter)          :- isRoot(NodeId, Counter).
            isCausallyReady(NodeId, Counter)          :- isCausallyReady(FromNodeId = NodeId, FromCounter = Counter),
                                                         pred(FromNodeId, FromCounter, NodeId = ToNodeId, Counter = ToCounter).

            mvrStore(Key, Value)                      :- set(NodeId, Counter, Key, Value),
                                                         isCausallyReady(NodeId, Counter),
                                                         isLeaf(NodeId, Counter).
        "#;

        let (mut handle, inputs, output) = inc_data_log.build_circuit_from_datalog(input)?;

        let pred_rel_input = inputs.get("pred").unwrap();
        let set_op_input = inputs.get("set").unwrap();

        // The operation history is as follows:
        // In first step (just one root operation setting register with key 1 to value 1):
        //
        // set_0_0(1, 1)
        //
        // In second step (concurrent writes by replica 0 and 1):
        //
        //               ---> set_0_1(1, 2)
        // set_0_0(1, 1)
        //               ---> set_1_0(1, 3)
        //
        // In third step (replica 1 does a "merge" operation overwriting the previous conflict):
        //
        //               ---> set_0_1(1, 2)
        // set_0_0(1, 1)                    ---> set_1_2(1, 4)
        //               ---> set_1_0(1, 3)
        //

        let pred_rel_data = [
            vec![],
            vec![PredRel::new(0, 0, 0, 1), PredRel::new(0, 0, 1, 0)],
            vec![PredRel::new(0, 1, 1, 2), PredRel::new(1, 0, 1, 2)],
        ];

        let set_op_data = [
            vec![SetOp::new(0, 0, 1, 1)],
            vec![SetOp::new(0, 1, 1, 2), SetOp::new(1, 0, 1, 3)],
            vec![SetOp::new(1, 2, 1, 4)],
        ];

        // let mut expected = [
        //     zset! {
        //         tuple!(1_u64, 1_u64) => 1,
        //     },
        //     zset! {
        //         tuple!(1_u64, 1_u64) => -1,
        //         tuple!(1_u64, 2_u64) => 1,
        //         tuple!(1_u64, 3_u64) => 1,
        //     },
        //     zset! {
        //         tuple!(1_u64, 2_u64) => -1,
        //         tuple!(1_u64, 3_u64) => -1,
        //         tuple!(1_u64, 4_u64) => 1,
        //     },
        // ]
        // .into_iter();

        for (pred_rel_step, set_op_step) in pred_rel_data.iter().zip(set_op_data.iter()) {
            pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
            set_op_input.insert_with_same_weight(set_op_step.iter(), 1);

            handle.step()?;

            let batch = output.to_batch();
            println!("{}", batch.as_table());
            // assert_eq!(batch.as_zset(), expected.next().unwrap());
        }
        Ok(())
    }
}
