#![allow(dead_code, unused_variables)]

use analysis::PrecedenceGraph;
use compute::{
    dbsp::{DbspInputs, RootCircuit},
    error::SyntaxError,
    stmt::Code,
};
use datalog::program;
use translation::Translator;

pub mod analysis;
pub mod ast;
mod crdts;
pub mod datalog;
pub mod expr;
mod graph;
pub mod literal;
pub mod parser_helper;
mod translation;
mod type_resolver;

pub struct Parser<'a> {
    root_circuit: &'a mut RootCircuit,
}

impl<'a> Parser<'a> {
    pub fn new(root_circuit: &'a mut RootCircuit) -> Self {
        Self { root_circuit }
    }
    pub fn parse(&mut self, input: &str) -> Result<(DbspInputs, Code), SyntaxError> {
        let datalog_program = program(input)
            .map(|(_, program)| program)
            .map_err(|e| SyntaxError::new(format!("Failed to parse input: {}", e)))?;
        let precedence_graph = PrecedenceGraph::from_ast(datalog_program)?;
        let execution_order = precedence_graph.into_execution_order()?;
        let inclog_program = Translator::new(self.root_circuit, execution_order).translate()?;
        Ok(inclog_program)
    }
}

#[cfg(test)]
mod test {
    use crate::crdts::mvr_crdt_store_datalog;

    use super::*;
    use compute::{
        dbsp::zset,
        relation::TupleValue,
        scalar::ScalarTypedValue,
        test_helper::{setup_inc_data_log, PredRel, SetOp},
        tuple,
    };

    #[test]
    fn test_mvr_crdt_store_end_to_end() -> Result<(), anyhow::Error> {
        let inc_data_log = setup_inc_data_log();

        let (mut handle, inputs, output) =
            inc_data_log.build_circuit_from_parser(|root_circuit| {
                Parser::new(root_circuit).parse(mvr_crdt_store_datalog())
            })?;

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

        let mut expected = [
            zset! {
                tuple!(1_u64, 1_u64) => 1,
            },
            zset! {
                tuple!(1_u64, 1_u64) => -1,
                tuple!(1_u64, 2_u64) => 1,
                tuple!(1_u64, 3_u64) => 1,
            },
            zset! {
                tuple!(1_u64, 2_u64) => -1,
                tuple!(1_u64, 3_u64) => -1,
                tuple!(1_u64, 4_u64) => 1,
            },
        ]
        .into_iter();

        for (pred_rel_step, set_op_step) in pred_rel_data.iter().zip(set_op_data.iter()) {
            pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
            set_op_input.insert_with_same_weight(set_op_step.iter(), 1);

            handle.step()?;

            let batch = output.to_batch();
            println!("{}", batch.as_table());
            assert_eq!(batch.as_zset(), expected.next().unwrap());
        }
        Ok(())
    }
}
