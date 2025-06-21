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
pub mod crdts;
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
    use crate::crdts::{mvr_crdt_store_datalog, mvr_store_datalog};

    use super::*;
    use compute::{
        dbsp::zset,
        relation::TupleValue,
        scalar::ScalarTypedValue,
        test_helper::{mvr_store_operation_history, setup_inc_data_log},
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

        let (pred_rel_data, set_op_data) = mvr_store_operation_history();

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
            // No change in the fourth step, as the operation is not yet causally ready.
            zset! {},
            zset! {
                tuple!(1_u64, 4_u64) => -1,
                tuple!(1_u64, 6_u64) => 1,
            },
        ]
        .into_iter();

        for (pred_rel_step, set_op_step) in pred_rel_data.into_iter().zip(set_op_data) {
            pred_rel_input.insert_with_same_weight(pred_rel_step.iter(), 1);
            set_op_input.insert_with_same_weight(set_op_step.iter(), 1);

            handle.step()?;

            let batch = output.to_batch();
            println!("{}", batch.as_table());
            assert_eq!(batch.as_zset(), expected.next().unwrap());
        }
        Ok(())
    }

    #[test]
    fn test_mvr_store_end_to_end() -> Result<(), anyhow::Error> {
        let inc_data_log = setup_inc_data_log();

        let (mut handle, inputs, output) =
            inc_data_log.build_circuit_from_parser(|root_circuit| {
                Parser::new(root_circuit).parse(mvr_store_datalog())
            })?;

        let pred_rel_input = inputs.get("pred").unwrap();
        let set_op_input = inputs.get("set").unwrap();

        let (pred_rel_data, set_op_data) = mvr_store_operation_history();

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
            // This outcome is derived too early, as the operation causing it
            // is actually not yet causally ready.
            zset! {
                tuple!(1_u64, 6_u64) => 1,
            },
            zset! {
                tuple!(1_u64, 4_u64) => -1,
            },
        ]
        .into_iter();

        for (pred_rel_step, set_op_step) in pred_rel_data.into_iter().zip(set_op_data) {
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
