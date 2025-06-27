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
            .map_err(|e| SyntaxError::new(format!("Failed to parse input: {e}")))?;
        let precedence_graph = PrecedenceGraph::from_ast(datalog_program)?;
        let execution_order = precedence_graph.into_execution_order()?;
        let inclog_program = Translator::new(self.root_circuit, execution_order).translate()?;
        Ok(inclog_program)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::crdts::{list_crdt_datalog, mvr_crdt_store_datalog, mvr_store_datalog};
    use compute::{
        dbsp::zset,
        relation::TupleValue,
        scalar::ScalarTypedValue,
        test_helper::{
            PlainRelation, list_crdt_operation_history, mvr_store_operation_history,
            setup_inc_data_log,
        },
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

        for (pred_rel_step, set_op_step) in mvr_store_operation_history() {
            pred_rel_input.insert_with_same_weight(&pred_rel_step, 1);
            set_op_input.insert_with_same_weight(&set_op_step, 1);

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

        for (pred_rel_step, set_op_step) in mvr_store_operation_history() {
            pred_rel_input.insert_with_same_weight(&pred_rel_step, 1);
            set_op_input.insert_with_same_weight(&set_op_step, 1);

            handle.step()?;

            let batch = output.to_batch();
            println!("{}", batch.as_table());
            assert_eq!(batch.as_zset(), expected.next().unwrap());
        }
        Ok(())
    }

    #[test]
    fn test_list_crdt() -> Result<(), anyhow::Error> {
        let inc_data_log = setup_inc_data_log();

        let (mut handle, inputs, output) =
            inc_data_log.build_circuit_from_parser(|root_circuit| {
                Parser::new(root_circuit).parse(list_crdt_datalog())
            })?;

        let insert_op_input = inputs.get("insert").unwrap();
        let assert_op_input = inputs.get("assign").unwrap();
        let remove_op_input = inputs.get("remove").unwrap();

        let mut expected = [zset! {
            tuple!(0_u64, 0_u64, 0_u64, 2_u64) => 1,
            tuple!(0_u64, 1_u64, 0_u64, 4_u64) => 1,
            tuple!(0_u64, 2_u64, 0_u64, 6_u64) => 1,
            tuple!(0_u64, 3_u64, 0_u64, 1_u64) => 1,
            tuple!(0_u64, 5_u64, 0_u64, 3_u64) => 1,
            tuple!(0_u64, 6_u64, 0_u64, 5_u64) => 1,
        }]
        .into_iter();

        for (insert_op_step, assert_op_step, remove_op_step) in list_crdt_operation_history() {
            insert_op_input.insert_with_same_weight(&insert_op_step, 1);
            assert_op_input.insert_with_same_weight(&assert_op_step, 1);
            remove_op_input.insert_with_same_weight(&remove_op_step, 1);

            handle.step()?;

            let batch = output.to_batch();
            println!("{}", batch.as_table());
            assert_eq!(batch.as_zset(), expected.next().unwrap());
        }
        Ok(())
    }

    #[test]
    fn test_multiple_negation() -> Result<(), anyhow::Error> {
        let inc_data_log = setup_inc_data_log();

        let (mut handle, inputs, output) =
            inc_data_log.build_circuit_from_parser(|root_circuit| {
                let program = r#"
                    r1(a, b, c)                     :- .
                    r2(a, b, c)                     :- .
                    r3(a, b, c)                     :- .
                    test_multiple_negation(a, b, c) :- r1(a, b, c),
                                                       not r2(b),
                                                       not r3(c).
                "#;
                Parser::new(root_circuit).parse(program)
            })?;

        let r1_input = inputs.get("r1").unwrap();
        let r2_input = inputs.get("r2").unwrap();
        let r3_input = inputs.get("r3").unwrap();

        let mut expected = [zset! {
            tuple!(7_u64, 8_u64, 9_u64) => 1,
        }]
        .into_iter();

        let data_steps = PlainRelation::test_data_1()
            .into_iter()
            .zip(PlainRelation::test_data_2())
            .zip(PlainRelation::test_data_3());

        for ((r1_step, r2_step), r3_step) in data_steps {
            r1_input.insert_with_same_weight(&r1_step, 1);
            r2_input.insert_with_same_weight(&r2_step, 1);
            r3_input.insert_with_same_weight(&r3_step, 1);

            handle.step()?;

            let batch = output.to_batch();
            println!("{}", batch.as_table());
            assert_eq!(batch.as_zset(), expected.next().unwrap());
        }
        Ok(())
    }

    /// The safety condition demands that all variables occuring negatively
    /// in a rule must also occur at least once positively in the same rule.
    #[test]
    fn test_safety_condition() {
        let inc_data_log = setup_inc_data_log();

        let result = inc_data_log.build_circuit_from_parser(|root_circuit| {
            let invalid_program = r#"
                    r1(a, b, c) :- .
                    // Variable `b` is not occuring in a positive atom.
                    r2(a, b)    :- r1(a), not r1(b).
                "#;
            Parser::new(root_circuit).parse(invalid_program)
        });

        if let Err(ref e) = result {
            println!("Error: {e}");
        }
        assert!(result.is_err());
    }

    #[test]
    fn test_range_restriction() {
        let inc_data_log = setup_inc_data_log();

        let result = inc_data_log.build_circuit_from_parser(|root_circuit| {
            let invalid_program = r#"
                    r1(a, b, c) :- .
                    // Variable `b` is not bound.
                    r2(a, b)    :- r1(a).
                "#;
            Parser::new(root_circuit).parse(invalid_program)
        });

        if let Err(ref e) = result {
            println!("Error: {e}");
        }
        assert!(result.is_err());
    }
}
