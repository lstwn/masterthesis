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
pub mod datalog;
pub mod expr;
mod graph;
pub mod key_value_store_crdts;
pub mod list_crdt;
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
    use crate::{
        key_value_store_crdts::{MVR_KV_STORE_CRDT_DATALOG, MVR_KV_STORE_DATALOG},
        list_crdt::{AssignOp, InsertOp, LIST_CRDT_DATALOG, RemoveOp},
    };
    use compute::{
        dbsp::zset,
        relation::TupleValue,
        scalar::ScalarTypedValue,
        test_helper::{PlainRelation, mvr_store_operation_history, setup_inc_data_log},
        tuple,
    };

    #[test]
    fn test_mvr_crdt_store_end_to_end() -> Result<(), anyhow::Error> {
        let inc_data_log = setup_inc_data_log();

        let (handle, inputs, output) = inc_data_log.build_circuit_from_parser(|root_circuit| {
            Parser::new(root_circuit).parse(MVR_KV_STORE_CRDT_DATALOG)
        })?;

        let pred_op_input = inputs.get("pred").unwrap();
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

        for (pred_op_step, set_op_step) in mvr_store_operation_history() {
            pred_op_input.insert_with_same_weight(&pred_op_step, 1);
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

        let (handle, inputs, output) = inc_data_log.build_circuit_from_parser(|root_circuit| {
            Parser::new(root_circuit).parse(MVR_KV_STORE_DATALOG)
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
    fn test_list_crdt_martin() -> Result<(), anyhow::Error> {
        let inc_data_log = setup_inc_data_log();

        let (handle, inputs, output) = inc_data_log.build_circuit_from_parser(|root_circuit| {
            Parser::new(root_circuit).parse(LIST_CRDT_DATALOG)
        })?;

        let insert_op_input = inputs.get("insert").unwrap();
        let assign_op_input = inputs.get("assign").unwrap();
        let remove_op_input = inputs.get("remove").unwrap();

        // Example tree, encoded as `insert(Child, Parent)` facts.
        // The example only shows counters but no replica ids because
        // all updates stem from the same replica.
        //
        // ```text
        //        0 <- sentinel element
        //      /   \
        //     2     1
        //   / | \   |
        //  6  5  3  4
        // ```
        let data: [(Vec<InsertOp>, Vec<AssignOp>, Vec<RemoveOp>); 1] = [(
            vec![
                InsertOp::new(0, 1, 0, 0),
                InsertOp::new(0, 2, 0, 0),
                InsertOp::new(0, 3, 0, 2),
                InsertOp::new(0, 4, 0, 1),
                InsertOp::new(0, 5, 0, 2),
                InsertOp::new(0, 6, 0, 2),
            ],
            vec![
                // Oddly, another replica assigns the values to the list elements.
                // I think it should be alternating between the insert ops and assign
                // ops but this would break Martin's example.
                AssignOp::new(0, 0, 0, 0, '#'), // dummy element
                AssignOp::new(1, 2, 0, 2, 'H'),
                AssignOp::new(1, 3, 0, 6, 'E'),
                AssignOp::new(1, 4, 0, 5, 'L'),
                AssignOp::new(1, 5, 0, 3, 'L'),
                AssignOp::new(1, 6, 0, 1, 'O'),
                AssignOp::new(1, 7, 0, 4, '!'),
            ],
            // No removals here.
            vec![],
        )];

        let mut expected = [zset! {
            // Schema: PrevRepId, PrevCtr, Char (Value), AssignRepId, AssignCtr, NextRepId, NextCtr.
            tuple!(0_u64, 0_u64, 'H', 1_u64, 2_u64, 0_u64, 2_u64) => 1,
            tuple!(0_u64, 1_u64, '!', 1_u64, 7_u64, 0_u64, 4_u64) => 1,
            tuple!(0_u64, 2_u64, 'E', 1_u64, 3_u64, 0_u64, 6_u64) => 1,
            tuple!(0_u64, 3_u64, 'O', 1_u64, 6_u64, 0_u64, 1_u64) => 1,
            tuple!(0_u64, 5_u64, 'L', 1_u64, 5_u64, 0_u64, 3_u64) => 1,
            tuple!(0_u64, 6_u64, 'L', 1_u64, 4_u64, 0_u64, 5_u64) => 1,
        }]
        .into_iter();

        for (insert_op_step, assign_op_step, remove_op_step) in data {
            insert_op_input.insert_with_same_weight(&insert_op_step, 1);
            assign_op_input.insert_with_same_weight(&assign_op_step, 1);
            remove_op_input.insert_with_same_weight(&remove_op_step, 1);
            handle.step()?;
            let batch = output.to_batch();
            println!("{}", batch.as_table());
            assert_eq!(batch.as_zset(), expected.next().unwrap());
        }
        Ok(())
    }

    #[test]
    fn test_list_crdt_multi_replicas() -> Result<(), anyhow::Error> {
        let inc_data_log = setup_inc_data_log();

        let (handle, inputs, output) = inc_data_log.build_circuit_from_parser(|root_circuit| {
            Parser::new(root_circuit).parse(LIST_CRDT_DATALOG)
        })?;

        let insert_op_input = inputs.get("insert").unwrap();
        let assign_op_input = inputs.get("assign").unwrap();
        let remove_op_input = inputs.get("remove").unwrap();

        // Example tree, encoded as `insert(ChildRepId, ChildCtr, ParentRepId, ParentCtr)`
        // facts. Below, a node depicts `(RepId, Ctr)`:
        //
        // ```text
        //             (0,0) <- sentinel element
        //         /           \
        //       (2,1)        (1,1)
        //     /   |   \        |
        // (2,3) (1,3) (3,2)  (2,5)
        // ```
        let data: [(Vec<InsertOp>, Vec<AssignOp>, Vec<RemoveOp>); 1] = [(
            vec![
                InsertOp::new(2, 1, 0, 0),
                InsertOp::new(1, 1, 0, 0),
                InsertOp::new(2, 5, 1, 1),
                InsertOp::new(2, 3, 2, 1),
                InsertOp::new(1, 3, 2, 1),
                InsertOp::new(3, 2, 2, 1),
            ],
            vec![
                AssignOp::new(0, 0, 0, 0, '#'), // dummy element
                AssignOp::new(2, 2, 2, 1, 'H'),
                AssignOp::new(2, 4, 2, 3, 'E'),
                AssignOp::new(1, 4, 1, 3, 'L'),
                AssignOp::new(3, 3, 3, 2, 'L'),
                AssignOp::new(1, 2, 1, 1, 'O'),
                AssignOp::new(2, 6, 2, 5, '!'),
            ],
            // We can remove the 'E' in the *middle* of the list! :)
            vec![RemoveOp::new(2, 4)],
        )];

        let mut expected = [zset! {
            // Schema: PrevRepId, PrevCtr, Char (Value), AssignRepId, AssignCtr, NextRepId, NextCtr.
            tuple!(0_u64, 0_u64, 'H', 2_u64, 2_u64, 2_u64, 1_u64) => 1,
            tuple!(2_u64, 1_u64, 'L', 1_u64, 4_u64, 1_u64, 3_u64) => 1,
            tuple!(1_u64, 3_u64, 'L', 3_u64, 3_u64, 3_u64, 2_u64) => 1,
            tuple!(3_u64, 2_u64, 'O', 1_u64, 2_u64, 1_u64, 1_u64) => 1,
            tuple!(1_u64, 1_u64, '!', 2_u64, 6_u64, 2_u64, 5_u64) => 1,
        }]
        .into_iter();

        for (insert_op_step, assign_op_step, remove_op_step) in data {
            insert_op_input.insert_with_same_weight(&insert_op_step, 1);
            assign_op_input.insert_with_same_weight(&assign_op_step, 1);
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

        let (handle, inputs, output) = inc_data_log.build_circuit_from_parser(|root_circuit| {
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
