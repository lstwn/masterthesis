//! This module performs static analysis on an [`AST`](crate::ast)
//! (1) to ensure that a parsed program is a _valid_ Datalog program, and
//! (2) to perform initial steps towards figuring out its query plan.
//!
//! To us, this encompasses:
//! 1. For rules with the same head, the bodies are grouped together.
//! 1. Finding a valid evaluation order via the precedence graph.
//! 1. Checking that there are no cycles to avoid the problems of:
//!     1. Stratified negation.
//!     1. Not having to deal with mutual recursion.
//! 1. Finding the set of all variables in the program and on which relations
//!    they depend.

// Build a precedence graph to:
// 1. Find all cycles.
// 2. Serialize the graph to a topological order.

use crate::{
    ast::{Atom, Body, Head, Program, Rule},
    graph::{DirectedGraph, ListGraph, NodeId},
};
use compute::error::SyntaxError;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct PrecedenceGraph {
    inner: ListGraph<AggregatedRule, Precedence>,
}

impl PrecedenceGraph {
    pub fn from_ast(ast: Program) -> Result<Self, SyntaxError> {
        let mut inner = ListGraph::<AggregatedRule, Precedence>::with_capacity(ast.rules.len());
        let mut aggregated_nodes = HashMap::<String, NodeId>::with_capacity(ast.rules.len());

        for rule in ast.rules {
            match aggregated_nodes.get_mut(rule.name()) {
                Some(idx) => {
                    inner
                        .node_mut(*idx)
                        .try_insert(rule)
                        .map_err(|(_rule, error)| error)?;
                }
                None => {
                    let name = rule.name().clone();
                    let idx = inner.insert_node(AggregatedRule::new(rule));
                    aggregated_nodes.insert(name, idx);
                }
            };
        }

        let mut edges = Vec::with_capacity(aggregated_nodes.len());

        for (to, node) in inner.nodes() {
            for atom in node.atoms() {
                let undefined_predicate_error = |name: &str| {
                    SyntaxError::new(format!(
                        "Predicate '{}' is not defined but referenced within rule '{}'.",
                        name,
                        node.name()
                    ))
                };
                let edge = match atom {
                    Atom::Positive(predicate) => {
                        let from = *aggregated_nodes
                            .get(predicate.name())
                            .ok_or_else(|| undefined_predicate_error(predicate.name()))?;
                        Some((from, to, Precedence::new(PrecedenceKind::Positive)))
                    }
                    Atom::Negative(predicate) => {
                        let from = *aggregated_nodes
                            .get(predicate.name())
                            .ok_or_else(|| undefined_predicate_error(predicate.name()))?;
                        Some((from, to, Precedence::new(PrecedenceKind::Negative)))
                    }
                    Atom::Comparison(_) => None,
                };
                if let Some((from, to, edge)) = edge {
                    // We tolerate self-loops, but we don't want to add them to
                    // the graph to avoid a false cycle detection.
                    if from != to {
                        edges.push((from, to, edge));
                    }
                }
            }
        }

        for (from, to, dependency) in edges {
            inner.insert_edge(from, to, dependency);
        }

        Ok(Self { inner })
    }
    pub fn into_execution_order(self) -> Result<Vec<AggregatedRule>, SyntaxError> {
        let sorting = self.inner.kahn_topo_sort().ok_or_else(|| {
            SyntaxError::new("Cannot produce execution order as precedence graph contains cycles.")
        })?;
        let (nodes, _edges) = self.inner.into_inner();
        // Wrapping in Options here is required to allow moving nodes out of
        // the `nodes` vector in the order given by the `sorting` vector.
        // Rust does not offer move semantics for vectors' elements.
        let mut nodes: Vec<Option<AggregatedRule>> = nodes.into_iter().map(Some).collect();
        Ok(sorting
            .into_iter()
            .map(|idx| nodes[idx].take().unwrap())
            .collect())
    }
}

pub type DistinctFlaggedBody = (bool, Body);

/// An aggregated rule contains all rules that share the same head, that is,
/// have the same name and same sequence of variables.
#[derive(Clone, Debug)]
pub struct AggregatedRule {
    head: Head,
    non_recursive_bodies: Vec<DistinctFlaggedBody>,
    recursive_bodies: Vec<DistinctFlaggedBody>,
}

impl AggregatedRule {
    fn new(rule: Rule) -> Self {
        if rule.is_extensional() {
            Self {
                head: rule.head,
                non_recursive_bodies: Vec::with_capacity(0), // Does not allocate.
                recursive_bodies: Vec::with_capacity(0),     // Does not allocate.
            }
        } else {
            let distinct = rule.head.distinct;
            let mut aggregated_rule = Self {
                head: rule.head,
                non_recursive_bodies: Vec::with_capacity(4),
                recursive_bodies: Vec::with_capacity(2),
            };
            aggregated_rule.insert_body(distinct, rule.body);
            aggregated_rule
        }
    }
    pub fn name(&self) -> &String {
        self.head.name()
    }
    pub fn head(&self) -> &Head {
        &self.head
    }
    pub fn is_extensional(&self) -> bool {
        self.recursive_bodies.is_empty() && self.non_recursive_bodies.is_empty()
    }
    pub fn is_intensional(&self) -> bool {
        !self.is_extensional()
    }
    pub fn is_self_recursive(&self) -> bool {
        !self.recursive_bodies.is_empty()
    }
    /// Iterates over all bodies of this aggregated rule, starting with the
    /// non-recursive bodies, followed by the recursive bodies.
    pub fn bodies(&self) -> impl Iterator<Item = &Body> {
        self.non_recursive_bodies
            .iter()
            .map(|(_distinct, body)| body)
            .chain(self.recursive_bodies.iter().map(|(_distinct, body)| body))
    }
    pub fn atoms(&self) -> impl Iterator<Item = &Atom> {
        self.bodies().flat_map(|body| body.atoms.iter())
    }
    pub fn non_recursive_atoms(&self) -> impl Iterator<Item = &Atom> {
        self.non_recursive_bodies
            .iter()
            .flat_map(|body| body.1.atoms.iter())
    }
    pub fn recursive_atoms(&self) -> impl Iterator<Item = &Atom> {
        self.recursive_bodies
            .iter()
            .flat_map(|body| body.1.atoms.iter())
    }
    /// Returns the head and all bodies of this aggregated rule. It provides
    /// the invariant that the bodies are sorted such that the non-recursive
    /// bodies come first, followed by the recursive bodies.
    pub fn into_head_and_bodies(
        mut self,
    ) -> (
        Head,
        impl DoubleEndedIterator<Item = DistinctFlaggedBody>
            + ExactSizeIterator<Item = DistinctFlaggedBody>,
    ) {
        // std::iter::chain destroys the double-ended and exact-size property..
        self.non_recursive_bodies.extend(self.recursive_bodies);
        (self.head, self.non_recursive_bodies.into_iter())
    }
    pub fn into_head_and_non_rec_rec_bodies(
        self,
    ) -> (Head, Vec<DistinctFlaggedBody>, Vec<DistinctFlaggedBody>) {
        (self.head, self.non_recursive_bodies, self.recursive_bodies)
    }
    fn try_insert(&mut self, candidate: Rule) -> Result<(), (Rule, SyntaxError)> {
        // We only allow intensional rules to be aggregated.
        if self.is_extensional() {
            return Err((
                candidate,
                SyntaxError::new(format!(
                    "Rule '{}' is extensional but defined multiple times.",
                    self.head.name()
                )),
            ));
        }
        // We only allow rules with a compatible head to be aggregated.
        if self.head.aggregatable_with(&candidate.head) {
            return Err((
                candidate,
                SyntaxError::new(format!(
                    "Rule '{}' is defined multiple times with incompatible heads.",
                    self.head.name()
                )),
            ));
        }
        self.insert_body(candidate.head.distinct, candidate.body);
        Ok(())
    }
    fn insert_body(&mut self, distinct: bool, body: Body) {
        if body.atoms.iter().any(|atom| match atom {
            Atom::Positive(predicate) => predicate.name() == self.name(),
            Atom::Negative(predicate) => predicate.name() == self.name(),
            Atom::Comparison(_) => false,
        }) {
            self.recursive_bodies.push((distinct, body));
        } else {
            self.non_recursive_bodies.push((distinct, body));
        }
    }
}

impl PartialEq<Rule> for AggregatedRule {
    fn eq(&self, other: &Rule) -> bool {
        self.head == other.head
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Precedence {
    kind: PrecedenceKind,
}

impl Precedence {
    fn new(kind: PrecedenceKind) -> Self {
        Self { kind }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrecedenceKind {
    Positive,
    Negative,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::datalog::test::mvr_store_crdt_ast;

    #[test]
    fn test_build_dependency_graph() -> Result<(), SyntaxError> {
        let ast = mvr_store_crdt_ast();
        let graph = PrecedenceGraph::from_ast(ast)?;
        println!("{:#?}", graph);
        let order = graph.into_execution_order()?;
        let expected = [
            "pred",
            "set",
            "overwritten",
            "overwrites",
            "isLeaf",
            "isRoot",
            "isCausallyReady",
            "mvrStore",
        ]
        .into_iter();
        assert!(order.iter().map(|rule| rule.name()).eq(expected));
        println!("{:#?}", order);
        Ok(())
    }
}
