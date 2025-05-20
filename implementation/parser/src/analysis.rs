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

use crate::ast::{Atom, Body, Head, Program, Rule};
use compute::error::SyntaxError;
use std::collections::{HashMap, VecDeque};

pub trait DirectedGraph<'a, N, E>
where
    N: 'a,
    E: 'a,
{
    fn nodes(&'a self) -> impl Iterator<Item = (NodeId, &'a N)>;
    fn nodes_len(&'a self) -> usize {
        self.nodes().count()
    }
    fn edges(&'a self) -> impl Iterator<Item = (NodeId, NodeId, &'a E)>;
    fn edges_len(&'a self) -> usize {
        self.edges().count()
    }
    fn adjacent(&'a self, node: NodeId) -> impl Iterator<Item = (NodeId, &'a E)>;
    fn in_degrees(&'a self) -> Vec<usize> {
        let mut in_degrees = vec![0; self.nodes_len()];
        for (from, to, _) in self.edges() {
            in_degrees[to] += 1;
        }
        in_degrees
    }
    fn kahn_topo_sort(&'a self) -> Option<Vec<NodeId>> {
        let mut in_degrees = self.in_degrees();
        let mut queue = VecDeque::new();
        for (node, in_degree) in in_degrees.iter().enumerate() {
            if *in_degree == 0 {
                queue.push_back(node);
            }
        }
        let mut sorted = Vec::with_capacity(self.nodes_len());
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            sorted.push(node);
            for (adj_node, _) in self.adjacent(node) {
                in_degrees[adj_node] -= 1;
                if in_degrees[adj_node] == 0 {
                    queue.push_back(adj_node);
                }
            }
        }
        if sorted.len() == self.nodes_len() {
            Some(sorted)
        } else {
            println!("Graph has cycles: {:?}", sorted);
            println!("Indegrees: {:?}", in_degrees);
            None
        }
    }
}

impl<'a, N, E> DirectedGraph<'a, N, E> for ListGraph<N, E>
where
    N: 'a,
    E: 'a,
{
    fn nodes_len(&self) -> usize {
        self.nodes.len()
    }
    fn nodes(&'a self) -> impl Iterator<Item = (NodeId, &'a N)> {
        self.nodes()
    }
    fn edges_len(&self) -> usize {
        self.edges
            .iter()
            .map(|adjacency_list| adjacency_list.len())
            .sum()
    }
    fn edges(&'a self) -> impl Iterator<Item = (NodeId, NodeId, &'a E)> {
        self.edges()
    }
    fn adjacent(&'a self, node: NodeId) -> impl Iterator<Item = (NodeId, &'a E)> {
        self.edges[node].iter().map(|(to, edge)| (*to, edge))
    }
}

type NodeId = usize;
type EdgeId = (usize, usize);

/// An implementation of a graph represented as an adjacency list.
/// `N` is the associated node data type and `E` is the associated edge data type.
#[derive(Clone, Debug, Default)]
struct ListGraph<N, E> {
    nodes: Vec<N>,
    edges: Vec<Vec<(NodeId, E)>>,
}

impl<N, E> ListGraph<N, E> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            edges: Vec::with_capacity(capacity),
        }
    }
    fn insert_node(&mut self, node: N) -> NodeId {
        self.nodes.push(node);
        self.edges.push(Vec::new());
        self.nodes.len() - 1
    }
    fn insert_edge(&mut self, from: NodeId, to: NodeId, edge: E) -> EdgeId {
        let adjacency_list = &mut self.edges[from];
        adjacency_list.push((to, edge));
        (from, adjacency_list.len() - 1)
    }
    fn node(&self, idx: NodeId) -> &N {
        &self.nodes[idx]
    }
    fn node_mut(&mut self, idx: NodeId) -> &mut N {
        &mut self.nodes[idx]
    }
    fn edge(&self, idx: EdgeId) -> (NodeId, NodeId, &E) {
        let from = idx.0;
        let adjacency_list = &self.edges[from];
        let (to, edge) = &adjacency_list[idx.1];
        (from, *to, edge)
    }
    fn edge_mut(&mut self, idx: EdgeId) -> (NodeId, NodeId, &mut E) {
        let from = idx.0;
        let adjacency_list = &mut self.edges[from];
        let (to, edge) = &mut adjacency_list[idx.1];
        (from, *to, edge)
    }
    fn nodes(&self) -> impl Iterator<Item = (NodeId, &N)> {
        self.nodes.iter().enumerate()
    }
    fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, &E)> {
        self.edges
            .iter()
            .enumerate()
            .flat_map(|(from, adjacency_list)| {
                adjacency_list
                    .iter()
                    .map(move |(to, edge)| (from, *to, edge))
            })
    }
}

#[derive(Clone, Debug)]
struct PrecedenceGraph {
    inner: ListGraph<AggregatedRule, Precedence>,
}

impl PrecedenceGraph {
    fn from_ast(ast: Program) -> Result<Self, SyntaxError> {
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
            for atom in node.all_bodies() {
                let undefined_predicate_error = |name: &str| {
                    SyntaxError::new(format!("Predicate {} is not defined but referenced.", name))
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
    fn to_execution_order(self) -> Result<Vec<AggregatedRule>, SyntaxError> {
        let sorting = self.inner.kahn_topo_sort().ok_or_else(|| {
            SyntaxError::new("Cannot produce execution order, precedence graph has cycles.")
        })?;
        let mut nodes: Vec<Option<AggregatedRule>> = self
            .inner
            .nodes
            .into_iter()
            .map(|node| Some(node))
            .collect();
        Ok(sorting
            .into_iter()
            .map(|idx| nodes[idx].take().unwrap())
            .collect())
    }
}

/// An aggregated rule contains all rules that share the same head, that is,
/// have the same name and same list of variables.
#[derive(Clone, Debug)]
struct AggregatedRule {
    head: Head,
    bodies: Vec<Body>,
}

impl AggregatedRule {
    fn new(rule: Rule) -> Self {
        if rule.is_extensional() {
            Self {
                head: rule.head,
                bodies: Vec::default(),
            }
        } else {
            Self {
                head: rule.head,
                bodies: vec![rule.body],
            }
        }
    }
    pub fn name(&self) -> &str {
        self.head.name()
    }
    fn is_extensional(&self) -> bool {
        self.bodies.is_empty()
    }
    fn is_intensional(&self) -> bool {
        !self.is_extensional()
    }
    fn try_insert(&mut self, rule: Rule) -> Result<(), (Rule, SyntaxError)> {
        // We only allow intensional rules to be aggregated.
        if self.is_extensional() {
            return Err((
                rule,
                SyntaxError::new(format!(
                    "Rule {} is extensional but is defined multiple times.",
                    self.head.name()
                )),
            ));
        }
        // We only allow rules with the same head to be aggregated.
        if self.head != rule.head {
            return Err((
                rule,
                SyntaxError::new(format!(
                    "Rule {} is defined multiple times with different heads.",
                    self.head.name()
                )),
            ));
        }
        self.bodies.push(rule.body);
        return Ok(());
    }
    fn bodies(&self) -> impl Iterator<Item = &Body> {
        self.bodies.iter()
    }
    fn all_bodies(&self) -> impl Iterator<Item = &Atom> {
        self.bodies().flat_map(|body| body.atoms.iter())
    }
    fn is_self_recursive(&self) -> bool {
        self.all_bodies().any(|atom| match atom {
            Atom::Positive(predicate) => predicate.name == self.head.name,
            Atom::Negative(predicate) => predicate.name == self.head.name,
            Atom::Comparison(_) => false,
        })
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
        let order = graph.to_execution_order()?;
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
