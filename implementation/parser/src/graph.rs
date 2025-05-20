//! Graph data structures and algorithms.

use std::collections::VecDeque;

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

pub type NodeId = usize;
pub type EdgeId = (usize, usize);

/// An implementation of a graph represented as an adjacency list.
/// `N` is the associated node data type and `E` is the associated edge data type.
#[derive(Clone, Debug, Default)]
pub struct ListGraph<N, E> {
    nodes: Vec<N>,
    edges: Vec<Vec<(NodeId, E)>>,
}

impl<N, E> ListGraph<N, E> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            edges: Vec::with_capacity(capacity),
        }
    }
    pub fn into_inner(self) -> (Vec<N>, Vec<Vec<(NodeId, E)>>) {
        (self.nodes, self.edges)
    }
    pub fn insert_node(&mut self, node: N) -> NodeId {
        self.nodes.push(node);
        self.edges.push(Vec::new());
        self.nodes.len() - 1
    }
    pub fn insert_edge(&mut self, from: NodeId, to: NodeId, edge: E) -> EdgeId {
        let adjacency_list = &mut self.edges[from];
        adjacency_list.push((to, edge));
        (from, adjacency_list.len() - 1)
    }
    pub fn node(&self, idx: NodeId) -> &N {
        &self.nodes[idx]
    }
    pub fn node_mut(&mut self, idx: NodeId) -> &mut N {
        &mut self.nodes[idx]
    }
    pub fn edge(&self, idx: EdgeId) -> (NodeId, NodeId, &E) {
        let from = idx.0;
        let adjacency_list = &self.edges[from];
        let (to, edge) = &adjacency_list[idx.1];
        (from, *to, edge)
    }
    pub fn edge_mut(&mut self, idx: EdgeId) -> (NodeId, NodeId, &mut E) {
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
