// Modified derivative of RivGraph's directionality.py.
// Copyright 2019 Triad National Security, LLC. All rights reserved.
// Produced under U.S. Government contract 89233218CNA000001 for LANL.
// See ../LICENSE.txt for the complete LANL notice and BSD-3 terms.
use std::collections::{HashMap, HashSet};

use petgraph::{
    algo::is_cyclic_directed,
    stable_graph::{EdgeIndex, NodeIndex, StableDiGraph},
    Direction::{self, Incoming, Outgoing},
};
use pyo3::{exceptions::PyValueError, prelude::*};

struct CycleSearch {
    graph: StableDiGraph<(), ()>,
    edges: Vec<EdgeIndex>,
    boundary: HashSet<NodeIndex>,
    inlets: Vec<NodeIndex>,
    outlets: Vec<NodeIndex>,
}

impl CycleSearch {
    fn flip(&mut self, index: usize) {
        // Every live link has its own edge handle, including parallel links.
        let edge = self.edges[index];
        let (u, v) = self.graph.edge_endpoints(edge).expect("live edge handle");
        self.graph.remove_edge(edge);
        self.edges[index] = self.graph.add_edge(v, u, ());
    }

    fn reachable(&self, starts: &[NodeIndex], direction: Direction) -> HashSet<NodeIndex> {
        let mut seen: HashSet<_> = starts.iter().copied().collect();
        let mut stack = starts.to_vec();
        while let Some(node) = stack.pop() {
            for next in self.graph.neighbors_directed(node, direction) {
                if seen.insert(next) {
                    stack.push(next);
                }
            }
        }
        seen
    }

    fn valid(&self) -> bool {
        if self.graph.node_indices().any(|n| {
            !self.boundary.contains(&n)
                && (self.graph.neighbors_directed(n, Incoming).next().is_none()
                    || self.graph.neighbors_directed(n, Outgoing).next().is_none())
        }) || is_cyclic_directed(&self.graph)
        {
            return false;
        }
        let from_inlets = self.reachable(&self.inlets, Outgoing);
        let to_outlets = self.reachable(&self.outlets, Incoming);
        self.outlets.iter().all(|n| from_inlets.contains(n))
            && self.inlets.iter().all(|n| to_outlets.contains(n))
    }

    fn combinations(
        &mut self,
        candidates: &[usize],
        start: usize,
        remaining: usize,
        selected: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        if remaining == 0 {
            return self.valid().then(|| selected.clone());
        }
        for i in start..=candidates.len() - remaining {
            let edge = candidates[i];
            self.flip(edge);
            selected.push(edge);
            let result = self.combinations(candidates, i + 1, remaining - 1, selected);
            selected.pop();
            self.flip(edge);
            if result.is_some() {
                return result;
            }
        }
        None
    }

    fn solve(&mut self, candidates: &[usize]) -> Option<Vec<usize>> {
        for count in 1..=candidates.len() {
            if let Some(selected) = self.combinations(candidates, 0, count, &mut Vec::new()) {
                return Some(selected);
            }
        }
        None
    }
}

fn map_terminals(nodes: &HashMap<i64, NodeIndex>, ids: &[i64]) -> PyResult<Vec<NodeIndex>> {
    ids.iter()
        .map(|id| {
            nodes
                .get(id)
                .copied()
                .ok_or_else(|| PyValueError::new_err("terminal or boundary node absent from edges"))
        })
        .collect()
}

/// One boundary crossing for the entire bounded candidate search, not per flip.
#[pyfunction]
pub fn first_valid_cycle_flip(
    py: Python<'_>,
    edges: Vec<(i64, i64)>,
    candidates: Vec<usize>,
    boundary: Vec<i64>,
    inlets: Vec<i64>,
    outlets: Vec<i64>,
) -> PyResult<Option<Vec<usize>>> {
    if candidates.len() > 10 {
        return Err(PyValueError::new_err(
            "at most ten candidate links are allowed",
        ));
    }
    let mut unique = HashSet::new();
    if candidates
        .iter()
        .any(|&i| i >= edges.len() || !unique.insert(i))
    {
        return Err(PyValueError::new_err(
            "candidate indices must be distinct and in range",
        ));
    }
    let mut graph = StableDiGraph::new();
    let mut nodes = HashMap::new();
    let mut handles = Vec::with_capacity(edges.len());
    for (u, v) in edges {
        let u = *nodes.entry(u).or_insert_with(|| graph.add_node(()));
        let v = *nodes.entry(v).or_insert_with(|| graph.add_node(()));
        handles.push(graph.add_edge(u, v, ()));
    }
    let mut search = CycleSearch {
        graph,
        edges: handles,
        boundary: map_terminals(&nodes, &boundary)?.into_iter().collect(),
        inlets: map_terminals(&nodes, &inlets)?,
        outlets: map_terminals(&nodes, &outlets)?,
    };
    Ok(py.detach(move || search.solve(&candidates)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle_with_terminals() -> CycleSearch {
        let mut graph = StableDiGraph::new();
        let nodes: Vec<_> = (0..4).map(|_| graph.add_node(())).collect();
        let edges = [(0, 1), (1, 2), (2, 1), (2, 3)]
            .iter()
            .map(|&(u, v)| graph.add_edge(nodes[u], nodes[v], ()))
            .collect();
        CycleSearch {
            graph,
            edges,
            boundary: [nodes[0], nodes[3]].into_iter().collect(),
            inlets: vec![nodes[0]],
            outlets: vec![nodes[3]],
        }
    }

    #[test]
    fn minimum_repair_restores_working_graph() {
        let mut search = cycle_with_terminals();
        let before: Vec<_> = search
            .edges
            .iter()
            .map(|&e| search.graph.edge_endpoints(e))
            .collect();
        assert_eq!(search.solve(&[1, 2]), Some(vec![2]));
        let after: Vec<_> = search
            .edges
            .iter()
            .map(|&e| search.graph.edge_endpoints(e))
            .collect();
        assert_eq!(before, after);
        assert_eq!(search.solve(&[1, 2]), Some(vec![2]));
    }

    #[test]
    fn no_candidate_and_unrepairable_network() {
        let mut search = cycle_with_terminals();
        assert_eq!(search.solve(&[]), None);
        search.boundary.clear();
        search.inlets.clear();
        search.outlets.clear();
        assert_eq!(search.solve(&[0, 1, 2, 3]), None);
    }
}
