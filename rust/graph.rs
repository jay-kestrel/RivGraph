// Modified RivGraph mask_to_graph.py/walk.py/ln_utils.py.
// Copyright 2019 Triad National Security, LLC. All rights reserved.
// LANL contract 89233218CNA000001. Full notice and BSD-3 terms: ../LICENSE.txt.
use crate::{
    branchpoints::{walkable, Walker},
    sets::{self, Set},
};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PySet, PySetMethods, PyTuple},
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Default)]
struct Link {
    pixels: Vec<usize>,
    conn: Vec<usize>,
}
#[derive(Default)]
struct Graph {
    links: BTreeMap<usize, Link>,
    pixels: Vec<usize>,
    conn: Vec<Vec<usize>>,
    node_ids: HashMap<usize, usize>,
    todo: BTreeSet<usize>,
}
impl Graph {
    fn node(&mut self, p: usize, lid: usize) -> usize {
        let nid = *self.node_ids.entry(p).or_insert_with(|| {
            let n = self.pixels.len();
            self.pixels.push(p);
            self.conn.push(Vec::new());
            n
        });
        self.conn[nid].push(lid);
        nid
    }
    fn issue(&mut self, pixels: Vec<usize>, complete: bool) {
        let lid = self.links.last_key_value().map_or(0, |(&id, _)| id + 1);
        let mut conn = vec![self.node(pixels[0], lid)];
        if complete {
            conn.push(self.node(pixels[1], lid));
        } else {
            self.todo.insert(lid);
        }
        self.links.insert(lid, Link { pixels, conn });
    }
    fn delete(&mut self, lid: usize) -> PyResult<()> {
        let link = self
            .links
            .remove(&lid)
            .ok_or_else(|| PyRuntimeError::new_err("Missing link to delete"))?;
        for n in link.conn {
            let pos = self.conn[n]
                .iter()
                .position(|&id| id == lid)
                .ok_or_else(|| PyRuntimeError::new_err("Missing node/link connectivity"))?;
            self.conn[n].remove(pos);
        }
        Ok(())
    }
    fn cant_walk<'py>(
        &self,
        py: Python<'py>,
        walker: &mut Walker,
        lid: usize,
    ) -> PyResult<Set<'py>> {
        let start = self.links[&lid].pixels[0];
        let mut bps = vec![start];
        walker.cluster(py, &mut bps)?;
        let mut walked = sets::new(py, bps.iter().copied())?;
        for b in bps {
            walked = sets::union(&walked, &walkable(py, &walker.image, &[b])?)?;
        }
        for cl in &self.conn[self.node_ids[&start]] {
            let p = &self.links[cl].pixels;
            let slice = if p[0] == start {
                &p[..p.len() - 1]
            } else {
                &p[1..]
            };
            walked = sets::union(&walked, &sets::new(py, slice.iter().copied())?)?;
        }
        Ok(walked)
    }
    fn handle_bp(
        &mut self,
        py: Python<'_>,
        walker: &mut Walker,
        lid: usize,
        bp: usize,
    ) -> PyResult<()> {
        self.todo.remove(&lid);
        let done = self.node_ids.contains_key(&bp);
        let nid = self.node(bp, lid);
        self.links.get_mut(&lid).unwrap().conn.push(nid);
        if done {
            return Ok(());
        }
        let mut bps = vec![bp];
        walker.cluster(py, &mut bps)?;
        let mut em = sets::new(py, [])?;
        for &b in &bps {
            em = sets::union(&em, &walkable(py, &walker.image, &[b])?)?;
        }
        em = sets::sub(&em, &sets::new(py, bps.iter().copied())?)?;
        // find_emanators subtracts the cluster, then handle_bp subtracts it again.
        em = sets::sub(&em, &sets::new(py, bps.iter().copied())?)?;
        em = sets::sub(
            &em,
            &sets::new(py, self.links[&lid].pixels.iter().copied())?,
        )?;
        let mut emanators = sets::values(&em)?;
        let mut four = Vec::new();
        for &b in &bps {
            for &e in &emanators {
                if walker.image.is_four(b, e) {
                    four.push([b, e]);
                }
            }
        }
        emanators.retain(|e| !four.iter().any(|p| p[1] == *e));
        let mut diag = Vec::new();
        for &b in &bps {
            for &e in &emanators {
                let d = b.abs_diff(e);
                if d == walker.image.cols + 1 || d == walker.image.cols - 1 {
                    diag.push([b, e]);
                }
            }
        }
        let pairs = PySet::empty(py)?;
        for &b in &bps {
            let bn = walkable(py, &walker.image, &[b])?;
            for &bb in &bps {
                if bn.contains(bb)? {
                    let pair = sets::new(py, [b, bb])?;
                    pairs.add(PyTuple::new(py, sets::values(&pair)?)?)?;
                }
            }
        }
        for pair in pairs.iter() {
            self.issue(pair.extract()?, true);
        }
        for p in four.into_iter().chain(diag) {
            let node = self.node_ids[&p[0]];
            let done = self.conn[node].iter().any(|id| {
                let link = &self.links[id].pixels;
                same_pair(&link[link.len().saturating_sub(2)..], &p)
            });
            if !done {
                self.issue(p.to_vec(), false);
            }
        }
        Ok(())
    }
    fn check_duplicates(&mut self, py: Python<'_>, lid: usize) -> PyResult<()> {
        let link = &self.links[&lid];
        let tail = link.pixels[link.pixels.len().saturating_sub(2)..].to_vec();
        let connected = sets::new(py, self.conn[link.conn[1]].iter().copied())?;
        connected.discard(lid)?;
        for other in sets::values(&connected)? {
            let p = &self.links[&other].pixels;
            if same_pair(&tail, &p[..p.len().min(2)]) {
                self.delete(other)?;
                self.todo.remove(&other);
            }
        }
        Ok(())
    }
}
fn same_pair(a: &[usize], b: &[usize]) -> bool {
    a.iter().all(|x| b.contains(x)) && b.iter().all(|x| a.contains(x))
}
fn no_turn(a: usize, b: usize, cols: usize) -> Vec<usize> {
    let w = cols as isize;
    let diff = a as isize - b as isize;
    let dirs = if diff == -w - 1 {
        vec![1, w, w + 1]
    } else if diff == -w {
        vec![w - 1, w, w + 1]
    } else if diff == -w + 1 {
        vec![-1, w - 1, w]
    } else if diff == -1 {
        vec![-w + 1, 1, w + 1]
    }
    // The duplicated offset is intentional: it is part of the reference rule.
    else if diff == 1 {
        vec![-w - 1, -1, -w - 1]
    } else if diff == w - 1 {
        vec![-w, -w + 1, 1]
    } else if diff == w {
        vec![-w - 1, -w, -w + 1]
    } else {
        vec![-1, -w - 1, -w]
    };
    dirs.into_iter()
        .map(|d| (b as isize + d) as usize)
        .collect()
}
fn check_start(py: Python<'_>, walker: &mut Walker, p: usize) -> PyResult<bool> {
    let neighbors = walkable(py, &walker.image, &[p])?;
    // Empty-set pop deliberately propagates the reference KeyError for isolates.
    let n = sets::pop(&neighbors)?;
    Ok(!walker.is_bp(py, n)?)
}

pub fn resolve<'py>(
    py: Python<'py>,
    mut walker: Walker,
    original_cols: usize,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let components = walker.image.components(false);
    let endpoints =
        sets::new(
            py,
            walker.image.data.iter().enumerate().filter_map(|(i, &v)| {
                (v != 0 && walker.image.neighbors(i).len() == 1).then_some(i)
            }),
        )?;
    let mut starts = Vec::new();
    for component in &components {
        let pixels = sets::new(py, component.iter().copied())?;
        let possible = sets::op(&pixels, "intersection", &endpoints)?;
        for p in sets::values(&possible)? {
            if check_start(py, &mut walker, p)? {
                starts.push(p);
                break;
            }
        }
    }
    if starts.is_empty() {
        for component in &components {
            let mut seed = None;
            for &p in component {
                if check_start(py, &mut walker, p)? {
                    seed = Some(p);
                    break;
                }
            }
            if let Some(p) = seed.or_else(|| component.first().copied()) {
                starts.push(p);
            }
        }
    }
    if starts.is_empty() {
        return Err(PyRuntimeError::new_err(
            "Skeleton contains no walkable pixels; cannot resolve graph.",
        ));
    }
    let networks = starts.len();
    let mut graph = Graph::default();
    for (id, p) in starts.into_iter().enumerate() {
        let n = graph.node(p, id);
        let first = walkable(py, &walker.image, &[p])?;
        let mut pixels = vec![p];
        if first.is_empty() {
            graph.node(p, id);
        } else {
            pixels.push(sets::pop(&first)?);
        }
        graph.links.insert(
            id,
            Link {
                pixels,
                conn: vec![n],
            },
        );
        graph.todo.insert(id);
    }
    while let Some(&id) = graph.todo.first() {
        let cantwalk = graph.cant_walk(py, &mut walker, id)?;
        loop {
            let link = &graph.links[&id].pixels;
            let last = *link.last().unwrap();
            let mut possible = walkable(py, &walker.image, link)?;
            if possible.is_empty() {
                let nid = graph.node(last, id);
                graph.links.get_mut(&id).unwrap().conn.push(nid);
                graph.todo.remove(&id);
                break;
            }
            if link.len() < 4 {
                possible = sets::sub(&possible, &cantwalk)?;
            }
            let options = sets::values(&possible)?;
            if options.is_empty() {
                graph.delete(id)?;
                graph.todo.remove(&id);
                break;
            }
            let flags = options
                .iter()
                .map(|&p| walker.is_bp(py, p))
                .collect::<PyResult<Vec<_>>>()?;
            let nbp = flags.iter().filter(|&&v| v).count();
            let next = if options.len() == 1 {
                options[0]
            } else if nbp == 0 {
                let four: Vec<_> = options
                    .iter()
                    .copied()
                    .filter(|&p| walker.image.is_four(last, p))
                    .collect();
                let noturn: Vec<_> = no_turn(link[link.len() - 2], last, walker.image.cols)
                    .into_iter()
                    .filter(|p| options.contains(p))
                    .collect();
                if four.len() == 1 {
                    four[0]
                } else if noturn.len() == 1 {
                    noturn[0]
                } else {
                    return Err(PyRuntimeError::new_err("Ambiguous which step to take next :( Please raise issue at https://github.com/jonschwenk/RivGraph/issues."));
                }
            } else if nbp == 1 {
                options[flags.iter().position(|&v| v).unwrap()]
            } else {
                let valid: Vec<_> = options
                    .iter()
                    .zip(&flags)
                    .filter_map(|(&p, &bp)| (bp && walker.image.is_four(last, p)).then_some(p))
                    .collect();
                if valid.len() != 1 {
                    return Err(PyRuntimeError::new_err(
                        "There is not a unique branchpoint to step to.",
                    ));
                }
                valid[0]
            };
            graph.links.get_mut(&id).unwrap().pixels.push(next);
            if walker.is_bp(py, next)? {
                graph.handle_bp(py, &mut walker, id, next)?;
                graph.check_duplicates(py, id)?;
                break;
            }
        }
    }
    let unpad =
        |p: usize| (p / walker.image.cols - 20) * original_cols + p % walker.image.cols - 20;
    let links = PyDict::new(py);
    let nodes = PyDict::new(py);
    links.set_item("id", graph.links.keys().copied().collect::<Vec<_>>())?;
    links.set_item(
        "idx",
        graph
            .links
            .values()
            .map(|l| l.pixels.iter().map(|&p| unpad(p)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    links.set_item(
        "conn",
        graph
            .links
            .values()
            .map(|l| l.conn.clone())
            .collect::<Vec<_>>(),
    )?;
    links.set_item("n_networks", networks)?;
    nodes.set_item(
        "idx",
        graph.pixels.iter().map(|&p| unpad(p)).collect::<Vec<_>>(),
    )?;
    nodes.set_item("id", (0..graph.pixels.len()).collect::<Vec<_>>())?;
    nodes.set_item("conn", graph.conn)?;
    Ok((links, nodes))
}
