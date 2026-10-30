// Modified RivGraph walk.py; copyright 2019 Triad National Security, LLC.
// LANL contract 89233218CNA000001. Full notice and BSD-3 terms: ../LICENSE.txt.
use crate::{
    image::Image,
    sets::{self, Set},
};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PySetMethods};
use std::collections::BTreeMap;

pub fn walkable<'py>(py: Python<'py>, image: &Image, link: &[usize]) -> PyResult<Set<'py>> {
    sets::sub(
        &sets::new(py, image.neighbors(link[link.len() - 1]))?,
        &sets::new(py, link[link.len().saturating_sub(2)..].iter().copied())?,
    )
}

fn walk_for_bps<'py>(py: Python<'py>, image: &Image, initial: &[usize]) -> PyResult<Set<'py>> {
    let bps = sets::new(py, initial.iter().copied())?;
    let first = sets::new(py, [])?;
    let mut emanators = sets::new(py, [])?;
    for bp in sets::values(&bps)? {
        emanators = sets::union(&emanators, &sets::new(py, image.neighbors(bp))?)?;
        for n in image.four(bp) {
            first.add(n)?;
        }
    }
    let walked = sets::union(&bps, &emanators)?;
    while !emanators.is_empty() {
        let mut idx = if !first.is_empty() {
            let i = sets::pop(&first)?;
            emanators.call_method1("remove", (i,))?;
            i
        } else {
            sets::pop(&emanators)?
        };
        loop {
            walked.add(idx)?;
            let neighs = sets::sub(&sets::new(py, image.neighbors(idx))?, &walked)?;
            match neighs.len() {
                0 => break,
                1 => {
                    idx = sets::pop(&neighs)?;
                    if image.edge(idx) {
                        break;
                    }
                }
                _ => {
                    bps.add(idx)?;
                    for n in image.four(idx) {
                        if neighs.contains(n)? {
                            first.add(n)?;
                        }
                    }
                    emanators = sets::union(&emanators, &neighs)?;
                    walked.add(idx)?;
                    walked.call_method1("update", (&neighs,))?;
                }
            }
        }
    }
    Ok(bps)
}

fn parsimonious(py: Python<'_>, ic: &Image, image: &Image) -> PyResult<Vec<usize>> {
    let possible = sets::sub(
        &sets::new(
            py,
            ic.data
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (v > 2).then_some(i)),
        )?,
        &sets::new(py, ic.edges())?,
    )?;
    let mut binary = ic.clone();
    for v in &mut binary.data {
        *v = u8::from(*v != 0);
    }
    let candidates: Vec<_> = sets::values(&possible)?
        .iter()
        .map(|&p| walk_for_bps(py, &binary, &[p]))
        .collect::<PyResult<_>>()?;
    let solo: Vec<_> = candidates
        .iter()
        .filter(|s| s.len() == 1)
        .map(sets::pop)
        .collect::<PyResult<_>>()?;
    let four = |i| {
        if image.edge(i) || image.data[i] != 1 {
            0
        } else {
            image.four(i).len()
        }
    };
    if !solo.is_empty() {
        let maxax = solo.iter().map(|&p| image.axes(p)).max().unwrap();
        let maxnax: Vec<_> = solo
            .iter()
            .copied()
            .filter(|&p| image.axes(p) == maxax)
            .collect();
        if maxnax.len() == 1 {
            return Ok(maxnax);
        }
        let maxfc = solo.iter().map(|&p| four(p)).max().unwrap();
        let maxfour: Vec<_> = solo.iter().copied().filter(|&p| four(p) == maxfc).collect();
        if maxfour.len() == 1 {
            return Ok(maxfour);
        }
        let maxfc = maxnax.iter().map(|&p| four(p)).max().unwrap();
        return Ok(vec![*maxnax
            .iter()
            .filter(|&&p| four(p) == maxfc)
            .min()
            .unwrap()]);
    }
    let matches = |pattern: [u8; 3]| -> Vec<usize> {
        (0..ic.data.len())
            .filter(|&i| [ic.data[i], image.axes(i), four(i) as u8] == pattern)
            .collect()
    };
    let mut must = Vec::new();
    for pattern in [
        [6, 4, 2],
        [5, 3, 1],
        [5, 3, 4],
        [3, 3, 2],
        [3, 3, 1],
        [4, 2, 4],
    ] {
        must.extend(matches(pattern));
    }
    let special = matches([4, 4, 2]);
    if special.len() == 2 {
        must.push(special[0]);
    }
    if !must.is_empty() {
        return sets::values(&walk_for_bps(py, &binary, &must)?);
    }
    let mincount = candidates
        .iter()
        .map(|s| s.len())
        .min()
        .ok_or_else(|| PyRuntimeError::new_err("No candidate branchpoints"))?;
    let smallest: Vec<_> = candidates.iter().filter(|s| s.len() == mincount).collect();
    if smallest.len() == 1 {
        return sets::values(smallest[0]);
    }
    let mut counts = BTreeMap::new();
    for s in &candidates {
        for p in sets::values(s)? {
            *counts.entry(p).or_insert(0usize) += 1;
        }
    }
    let max = counts.values().copied().max().unwrap();
    let mode = *counts.iter().find(|(_, v)| **v == max).unwrap().0;
    sets::values(&walk_for_bps(py, &binary, &[mode])?)
}

pub struct Walker {
    pub image: Image,
    cache: Vec<Option<bool>>,
}
impl Walker {
    pub fn new(image: Image) -> Self {
        let cache = vec![None; image.data.len()];
        Self { image, cache }
    }
    pub fn is_bp(&mut self, py: Python<'_>, idx: usize) -> PyResult<bool> {
        if let Some(bp) = self.cache[idx] {
            return Ok(bp);
        }
        let bp = self.compute_bp(py, idx)?;
        self.cache[idx] = Some(bp);
        Ok(bp)
    }
    fn compute_bp(&self, py: Python<'_>, idx: usize) -> PyResult<bool> {
        if self.image.neighbors(idx).len() < 3 {
            return Ok(false);
        }
        let (r, c) = (idx / self.image.cols, idx % self.image.cols);
        let (mut rows, mut cols) = (7, 7);
        let (mut image, mut ic, component, top, left) = loop {
            if r < rows / 2
                || c < cols / 2
                || r + rows / 2 >= self.image.rows
                || c + cols / 2 >= self.image.cols
            {
                return Err(PyRuntimeError::new_err(
                    "Branchpoint neighborhood exceeds skeleton padding",
                ));
            }
            let (top, left) = (r - rows / 2, c - cols / 2);
            let mut image = Image::zero(rows, cols);
            for y in 0..rows {
                image.data[y * cols..(y + 1) * cols].copy_from_slice(
                    &self.image.data[(top + y) * self.image.cols + left
                        ..(top + y) * self.image.cols + left + cols],
                );
            }
            let ic = image.connectivity();
            let mut high = ic.clone();
            for v in &mut high.data {
                *v = u8::from(*v > 2);
            }
            let center = rows / 2 * cols + cols / 2;
            let component = high
                .components(true)
                .into_iter()
                .find(|pixels| pixels.contains(&center))
                .ok_or_else(|| PyRuntimeError::new_err("Missing branchpoint component"))?;
            // Preserve the reference's transposed growth tests, including rectangles.
            let grow_rows = component
                .iter()
                .any(|&p| p % cols == 1 || p % cols == rows - 2);
            let grow_cols = component
                .iter()
                .any(|&p| p / cols == 1 || p / cols == cols - 2);
            if !grow_rows && !grow_cols {
                break (image, ic, component, top, left);
            }
            if grow_rows {
                rows += 4;
            }
            if grow_cols {
                cols += 4;
            }
        };
        let ymin = component.iter().map(|p| p / cols).min().unwrap();
        let ymax = component.iter().map(|p| p / cols).max().unwrap();
        let xmin = component.iter().map(|p| p % cols).min().unwrap();
        let xmax = component.iter().map(|p| p % cols).max().unwrap();
        for p in 0..image.data.len() {
            if p / cols < ymin.saturating_sub(1)
                || p / cols > ymax + 1
                || p % cols < xmin.saturating_sub(1)
                || p % cols > xmax + 1
            {
                image.data[p] = 0;
            }
        }
        let largest = image
            .components(false)
            .into_iter()
            .max_by_key(|p| p.len())
            .ok_or_else(|| PyRuntimeError::new_err("Empty branchpoint neighborhood"))?;
        image.data.fill(0);
        for p in largest {
            image.data[p] = 1;
        }
        for p in 0..ic.data.len() {
            if ic.data[p] > 2 && !component.contains(&p) {
                ic.data[p] = 1;
            }
            if image.data[p] != 1 {
                ic.data[p] = 0;
            }
        }
        if ic.data.iter().filter(|&&v| v > 2).count() == 1 {
            return Ok(true);
        }
        Ok(parsimonious(py, &ic, &image)?
            .iter()
            .any(|&p| (p / cols + top) * self.image.cols + p % cols + left == idx))
    }
    pub fn cluster(&mut self, py: Python<'_>, bps: &mut Vec<usize>) -> PyResult<()> {
        // An explicit stack preserves recursive DFS/pop ordering without a native stack limit.
        let mut stack = vec![walkable(py, &self.image, bps)?];
        while let Some(neighs) = stack.last() {
            if neighs.is_empty() {
                stack.pop();
                continue;
            }
            let b = sets::pop(neighs)?;
            if self.is_bp(py, b)? && !bps.contains(&b) {
                bps.push(b);
                stack.push(walkable(py, &self.image, bps)?);
            }
        }
        Ok(())
    }
}
