// Modified derivative of RivGraph's ln_utils.py (scan_mask, link_widths_and_lengths).
// Copyright 2019 Triad National Security, LLC. All rights reserved.
// Produced under U.S. Government contract 89233218CNA000001 for LANL.
// See ../LICENSE.txt for the complete LANL notice and BSD-3 terms.

//! Phase port of the width ray-marcher: one boundary crossing per
//! `link_widths_and_lengths` call. The ray DDA (`scan_mask_ray`) performs the
//! same float operations in the same order as `ln_utils._scan_mask`, so ray
//! distances are bit-identical to the Python reference; per-link statistics
//! differ only in mean summation order (numpy uses pairwise summation), which
//! the rtol <= 1e-9 parity gate absorbs.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

/// `np.isclose(step, 0.0)` with numpy's defaults (rtol=1e-5, atol=1e-8)
/// reduces to `|step| <= 1e-8` against 0.0; such steps march exactly axis
/// aligned. tests/test_native_widths.py pins this against the reference.
const STEP_ZERO_TOL: f64 = 1e-8;

/// Borrowed distance-transform slice plus its dimensions; the reference's
/// `Imask` is this data strictly greater than zero.
struct Dt<'a> {
    data: &'a [f64],
    rows: i64,
    cols: i64,
}

impl Dt<'_> {
    fn inside(&self, r: i64, c: i64) -> bool {
        self.data[(r * self.cols + c) as usize] > 0.0
    }
}

/// Widths/lengths of one link, matching the attributes the Python reference
/// appends to the links dict. `len`/`wid`/`wid_adj`/`len_adj` are returned
/// un-clamped; the Python wrapper applies `max(pixlen, value)` so the
/// tie-breaking and result types stay in the reference path.
struct LinkWidths {
    wid_pix: Vec<f64>,
    len: f64,
    wid: f64,
    wid_adj: f64,
    wid_med: f64,
    len_adj: f64,
    sinuosity: f64,
}

fn step_sign(step: f64) -> i64 {
    if step.abs() <= STEP_ZERO_TOL {
        0
    } else if step < 0.0 {
        -1
    } else {
        1
    }
}

/// First-crossing times for one axis: (t_max, t_delta), both infinite when
/// the ray does not move along this axis.
fn axis_terms(step: f64) -> (f64, f64) {
    match step_sign(step) {
        0 => (f64::INFINITY, f64::INFINITY),
        _ => {
            let abs = step.abs();
            (0.5 / abs, 1.0 / abs)
        }
    }
}

/// DDA ray march mirroring `ln_utils._scan_mask` exactly: identical float
/// operations in identical order, so the returned distance is bit-identical.
/// The stepping pixel is never mask-checked, exactly as in the reference.
fn scan_mask_ray(dt: &Dt<'_>, ray_r: f64, ray_c: f64, step_r: f64, step_c: f64) -> f64 {
    let (mut t_max_r, t_delta_r) = axis_terms(step_r);
    let (mut t_max_c, t_delta_c) = axis_terms(step_c);
    let (sign_r, sign_c) = (step_sign(step_r), step_sign(step_c));
    let (mut cur_r, mut cur_c) = (ray_r as i64, ray_c as i64);
    loop {
        let dist;
        if t_max_r < t_max_c {
            dist = t_max_r;
            cur_r += sign_r;
            t_max_r += t_delta_r;
        } else if t_max_c < t_max_r {
            dist = t_max_c;
            cur_c += sign_c;
            t_max_c += t_delta_c;
        } else {
            dist = t_max_r;
            cur_r += sign_r;
            cur_c += sign_c;
            t_max_r += t_delta_r;
            t_max_c += t_delta_c;
        }
        if cur_r < 0 || cur_r >= dt.rows || cur_c < 0 || cur_c >= dt.cols {
            return dist;
        }
        if !dt.inside(cur_r, cur_c) {
            return dist;
        }
    }
}

/// Width measured along the link's local cross-section normal, mirroring the
/// reference closure `width_along_normal`.
fn width_along_normal(
    dt: &Dt<'_>,
    link_r: &[i64],
    link_c: &[i64],
    pix_i: usize,
    fallback_width: f64,
    pixlen: f64,
) -> f64 {
    if link_r.len() == 1 {
        return fallback_width;
    }
    let (dr, dc) = if pix_i == 0 {
        (link_r[1] - link_r[0], link_c[1] - link_c[0])
    } else if pix_i == link_r.len() - 1 {
        (
            link_r[pix_i] - link_r[pix_i - 1],
            link_c[pix_i] - link_c[pix_i - 1],
        )
    } else {
        (
            link_r[pix_i + 1] - link_r[pix_i - 1],
            link_c[pix_i + 1] - link_c[pix_i - 1],
        )
    };
    // norm = np.hypot(nr, nc) on the pre-normalization integer components.
    let norm = (dc as f64).hypot(dr as f64);
    if norm == 0.0 {
        return fallback_width;
    }
    let nr = -(dc as f64) / norm;
    let nc = (dr as f64) / norm;
    let (ray_r, ray_c) = (link_r[pix_i] as f64, link_c[pix_i] as f64);
    let left = scan_mask_ray(dt, ray_r, ray_c, nr, nc);
    let right = scan_mask_ray(dt, ray_r, ray_c, -nr, -nc);
    (left + right) * pixlen
}

fn mean(values: &[f64]) -> f64 {
    // Left-to-right sum; numpy's pairwise summation is the only sanctioned
    // difference from the reference (absorbed by the 1e-9 rtol gate).
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// First index minimizing |value - target|, as np.argmin returns the first
/// minimum on ties.
fn argmin_abs(values: &[f64], target: f64) -> usize {
    let mut best = 0;
    let mut best_val = f64::INFINITY;
    for (i, &v) in values.iter().enumerate() {
        let d = (v - target).abs();
        if d < best_val {
            best_val = d;
            best = i;
        }
    }
    best
}

/// Cumulative pixel distances along a link (or its reverse), matching the
/// reference's np.cumsum + np.insert(0, ...) then * pixlen accumulation order.
fn cumdists(link_r: &[i64], link_c: &[i64], reverse: bool, pixlen: f64) -> Vec<f64> {
    let n = link_r.len();
    let mut dists = Vec::with_capacity(n);
    dists.push(0.0);
    let mut run = 0.0f64;
    for k in 0..n - 1 {
        let a = if reverse { n - 2 - k } else { k };
        let (dr, dc) = (link_r[a + 1] - link_r[a], link_c[a + 1] - link_c[a]);
        run += ((dr * dr + dc * dc) as f64).sqrt();
        dists.push(run);
    }
    for d in &mut dists {
        *d *= pixlen;
    }
    dists
}

/// Per-pixel widths plus trimmed/untrimmed per-link statistics for one link.
fn compute_link(dt: &Dt<'_>, pixlen: f64, idx: &[i64]) -> PyResult<LinkWidths> {
    let n = idx.len();
    if n == 0 {
        return Err(PyValueError::new_err(
            "Links must contain at least one pixel",
        ));
    }
    let mut link_r = Vec::with_capacity(n);
    let mut link_c = Vec::with_capacity(n);
    for &li in idx {
        if li < 0 || li >= dt.rows * dt.cols {
            return Err(PyValueError::new_err(
                "Link pixel index outside the distance transform",
            ));
        }
        link_r.push(li / dt.cols);
        link_c.push(li % dt.cols);
    }

    // Width at each pixel along the link.
    let mut widths = Vec::with_capacity(n);
    for (i, (r, c)) in link_r.iter().zip(link_c.iter()).enumerate() {
        let fallback = dt.data[(r * dt.cols + c) as usize] * 2.0 * pixlen;
        widths.push(width_along_normal(
            dt, &link_r, &link_c, i, fallback, pixlen,
        ));
    }

    // Trim the ends of each link by the endpoint half-widths.
    let dists = cumdists(&link_r, &link_c, false, pixlen);
    let revdists = cumdists(&link_r, &link_c, true, pixlen);
    let width_mult = 1.1;
    let startidx = argmin_abs(&dists, widths[0] / 2.0 * width_mult);
    let endidx = n - argmin_abs(&revdists, widths[n - 1] / 2.0 * width_mult) - 1;

    let (wid_adj, wid_med, len_adj, sinuosity);
    if startidx >= endidx {
        wid_adj = mean(&widths);
        wid_med = median(&widths);
        len_adj = dists[n - 1];
        let (dr, dc) = (link_r[0] - link_r[n - 1], link_c[0] - link_c[n - 1]);
        let st_dist = ((dr * dr + dc * dc) as f64).sqrt() * pixlen;
        sinuosity = dists[n - 1] / st_dist;
    } else {
        let trimmed = &widths[startidx..endidx];
        wid_adj = mean(trimmed);
        wid_med = median(trimmed);
        len_adj = dists[endidx] - dists[startidx];
        let (dr, dc) = (
            link_r[startidx] - link_r[endidx],
            link_c[startidx] - link_c[endidx],
        );
        let st_dist = ((dr * dr + dc * dc) as f64).sqrt() * pixlen;
        sinuosity = (dists[endidx] - dists[startidx]) / st_dist;
    }

    let wid_mean = mean(&widths);
    Ok(LinkWidths {
        wid_pix: widths,
        len: dists[n - 1],
        wid: wid_mean,
        wid_adj,
        wid_med,
        len_adj,
        sinuosity,
    })
}

fn contig_dt<'a>(dt: &'a PyReadonlyArray2<'_, f64>) -> PyResult<Dt<'a>> {
    let dims = dt.shape();
    let data = dt
        .as_slice()
        .map_err(|_| PyValueError::new_err("Distance transform must be C-contiguous"))?;
    Ok(Dt {
        data,
        rows: dims[0] as i64,
        cols: dims[1] as i64,
    })
}

#[pyfunction]
#[pyo3(name = "scan_mask")]
pub fn scan_mask(
    dt: PyReadonlyArray2<'_, f64>,
    ray_r: f64,
    ray_c: f64,
    step_r: f64,
    step_c: f64,
) -> PyResult<f64> {
    let grid = contig_dt(&dt)?;
    if step_sign(step_r) == 0 && step_sign(step_c) == 0 {
        // The reference's while-loop never advances on zero steps (and is
        // unreachable from width_along_normal, which guarantees a nonzero
        // normal); raise instead of hanging the interpreter uninterruptibly.
        return Err(PyValueError::new_err(
            "scan_mask received zero steps on both axes; the ray would never move",
        ));
    }
    Ok(scan_mask_ray(&grid, ray_r, ray_c, step_r, step_c))
}

#[pyfunction]
#[pyo3(name = "link_widths_and_lengths")]
pub fn link_widths_and_lengths<'py>(
    py: Python<'py>,
    idx_flat: PyReadonlyArray1<'py, i64>,
    link_lengths: PyReadonlyArray1<'py, i64>,
    dt: PyReadonlyArray2<'py, f64>,
    pixlen: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let grid = contig_dt(&dt)?;
    let idx = idx_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("Link indices must be C-contiguous"))?;
    let lengths = link_lengths
        .as_slice()
        .map_err(|_| PyValueError::new_err("Link lengths must be C-contiguous"))?;
    if lengths.iter().try_fold(0i64, |a, &b| a.checked_add(b)) != Some(idx.len() as i64) {
        return Err(PyValueError::new_err(
            "Link lengths must sum to the number of link indices",
        ));
    }

    let mut wid_pix = Vec::with_capacity(lengths.len());
    let mut len = Vec::with_capacity(lengths.len());
    let mut wid = Vec::with_capacity(lengths.len());
    let mut wid_adj = Vec::with_capacity(lengths.len());
    let mut wid_med = Vec::with_capacity(lengths.len());
    let mut len_adj = Vec::with_capacity(lengths.len());
    let mut sinuosity = Vec::with_capacity(lengths.len());

    let mut offset = 0usize;
    for &n_pixels in lengths {
        let count = n_pixels as usize;
        let lw = compute_link(&grid, pixlen, &idx[offset..offset + count])?;
        offset += count;
        wid_pix.push(PyArray1::from_slice(py, &lw.wid_pix));
        len.push(lw.len);
        wid.push(lw.wid);
        wid_adj.push(lw.wid_adj);
        wid_med.push(lw.wid_med);
        len_adj.push(lw.len_adj);
        sinuosity.push(lw.sinuosity);
    }

    let out = PyDict::new(py);
    out.set_item("wid_pix", wid_pix)?;
    out.set_item("len", len)?;
    out.set_item("wid", wid)?;
    out.set_item("wid_adj", wid_adj)?;
    out.set_item("wid_med", wid_med)?;
    out.set_item("len_adj", len_adj)?;
    out.set_item("sinuosity", sinuosity)?;
    Ok(out)
}
