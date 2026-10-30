// Modified derivative of RivGraph's mask_to_graph.py, walk.py and im_utils.py.
// Copyright 2019 Triad National Security, LLC. All rights reserved.
// Produced under U.S. Government contract 89233218CNA000001 for LANL.
// See ../LICENSE.txt for the complete LANL notice and BSD-3 terms.
mod branchpoints;
mod directionality;
mod graph;
mod image;
mod sets;
mod widths;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyfunction]
fn skel_to_graph<'py>(
    py: Python<'py>,
    skeleton: PyReadonlyArray2<'py, u8>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let shape = skeleton.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let padded_rows = rows
        .checked_add(40)
        .ok_or_else(|| PyValueError::new_err("Skeleton too large"))?;
    let padded_cols = cols
        .checked_add(40)
        .ok_or_else(|| PyValueError::new_err("Skeleton too large"))?;
    padded_rows
        .checked_mul(padded_cols)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| PyValueError::new_err("Skeleton too large"))?;
    let mut image = image::Image::zero(padded_rows, padded_cols);
    let input = skeleton.as_array();
    for ((r, c), &v) in input.indexed_iter() {
        if v > 1 {
            return Err(PyValueError::new_err("Skeleton must be binary"));
        }
        image.data[(r + 20) * padded_cols + c + 20] = v;
    }
    graph::resolve(py, branchpoints::Walker::new(image), cols)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(directionality::first_valid_cycle_flip, m)?)?;
    m.add_function(wrap_pyfunction!(skel_to_graph, m)?)?;
    m.add_function(wrap_pyfunction!(widths::scan_mask, m)?)?;
    m.add_function(wrap_pyfunction!(widths::link_widths_and_lengths, m)?)?;
    Ok(())
}
