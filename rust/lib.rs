// Modified derivative of RivGraph's mask_to_graph.py, walk.py and im_utils.py.
// Copyright 2019 Triad National Security, LLC. All rights reserved.
// Produced under U.S. Government contract 89233218CNA000001 for LANL.
// See ../LICENSE.txt for the complete LANL notice and BSD-3 terms.
use pyo3::prelude::*;

#[pymodule]
fn _native(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
