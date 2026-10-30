//! Python's set iteration order is part of the legacy graph-ID contract.
//! Keep actual interpreter sets, not a Rust hash table or a sorted substitute.
use pyo3::{prelude::*, types::PySet};

pub type Set<'py> = Bound<'py, PySet>;
pub fn new(py: Python<'_>, xs: impl IntoIterator<Item = usize>) -> PyResult<Set<'_>> {
    PySet::new(py, xs)
}
pub fn values(s: &Set<'_>) -> PyResult<Vec<usize>> {
    s.iter().map(|v| v.extract()).collect()
}
pub fn op<'py>(a: &Set<'py>, name: &str, b: &Set<'py>) -> PyResult<Set<'py>> {
    Ok(a.call_method1(name, (b,))?.cast_into::<PySet>()?)
}
pub fn sub<'py>(a: &Set<'py>, b: &Set<'py>) -> PyResult<Set<'py>> {
    op(a, "__sub__", b)
}
pub fn union<'py>(a: &Set<'py>, b: &Set<'py>) -> PyResult<Set<'py>> {
    op(a, "__or__", b)
}
pub fn pop(s: &Set<'_>) -> PyResult<usize> {
    s.call_method0("pop")?.extract()
}
