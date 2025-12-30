use pyo3::prelude::*;

use cqlib_core::add;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok(add(a as u64, b as u64).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_native")]
fn binding_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
