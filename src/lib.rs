use pyo3::prelude::*;

pub mod float;
pub mod line;
pub mod surface_sampling;
pub mod tmscnn;
pub mod triangle;
pub mod utils;

pub use tmscnn::Tmscnn;

#[pymodule]
fn tmscnn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tmscnn>()?;

    Ok(())
}
