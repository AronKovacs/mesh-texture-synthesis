use pyo3::prelude::*;

pub mod float;
pub mod tmscnn;
pub use tmscnn::Tmscnn;

#[pymodule]
fn tmscnn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tmscnn>()?;

    Ok(())
}
