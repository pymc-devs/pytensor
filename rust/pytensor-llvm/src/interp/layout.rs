//! Layouts (`crate::layout`) exchanged with Python as `(order, strides)`, a stride
//! being `"P"`, `"?"` or a number of bytes.

pub use crate::layout::{Layout, Stride};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

impl<'a, 'py> FromPyObject<'a, 'py> for Layout {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (order, strides): (Vec<usize>, Vec<Bound<'py, PyAny>>) = ob.extract()?;
        let strides = strides
            .iter()
            .map(|s| {
                if let Ok(n) = s.extract::<i64>() {
                    return Ok(Stride::Static(n));
                }
                match s.extract::<String>()?.as_str() {
                    "P" => Ok(Stride::Packed),
                    "?" => Ok(Stride::Dynamic),
                    other => Err(PyValueError::new_err(format!("unknown stride {other:?}"))),
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        if order.len() != strides.len() {
            return Err(PyValueError::new_err("a layout needs one stride per axis"));
        }
        Ok(Layout { order, strides })
    }
}

impl<'py> IntoPyObject<'py> for Layout {
    type Target = PyTuple;
    type Output = Bound<'py, PyTuple>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let strides = PyList::empty(py);
        for s in &self.strides {
            match s {
                Stride::Packed => strides.append(PyString::new(py, "P"))?,
                Stride::Dynamic => strides.append(PyString::new(py, "?"))?,
                Stride::Static(n) => strides.append(n)?,
            }
        }
        PyTuple::new(
            py,
            [PyList::new(py, &self.order)?.into_any(), strides.into_any()],
        )
    }
}
