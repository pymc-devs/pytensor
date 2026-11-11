//! Layouts as the kernels declare them (the Python side, `llvm/layout.py`, has the
//! full story): an axis order (outermost first) and per axis a packed, static or
//! dynamic stride. Exchanged with Python as `(order, strides)`, a stride being `"P"`,
//! `"?"` or a number of bytes.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

#[derive(Clone, Debug, PartialEq)]
pub enum Stride {
    Packed,
    Dynamic,
    Static(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub order: Vec<usize>,
    pub strides: Vec<Stride>,
}

impl Layout {
    /// C-contiguous.
    pub fn c(ndim: usize) -> Layout {
        Layout {
            order: (0..ndim).collect(),
            strides: vec![Stride::Packed; ndim],
        }
    }

    pub fn dynamic(ndim: usize) -> Layout {
        Layout {
            order: (0..ndim).collect(),
            strides: vec![Stride::Dynamic; ndim],
        }
    }

    pub fn ndim(&self) -> usize {
        self.strides.len()
    }

    /// The layout of a DimShuffle view (`order[k]`: input axis of output axis k, None
    /// for a new length-1 axis, which is packed). Dropped input axes have length 1.
    pub fn dimshuffle(&self, order: &[Option<usize>]) -> Layout {
        let mut out_of = vec![None; self.ndim()];
        let mut strides = Vec::with_capacity(order.len());
        for (j, o) in order.iter().enumerate() {
            match o {
                Some(a) => {
                    out_of[*a] = Some(j);
                    strides.push(self.strides[*a].clone());
                }
                None => strides.push(Stride::Packed),
            }
        }
        let mut new_order: Vec<usize> = order
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_none())
            .map(|(j, _)| j)
            .collect();
        new_order.extend(self.order.iter().filter_map(|&a| out_of[a]));
        Layout {
            order: new_order,
            strides,
        }
    }
}

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
