//! What is known about the strides of an array (the Python side, `analysis/layout.py`,
//! has the full story): an axis order (outermost first) and per axis a packed, static or
//! dynamic stride. Kernels get the layouts of their inputs, and declare the layouts of
//! their outputs, as these. (The exchange with Python is in `python::layout`.)

#[derive(Clone, Debug, PartialEq)]
pub enum Stride {
    /// the product of the lengths of the axes inside it, times the itemsize
    Packed,
    /// unknown
    Dynamic,
    /// this many bytes
    Static(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    /// all axes, outermost first
    pub order: Vec<usize>,
    /// per axis
    pub strides: Vec<Stride>,
}

impl Layout {
    /// Packed in `order` (the axes, outermost first).
    pub fn packed(order: Vec<usize>) -> Layout {
        let strides = vec![Stride::Packed; order.len()];
        Layout { order, strides }
    }

    /// C-contiguous.
    pub fn c(ndim: usize) -> Layout {
        Layout::packed((0..ndim).collect())
    }

    /// Column-major (Fortran order): axis 0 innermost.
    pub fn column_major(ndim: usize) -> Layout {
        Layout::packed((0..ndim).rev().collect())
    }

    pub fn ndim(&self) -> usize {
        self.strides.len()
    }

    /// Is every axis packed (in `order`)?
    pub fn is_packed(&self) -> bool {
        self.strides.iter().all(|s| *s == Stride::Packed)
    }

    /// New memory of `ndim` axes in the memory order of an array with `like`, if that
    /// is fully known (packed, with as many axes), else C-contiguous.
    pub fn packed_like(like: Option<&Layout>, ndim: usize) -> Layout {
        match like {
            Some(l) if l.ndim() == ndim && l.is_packed() => Layout::packed(l.order.clone()),
            _ => Layout::c(ndim),
        }
    }
}
