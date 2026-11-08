//! Types of values and signatures of functions.
//!
//! A signature is derived mechanically from the node's types and the op's `view_map`
//! (see `Graph::signature`); ops never guess how their outputs are allocated.

use super::index::{Idx, IndexVec};
use crate::define_index;
use crate::dtype::DType;

define_index!(
    /// Position in a node's (or function's) inputs.
    InputIdx
);
define_index!(
    /// Position in a node's (or function's) outputs.
    OutputIdx
);
define_index!(
    /// An axis of an array.
    Axis
);

/// Length of one axis, as far as the type knows it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AxisLen {
    Static(u64),
    Dynamic,
}

impl AxisLen {
    pub fn known(self) -> Option<u64> {
        match self {
            AxisLen::Static(n) => Some(n),
            AxisLen::Dynamic => None,
        }
    }
}

/// Memory layout guaranteed by the type. `None` in `TensorTy` means unknown.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Layout {
    C,
    F,
}

/// The type of a value: PyTensor's TensorType (scalars are 0-d tensors).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorTy {
    pub dtype: DType,
    pub shape: Vec<AxisLen>,
    pub layout: Option<Layout>,
}

impl TensorTy {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn axis(&self, a: Axis) -> AxisLen {
        self.shape[a.index()]
    }

    pub fn axes(&self) -> impl Iterator<Item = Axis> + 'static {
        (0..self.shape.len()).map(Axis::new)
    }

    /// The full shape, if every axis length is static.
    pub fn static_shape(&self) -> Option<Vec<u64>> {
        self.shape.iter().map(|a| a.known()).collect()
    }

    /// Number of elements if the shape is static (None on overflow too).
    pub fn static_size(&self) -> Option<u64> {
        self.static_shape()?.iter().try_fold(1u64, |acc, &n| acc.checked_mul(n))
    }

    /// Byte size if the shape is static.
    pub fn static_nbytes(&self) -> Option<u64> {
        self.static_size()?.checked_mul(self.dtype.itemsize() as u64)
    }

    /// The axis is statically of length 1 (PyTensor's broadcastable).
    pub fn broadcastable(&self, a: Axis) -> bool {
        self.axis(a) == AxisLen::Static(1)
    }
}

/// How an output's memory comes to be.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OutputAlloc {
    /// The shape is static: the caller provides the memory.
    Caller,
    /// The callee allocates (through the runtime) and hands over the buffer.
    Callee,
    /// The callee returns a view of this input (PyTensor `view_map`). It may also
    /// return a fresh buffer, e.g. a reshape of a non-contiguous array.
    ViewOf(InputIdx),
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct InputSig {
    pub ty: TensorTy,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OutputSig {
    pub ty: TensorTy,
    pub alloc: OutputAlloc,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Signature {
    pub inputs: IndexVec<InputIdx, InputSig>,
    pub outputs: IndexVec<OutputIdx, OutputSig>,
}

impl Signature {
    pub fn input(&self, i: InputIdx) -> &TensorTy {
        &self.inputs[i].ty
    }

    pub fn output(&self, o: OutputIdx) -> &TensorTy {
        &self.outputs[o].ty
    }

    /// Position of an argument in the flat ABI argument array (inputs, then outputs).
    pub fn arg_of_output(&self, o: OutputIdx) -> usize {
        self.inputs.len() + o.index()
    }
}

/// C-contiguous byte strides for a shape.
pub fn c_strides(shape: &[i64], itemsize: i64) -> Vec<i64> {
    let mut st = vec![0i64; shape.len()];
    let mut acc = itemsize;
    for d in (0..shape.len()).rev() {
        st[d] = acc;
        acc = acc.saturating_mul(shape[d].max(1));
    }
    st
}
