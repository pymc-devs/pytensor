//! Kernel specifications: everything a generated kernel depends on.
//!
//! A spec is content-hashed to get the kernel's symbol name and its on-disk cache key,
//! so two graphs that contain the same kernel share one compiled object.

use crate::dtype::DType;
use crate::ir::{FusedOut, IndexedAxis, ReduceOp, ScalarGraph};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Bump when code generation changes in a way that invalidates cached objects.
pub const CODEGEN_VERSION: u32 = 9;

/// What is statically known about an operand's memory layout. Strides passed at runtime
/// are always the truth; this only lets kernels skip runtime checks and pick loop order.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// C-contiguous (row major).
    C,
    /// Fortran-contiguous (column major).
    F,
    /// Unknown: checked at runtime.
    Any,
}

impl Layout {
    /// For ndim <= 1, C and F coincide; use C so equal kernels hash equally.
    pub fn normalize(self, ndim: usize) -> Layout {
        if ndim <= 1 && self == Layout::F { Layout::C } else { self }
    }

    /// Dimensions from outermost to innermost in memory order.
    pub fn order(self, ndim: usize) -> Vec<usize> {
        match self {
            Layout::F => (0..ndim).rev().collect(),
            _ => (0..ndim).collect(),
        }
    }

    /// Statically known to be contiguous in `target` order.
    pub fn contiguous_as(self, target: Layout, ndim: usize) -> bool {
        self != Layout::Any && (self == target || ndim <= 1)
    }

    /// Byte strides of a contiguous array with this layout (Any is treated as C).
    pub fn strides(self, shape: &[i64], itemsize: i64, out: &mut [i64]) {
        let n = shape.len();
        let mut acc = itemsize;
        for i in 0..n {
            // innermost first; no allocation (called on every execution)
            let d = if self == Layout::F { i } else { n - 1 - i };
            out[d] = acc;
            acc *= shape[d].max(1);
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Operand {
    pub dtype: DType,
    pub bcast: Vec<bool>,
    pub layout: Layout,
}

#[derive(Serialize, Clone, Debug)]
#[serde(tag = "kind")]
pub enum KernelSpec {
    Elemwise {
        ndim: usize,
        inputs: Vec<Operand>,
        /// Outputs are provided by the caller, all with the same (C or F) layout.
        outputs: Vec<Operand>,
        /// `inplace[k] = Some(i)`: output k is written into the buffer of input i.
        inplace: Vec<Option<usize>>,
        scalar: ScalarGraph,
        fastmath: bool,
    },
    /// Elemwise with indexed reads/writes and reductions in the loop.
    Fused {
        ndim: usize,
        /// all node inputs (Elemwise inputs, index arrays, write targets)
        inputs: Vec<Operand>,
        n_elem: usize,
        reads: Vec<Option<IndexedAxis>>,
        /// provided C-contiguous outputs
        outputs: Vec<Operand>,
        outs: Vec<FusedOut>,
        /// where each loop dim's length comes from at runtime: (argument, dim), None = 1
        dim_src: Vec<Option<(usize, usize)>>,
        scalar: ScalarGraph,
        fastmath: bool,
    },
    /// A segment of small nodes compiled together (see `segment.rs`).
    Segment(Box<crate::segment::SegmentSpec>),
    Reduce {
        op: ReduceOp,
        axes: Vec<usize>,
        input: Operand,
        /// Provided by the caller, C layout.
        output: Operand,
        fastmath: bool,
    },
}

impl KernelSpec {
    pub fn hash(&self, salt: &str) -> String {
        let mut h = Sha256::new();
        h.update(salt.as_bytes());
        h.update(serde_json::to_vec(self).expect("spec serializes"));
        hex::encode(&h.finalize()[..16])
    }

    pub fn describe(&self) -> String {
        match self {
            KernelSpec::Elemwise {
                ndim,
                scalar,
                inplace,
                ..
            } => {
                let ops: Vec<&str> = scalar.nodes.iter().map(|n| n.op.as_str()).collect();
                let ip = if inplace.iter().any(|x| x.is_some()) {
                    " inplace"
                } else {
                    ""
                };
                format!("Elemwise{{{}}} {}d{}", ops.join(","), ndim, ip)
            }
            KernelSpec::Reduce { op, axes, input, .. } => {
                format!("Reduce{{{:?}, axes={:?}}} {}d", op, axes, input.bcast.len())
            }
            KernelSpec::Segment(s) => format!("Segment[{} members]", s.members.len()),
            KernelSpec::Fused { ndim, scalar, reads, outs, .. } => {
                let ops: Vec<&str> = scalar.nodes.iter().map(|n| n.op.as_str()).collect();
                let n_reads = reads.iter().flatten().count();
                let tags: Vec<String> = outs
                    .iter()
                    .filter_map(|o| match o {
                        FusedOut::Plain => None,
                        FusedOut::Reduce { op, .. } => Some(format!("reduce:{op:?}")),
                        FusedOut::Write { set, .. } => Some(if *set { "set".to_string() } else { "inc".to_string() }),
                    })
                    .collect();
                format!("FusedElemwise{{{}}} {}d reads={} [{}]", ops.join(","), ndim, n_reads, tags.join(","))
            }
        }
    }
}
