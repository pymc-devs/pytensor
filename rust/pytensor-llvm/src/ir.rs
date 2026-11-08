//! The graph IR, built from Python through `GraphBuilder` (see `builder.rs`).
//!
//! It is deliberately close to PyTensor's own graph: typed variables (dtype + static
//! broadcast pattern) and Apply nodes in topological order. All decisions about buffers,
//! layouts and kernels are made by the planner.

use crate::dtype::DType;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct VarInfo {
    pub dtype: DType,
    /// Static broadcast pattern (`TensorType.broadcastable`). Length = ndim.
    pub bcast: Vec<bool>,
    /// Static shape (`TensorType.shape`), `None` for unknown dims.
    pub static_shape: Vec<Option<i64>>,
}

impl VarInfo {
    pub fn ndim(&self) -> usize {
        self.bcast.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub vars: Vec<VarInfo>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    /// Variables that are constants. Their data is passed separately, in the same order.
    pub constants: Vec<usize>,
    /// Nodes in topological order.
    pub nodes: Vec<Node>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
    Min,
    And,
    Or,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct IndexedAxis {
    /// node input position of a 1-d integer index array
    pub idx: usize,
    pub axis: usize,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub enum FusedOut {
    /// an ordinary output with the loop's shape
    Plain,
    /// reduced over loop `axes` with `op` (accumulates in the output dtype)
    Reduce { op: ReduceOp, axes: Vec<usize> },
    /// `target` (a node input) updated through an index: out = target; out[.., idx[i], ..] (+)= value
    Write { at: IndexedAxis, target: usize, set: bool },
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct ScanInfo {
    pub n_seqs: usize,
    pub mit_mot_in: Vec<Vec<i64>>,
    pub mit_mot_out: Vec<Vec<i64>>,
    pub mit_sot_in: Vec<Vec<i64>>,
    pub sit_sot_in: Vec<Vec<i64>>,
    pub n_untraced: usize,
    pub n_nit_sot: usize,
    pub n_non_seqs: usize,
    pub as_while: bool,
}

/// The step function of a Scan: a graph with its constants.
pub struct ScanInner {
    pub graph: Graph,
    pub consts: Vec<crate::runtime::ConstData>,
}

impl std::fmt::Debug for ScanInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScanInner({} nodes)", self.graph.nodes.len())
    }
}

/// One entry of a basic index. Numbers are positions in the node's inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexEntry {
    Int(usize),
    Slice { start: Option<usize>, stop: Option<usize>, step: Option<usize> },
}

#[derive(Clone, Debug)]
pub enum Op {
    /// Elemwise with a (possibly fused) scalar graph.
    Elemwise { scalar: ScalarGraph },
    /// CAReduce over `axes`; accumulation happens in the output dtype.
    Reduce { op: ReduceOp, axes: Vec<usize> },
    /// DimShuffle. `None` entries are new broadcastable ('x') dims. Declared view.
    DimShuffle { order: Vec<Option<usize>> },
    /// tensor.math.Dot (1d/2d operands). Precompiled Rust kernel.
    Dot,
    /// Cholesky factor of a square matrix (faer). NaN result if not positive definite.
    Cholesky { lower: bool },
    /// Solve A x = b for triangular A (faer); b is 1-d or 2-d.
    SolveTriangular { lower: bool, unit_diagonal: bool },
    /// CumOp: cumulative sum (or product) along an axis.
    CumOp { axis: usize, mul: bool },
    /// ExtractDiag: diagonal of two axes as a view (the diagonal is the last dim).
    ExtractDiag { offset: i64, axis1: usize, axis2: usize },
    /// SortOp along one axis. Precompiled Rust kernel.
    Sort { axis: usize, stable: bool },
    /// DeepCopyOp. Precompiled Rust kernel.
    Copy,
    /// Shape_i: 0-d int64 holding one dim of the input.
    ShapeI { i: usize },
    /// Shape: 1-d int64 vector with all dims of the input.
    Shape,
    /// MakeVector: stack 0-d inputs into a 1-d vector.
    MakeVector,
    /// CheckAndRaise: output is a declared view of input 0; inputs 1.. are 0-d conditions.
    Assert { msg: String },
    /// ARange(start, stop, step): kernel-allocated output.
    Arange,
    /// Alloc(value, *shape): kernel-allocated output.
    Alloc,
    /// AllocEmpty(*shape): kernel-allocated, uninitialized.
    AllocEmpty,
    /// PyTensor Scan: a loop running `inner` once per step (see `scan.rs`). Outer inputs
    /// and outputs follow Scan's order; all outputs are kernel-allocated.
    Scan { info: ScanInfo, inner: std::sync::Arc<ScanInner> },
    /// Basic indexing (integers and slices); declared view of input 0.
    Subtensor { entries: Vec<IndexEntry> },
    /// Basic set/inc: out = x with out[entries] (+)= y (broadcast). Inputs: x, y, index vars.
    IncSubtensor { entries: Vec<IndexEntry>, set: bool },
    /// x[..., idx, ...]: integer array `idx` (any ndim) on `axis`, all other axes whole.
    AdvancedTake { axis: usize },
    /// out = x with out[..., idx, ...] (+)= y; duplicate indices accumulate. Inputs: x, y, idx.
    AdvancedIncTake { axis: usize, set: bool },
    /// Concatenate along `axis`.
    Join { axis: usize },
    /// One Elemwise loop over `ndim` dims with indexed reads, indexed writes and
    /// reductions fused in (PyTensor's FusedElemwise). Inputs: the first `n_elem` are the
    /// Elemwise inputs (sources for indexed reads), followed by index arrays and write
    /// targets; positions below refer to the node's inputs.
    FusedElemwise {
        scalar: ScalarGraph,
        ndim: usize,
        n_elem: usize,
        /// per Elemwise input: read as `source[..., idx[i], ...]` on `axis`
        reads: Vec<Option<IndexedAxis>>,
        /// per Elemwise output / node output
        outs: Vec<FusedOut>,
    },
    /// Reshape(x, shape_vector): a view if x is known C-contiguous, else a copy.
    Reshape,
    /// A kernel compiled elsewhere (e.g. the numba fallback), called through the kernel
    /// ABI at `addr`. All outputs are kernel-allocated.
    Foreign { addr: usize, desc: String },
    /// Several small nodes compiled as one unit (created by `segment::form_segments`).
    /// `descs` describe the members, for error messages.
    Segment { spec: Box<crate::segment::SegmentSpec>, descs: Vec<String> },
}

/// A scalar expression DAG, the body of an Elemwise / Composite.
#[derive(Serialize, Clone, Debug)]
pub struct ScalarGraph {
    pub inputs: Vec<DType>,
    pub nodes: Vec<ScalarNode>,
    pub outputs: Vec<SRef>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ScalarNode {
    /// Lowercased PyTensor scalar op class name, e.g. "add", "exp", "switch".
    pub op: String,
    pub inputs: Vec<SRef>,
    /// Output dtype of the scalar op.
    pub dtype: DType,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "lowercase")]
pub enum SRef {
    In(usize),
    Node(usize),
    Const { f: f64, i: i64, dtype: DType },
}
