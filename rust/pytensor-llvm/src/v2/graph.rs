//! Tensor graphs: what Python builds, what the interpreter runs and what a compiled unit
//! is generated from. Close to a PyTensor FunctionGraph.

use super::index::{Idx, IndexVec};
use super::op::TensorOp;
use super::types::{InputIdx, InputSig, OutputAlloc, OutputIdx, OutputSig, Signature, TensorTy};
use crate::define_index;
use crate::dtype::DType;
use std::sync::Arc;

define_index!(
    /// A variable of a graph.
    ValueId
);
define_index!(
    /// An apply node of a graph.
    NodeId
);
define_index!(
    /// A constant of a graph.
    ConstIdx
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Input(InputIdx),
    Const(ConstIdx),
    Node(NodeId, OutputIdx),
}

#[derive(Clone, Debug)]
pub struct Value {
    pub ty: TensorTy,
    pub origin: Origin,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Arc<dyn TensorOp>,
    pub inputs: IndexVec<InputIdx, ValueId>,
    pub outputs: IndexVec<OutputIdx, ValueId>,
    /// PyTensor `view_map`: the output may share memory with the input.
    pub views: Vec<(OutputIdx, InputIdx)>,
}

/// Constant data: native byte order, C-contiguous, 64-byte aligned.
pub struct ConstData {
    pub dtype: DType,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    buf: Vec<Align64>,
    nbytes: usize,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

impl ConstData {
    pub fn new(dtype: DType, shape: Vec<i64>, bytes: &[u8]) -> Result<Self, String> {
        let n = crate::abi::checked_nbytes(&shape, dtype.itemsize())
            .ok_or_else(|| crate::abi::bad_size("constant", &shape))?;
        if n != bytes.len() {
            return Err(format!("constant of shape {shape:?} needs {n} bytes, got {}", bytes.len()));
        }
        let mut buf = vec![Align64([0; 64]); n.div_ceil(64).max(1)];
        // SAFETY: `buf` holds at least `n` bytes; the regions don't overlap.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr() as *mut u8, n) };
        let strides = super::types::c_strides(&shape, dtype.itemsize() as i64);
        Ok(ConstData { dtype, shape, strides, buf, nbytes: n })
    }

    pub fn data(&self) -> *const u8 {
        self.buf.as_ptr() as *const u8
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the first `nbytes` bytes of `buf` are initialized.
        unsafe { std::slice::from_raw_parts(self.data(), self.nbytes) }
    }

    /// The value of a 0-d constant as (f64, i64).
    pub fn scalar(&self) -> Option<(f64, i64)> {
        if !self.shape.is_empty() {
            return None;
        }
        // SAFETY: a 0-d constant holds one element of `dtype`.
        unsafe { Some(super::kernels::read_scalar(self.data(), self.dtype)) }
    }
}

impl std::fmt::Debug for ConstData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConstData({:?}, {:?})", self.dtype, self.shape)
    }
}

/// A function graph. Nodes are in topological order (checked when they are added).
#[derive(Debug, Default)]
pub struct Graph {
    pub values: IndexVec<ValueId, Value>,
    pub nodes: IndexVec<NodeId, Node>,
    pub inputs: IndexVec<InputIdx, ValueId>,
    pub outputs: IndexVec<OutputIdx, ValueId>,
    pub consts: IndexVec<ConstIdx, Arc<ConstData>>,
}

impl Graph {
    pub fn add_input(&mut self, ty: TensorTy) -> ValueId {
        let i = InputIdx::new(self.inputs.len());
        let v = self.values.push(Value { ty, origin: Origin::Input(i) });
        self.inputs.push(v);
        v
    }

    pub fn add_const(&mut self, data: Arc<ConstData>) -> ValueId {
        let ty = TensorTy {
            dtype: data.dtype,
            shape: data.shape.iter().map(|&n| super::types::AxisLen::Static(n as u64)).collect(),
            layout: Some(super::types::Layout::C),
        };
        let c = self.consts.push(data);
        self.values.push(Value { ty, origin: Origin::Const(c) })
    }

    /// Add a node whose outputs have the given types. Returns the output values.
    pub fn add_node(
        &mut self,
        op: Arc<dyn TensorOp>,
        inputs: Vec<ValueId>,
        output_types: Vec<TensorTy>,
        views: Vec<(OutputIdx, InputIdx)>,
    ) -> Result<Vec<ValueId>, String> {
        for &v in &inputs {
            if self.values.get(v).is_none() {
                return Err(format!("{op:?}: input {v:?} does not exist"));
            }
        }
        for &(o, i) in &views {
            if o.index() >= output_types.len() || i.index() >= inputs.len() {
                return Err(format!("{op:?}: view_map entry {o:?} -> {i:?} out of range"));
            }
        }
        let id = NodeId::new(self.nodes.len());
        let outputs: IndexVec<OutputIdx, ValueId> = output_types
            .into_iter()
            .enumerate()
            .map(|(k, ty)| self.values.push(Value { ty, origin: Origin::Node(id, OutputIdx::new(k)) }))
            .collect();
        let node = Node { op, inputs: IndexVec::from_vec(inputs), outputs, views };
        let sig = self.signature_of(&node);
        node.op.check(&sig).map_err(|e| format!("{:?}: {e}", node.op))?;
        let outs = node.outputs.as_slice().to_vec();
        self.nodes.push(node);
        Ok(outs)
    }

    pub fn ty(&self, v: ValueId) -> &TensorTy {
        &self.values[v].ty
    }

    pub fn signature(&self, n: NodeId) -> Signature {
        self.signature_of(&self.nodes[n])
    }

    /// The signature of a node, derived from its types and view_map.
    pub fn signature_of(&self, node: &Node) -> Signature {
        let inputs = node.inputs.iter().map(|&v| InputSig { ty: self.ty(v).clone() }).collect();
        let outputs = node
            .outputs
            .iter_enumerated()
            .map(|(o, &v)| {
                let ty = self.ty(v).clone();
                let alloc = match node.views.iter().find(|(vo, _)| *vo == o) {
                    Some(&(_, i)) => OutputAlloc::ViewOf(i),
                    None if ty.static_shape().is_some() => OutputAlloc::Caller,
                    None => OutputAlloc::Callee,
                };
                OutputSig { ty, alloc }
            })
            .collect();
        Signature { inputs, outputs }
    }

    /// The signature of the graph itself, seen as a function.
    pub fn own_signature(&self) -> Signature {
        let inputs = self.inputs.iter().map(|&v| InputSig { ty: self.ty(v).clone() }).collect();
        let outputs = self
            .outputs
            .iter()
            .map(|&v| {
                let ty = self.ty(v).clone();
                let alloc = if ty.static_shape().is_some() { OutputAlloc::Caller } else { OutputAlloc::Callee };
                OutputSig { ty, alloc }
            })
            .collect();
        Signature { inputs, outputs }
    }

    /// Index of the last node that uses each value (None: unused or only a graph output).
    pub fn last_uses(&self) -> IndexVec<ValueId, Option<NodeId>> {
        let mut last: IndexVec<ValueId, Option<NodeId>> = self.values.iter().map(|_| None).collect();
        for (n, node) in self.nodes.iter_enumerated() {
            for &v in &node.inputs {
                last[v] = Some(n);
            }
        }
        last
    }

    /// The value whose memory `v` may share through view_map chains (itself if none).
    pub fn view_root(&self, mut v: ValueId) -> ValueId {
        loop {
            match self.values[v].origin {
                Origin::Node(n, o) => {
                    let node = &self.nodes[n];
                    match node.views.iter().find(|(vo, _)| *vo == o) {
                        Some(&(_, i)) => v = node.inputs[i],
                        None => return v,
                    }
                }
                _ => return v,
            }
        }
    }
}
