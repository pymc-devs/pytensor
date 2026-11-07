//! Python API for building graphs, implemented in Rust.
//!
//! ```python
//! from pytensor_llvm import GraphBuilder, ScalarGraph
//!
//! sg = ScalarGraph(["float64", "float64"])
//! out = sg.apply("add", [sg.input(0), sg.apply("exp", [sg.input(1)], "float64")], "float64")
//! sg.set_outputs([out])
//!
//! g = GraphBuilder()
//! x = g.variable("float64", [False]); g.add_input(x)
//! y = g.variable("float64", [False]); g.add_input(y)
//! z = g.variable("float64", [False])
//! g.elemwise(sg, [x, y], [z])
//! g.add_output(z)
//! fn = g.compile()
//! ```
//!
//! Every method validates its arguments immediately (dtypes, ndims, topological order),
//! so mistakes surface at the node that caused them.

use crate::codegen::scalar_op_arity;
use crate::compile::Options;
use crate::dtype::DType;
use crate::ir::{FusedOut, Graph, IndexEntry, IndexedAxis, Node, Op, ReduceOp, SRef, ScalarGraph as IrScalarGraph, ScalarNode, VarInfo};
use crate::py::CompiledFunction;
use crate::runtime::{ConstData, Program};
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

fn dtype(name: &str) -> PyResult<DType> {
    DType::from_name(name).ok_or_else(|| PyNotImplementedError::new_err(format!("dtype {name} is not supported")))
}

// ------------------------------------------------------------------ scalar graphs

/// A value inside a `ScalarGraph`: an input, a constant, or the result of a scalar op.
#[pyclass(module = "pytensor_llvm", frozen, from_py_object)]
#[derive(Clone)]
pub struct ScalarRef {
    r: SRef,
    dtype: DType,
}

#[pymethods]
impl ScalarRef {
    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.numpy_name()
    }
    fn __repr__(&self) -> String {
        format!("ScalarRef({:?}, {})", self.r, self.dtype.numpy_name())
    }
}

/// The scalar body of an Elemwise (a fused Composite or a single scalar op).
#[pyclass(module = "pytensor_llvm")]
pub struct ScalarGraph {
    g: IrScalarGraph,
    has_outputs: bool,
}

#[pymethods]
impl ScalarGraph {
    #[new]
    fn new(input_dtypes: Vec<String>) -> PyResult<Self> {
        let inputs = input_dtypes.iter().map(|d| dtype(d)).collect::<PyResult<_>>()?;
        Ok(ScalarGraph { g: IrScalarGraph { inputs, nodes: vec![], outputs: vec![] }, has_outputs: false })
    }

    fn input(&self, i: usize) -> PyResult<ScalarRef> {
        match self.g.inputs.get(i) {
            Some(&dt) => Ok(ScalarRef { r: SRef::In(i), dtype: dt }),
            None => err(format!("scalar input {i} out of range ({} inputs)", self.g.inputs.len())),
        }
    }

    fn constant(&self, value: f64, dtype_name: &str) -> PyResult<ScalarRef> {
        let dt = dtype(dtype_name)?;
        let i = if value.is_finite() { value as i64 } else { 0 };
        Ok(ScalarRef { r: SRef::Const { f: value, i, dtype: dt }, dtype: dt })
    }

    /// Apply scalar op `op` (lowercase PyTensor scalar op name) producing `dtype`.
    fn apply(&mut self, op: &str, inputs: Vec<ScalarRef>, dtype_name: &str) -> PyResult<ScalarRef> {
        let dt = dtype(dtype_name)?;
        let (lo, hi) = scalar_op_arity(op)
            .ok_or_else(|| PyNotImplementedError::new_err(format!("scalar op '{op}' is not supported")))?;
        if inputs.len() < lo || inputs.len() > hi {
            return err(format!("scalar op '{op}' takes {lo}..{hi} inputs, got {}", inputs.len()));
        }
        for r in &inputs {
            if let SRef::Node(j) = r.r {
                if j >= self.g.nodes.len() {
                    return err("ScalarRef belongs to a different ScalarGraph");
                }
            }
        }
        self.g.nodes.push(ScalarNode { op: op.to_string(), inputs: inputs.into_iter().map(|r| r.r).collect(), dtype: dt });
        Ok(ScalarRef { r: SRef::Node(self.g.nodes.len() - 1), dtype: dt })
    }

    fn set_outputs(&mut self, outputs: Vec<ScalarRef>) -> PyResult<()> {
        if outputs.is_empty() {
            return err("a ScalarGraph needs at least one output");
        }
        self.g.outputs = outputs.iter().map(|r| r.r.clone()).collect();
        self.has_outputs = true;
        Ok(())
    }

    #[getter]
    fn output_dtypes(&self) -> Vec<&'static str> {
        self.g
            .outputs
            .iter()
            .map(|r| match r {
                SRef::In(i) => self.g.inputs[*i],
                SRef::Node(j) => self.g.nodes[*j].dtype,
                SRef::Const { dtype, .. } => *dtype,
            })
            .map(|d| d.numpy_name())
            .collect()
    }
}

// ------------------------------------------------------------------ graphs

#[pyclass(module = "pytensor_llvm")]
pub struct GraphBuilder {
    graph: Graph,
    consts: Vec<ConstData>,
    defined: Vec<bool>,
    keepalive: Vec<Py<PyAny>>,
}

/// Parse PyTensor's `idx_list`: ints and (start, stop, step) triples of index-variable
/// numbers or None. Numbers are shifted by `offset` to become input positions.
fn parse_idx_list(idx_list: &[Bound<'_, PyAny>], offset: usize, n_vars: usize) -> PyResult<Vec<IndexEntry>> {
    let chk = |k: usize| -> PyResult<usize> {
        if k >= n_vars {
            return err(format!("idx_list refers to index variable {k}, but only {n_vars} were given"));
        }
        Ok(k + offset)
    };
    idx_list
        .iter()
        .map(|e| {
            if let Ok(k) = e.extract::<usize>() {
                return Ok(IndexEntry::Int(chk(k)?));
            }
            let (a, b, c): (Option<usize>, Option<usize>, Option<usize>) = e
                .extract()
                .map_err(|_| PyValueError::new_err("idx_list entries must be ints or (start, stop, step) tuples"))?;
            Ok(IndexEntry::Slice { start: a.map(chk).transpose()?, stop: b.map(chk).transpose()?, step: c.map(chk).transpose()? })
        })
        .collect()
}

impl GraphBuilder {
    fn check_index_vars(&self, op: &str, vars: &[usize]) -> PyResult<()> {
        for &v in vars {
            self.require(self.ndim(v) == 0 && !self.dt(v).is_float(), || format!("{op}: index variables must be 0-d integers"))?;
        }
        Ok(())
    }

    fn info(&self, v: usize) -> PyResult<&VarInfo> {
        self.graph.vars.get(v).ok_or_else(|| PyValueError::new_err(format!("unknown variable {v}")))
    }

    fn check_inputs(&self, op: &str, vs: &[usize]) -> PyResult<()> {
        for &v in vs {
            self.info(v)?;
            if !self.defined[v] {
                return err(format!("{op}: input v{v} is not defined yet (nodes must be added in topological order)"));
            }
        }
        Ok(())
    }

    fn check_outputs(&self, op: &str, vs: &[usize]) -> PyResult<()> {
        for &v in vs {
            self.info(v)?;
            if self.defined[v] {
                return err(format!("{op}: output v{v} is already defined"));
            }
        }
        Ok(())
    }

    fn ndim(&self, v: usize) -> usize {
        self.graph.vars[v].ndim()
    }

    fn dt(&self, v: usize) -> DType {
        self.graph.vars[v].dtype
    }

    fn push(&mut self, op: Op, inputs: Vec<usize>, outputs: Vec<usize>) {
        for &o in &outputs {
            self.defined[o] = true;
        }
        self.graph.nodes.push(Node { op, inputs, outputs });
    }

    fn require(&self, cond: bool, msg: impl FnOnce() -> String) -> PyResult<()> {
        if cond { Ok(()) } else { err(msg()) }
    }
}

#[pymethods]
impl GraphBuilder {
    #[new]
    fn new() -> Self {
        GraphBuilder { graph: Graph::default(), consts: vec![], defined: vec![], keepalive: vec![] }
    }

    /// Declare a tensor variable. Returns its id.
    #[pyo3(signature = (dtype_name, broadcastable, static_shape=None))]
    fn variable(&mut self, dtype_name: &str, broadcastable: Vec<bool>, static_shape: Option<Vec<Option<i64>>>) -> PyResult<usize> {
        let dt = dtype(dtype_name)?;
        let static_shape = static_shape.unwrap_or_else(|| vec![None; broadcastable.len()]);
        if static_shape.len() != broadcastable.len() {
            return err("static_shape and broadcastable have different lengths");
        }
        self.graph.vars.push(VarInfo { dtype: dt, bcast: broadcastable, static_shape });
        self.defined.push(false);
        Ok(self.graph.vars.len() - 1)
    }

    /// Declare a constant from a numpy array. Returns its id.
    #[pyo3(signature = (value, broadcastable=None))]
    fn constant(&mut self, py: Python<'_>, value: Bound<'_, PyAny>, broadcastable: Option<Vec<bool>>) -> PyResult<usize> {
        let np = py.import("numpy")?;
        let kw = PyDict::new(py);
        kw.set_item("order", "C")?;
        // np.array(order="C") keeps 0-d arrays 0-d (ascontiguousarray would not)
        let arr = np.getattr("array")?.call((value,), Some(&kw))?;
        let arr = arr.cast::<PyUntypedArray>()?;
        let d = arr.dtype();
        let dt = DType::from_numpy(d.kind(), d.itemsize())
            .ok_or_else(|| PyNotImplementedError::new_err("constant dtype is not supported"))?;
        let shape: Vec<i64> = arr.shape().iter().map(|&s| s as i64).collect();
        let strides: Vec<i64> = arr.strides().iter().map(|&s| s as i64).collect();
        let bcast = broadcastable.unwrap_or_else(|| shape.iter().map(|&s| s == 1).collect());
        if bcast.len() != shape.len() {
            return err("broadcastable does not match the constant's ndim");
        }
        let nbytes = arr.len() * d.itemsize();
        let data = unsafe { (*arr.as_array_ptr()).fields.data as *const u8 };
        let bytes = unsafe { std::slice::from_raw_parts(data, nbytes) };
        let static_shape = shape.iter().map(|&s| Some(s)).collect();
        self.graph.vars.push(VarInfo { dtype: dt, bcast, static_shape });
        self.defined.push(true);
        let v = self.graph.vars.len() - 1;
        self.graph.constants.push(v);
        self.consts.push(ConstData::from_bytes(bytes, shape, strides));
        Ok(v)
    }

    fn add_input(&mut self, v: usize) -> PyResult<()> {
        self.info(v)?;
        if self.defined[v] {
            return err(format!("input v{v} is already defined"));
        }
        self.defined[v] = true;
        self.graph.inputs.push(v);
        Ok(())
    }

    fn add_output(&mut self, v: usize) -> PyResult<()> {
        self.check_inputs("output", &[v])?;
        self.graph.outputs.push(v);
        Ok(())
    }

    fn elemwise(&mut self, scalar: &ScalarGraph, inputs: Vec<usize>, outputs: Vec<usize>) -> PyResult<()> {
        self.check_inputs("elemwise", &inputs)?;
        self.check_outputs("elemwise", &outputs)?;
        if !scalar.has_outputs {
            return err("elemwise: ScalarGraph has no outputs (call set_outputs)");
        }
        let nd = self.ndim(outputs[0]);
        for &v in inputs.iter().chain(&outputs) {
            self.require(self.ndim(v) == nd, || format!("elemwise: all operands need {nd} dims, v{v} has {}", self.ndim(v)))?;
        }
        self.require(scalar.g.inputs.len() == inputs.len(), || "elemwise: ScalarGraph input count differs".into())?;
        for (i, &v) in inputs.iter().enumerate() {
            self.require(scalar.g.inputs[i] == self.dt(v), || format!("elemwise: scalar input {i} dtype differs from v{v}"))?;
        }
        let odts = scalar.output_dtypes();
        self.require(odts.len() == outputs.len(), || "elemwise: ScalarGraph output count differs".into())?;
        for (k, &v) in outputs.iter().enumerate() {
            self.require(odts[k] == self.dt(v).numpy_name(), || format!("elemwise: output {k} dtype differs from v{v}"))?;
        }
        self.push(Op::Elemwise { scalar: scalar.g.clone() }, inputs, outputs);
        Ok(())
    }

    /// CAReduce with `op` in {"add", "mul", "max", "min", "and", "or"}.
    fn reduce(&mut self, op: &str, axes: Vec<usize>, input: usize, output: usize) -> PyResult<()> {
        self.check_inputs("reduce", &[input])?;
        self.check_outputs("reduce", &[output])?;
        let rop = match op {
            "add" => ReduceOp::Add,
            "mul" => ReduceOp::Mul,
            "max" => ReduceOp::Max,
            "min" => ReduceOp::Min,
            "and" => ReduceOp::And,
            "or" => ReduceOp::Or,
            _ => return Err(PyNotImplementedError::new_err(format!("reduce op '{op}' is not supported"))),
        };
        let mut axes = axes;
        axes.sort();
        axes.dedup();
        let nd = self.ndim(input);
        self.require(axes.iter().all(|&a| a < nd), || "reduce: axis out of range".into())?;
        self.require(self.ndim(output) == nd - axes.len(), || "reduce: output ndim is wrong".into())?;
        self.push(Op::Reduce { op: rop, axes }, vec![input], vec![output]);
        Ok(())
    }

    /// Declared view. `order` entries are input dims, or None for a new broadcastable dim.
    fn dimshuffle(&mut self, order: Vec<Option<usize>>, input: usize, output: usize) -> PyResult<()> {
        self.check_inputs("dimshuffle", &[input])?;
        self.check_outputs("dimshuffle", &[output])?;
        let nd = self.ndim(input);
        self.require(order.len() == self.ndim(output), || "dimshuffle: order length must equal output ndim".into())?;
        self.require(order.len() <= crate::abi::MAX_NDIM, || "too many dims".into())?;
        for d in 0..nd {
            if !order.contains(&Some(d)) {
                self.require(self.graph.vars[input].bcast[d], || format!("dimshuffle: dropped dim {d} is not broadcastable"))?;
            }
        }
        self.require(order.iter().flatten().all(|&d| d < nd), || "dimshuffle: dim out of range".into())?;
        self.push(Op::DimShuffle { order }, vec![input], vec![output]);
        Ok(())
    }

    fn dot(&mut self, a: usize, b: usize, output: usize) -> PyResult<()> {
        self.check_inputs("dot", &[a, b])?;
        self.check_outputs("dot", &[output])?;
        let (na, nb) = (self.ndim(a), self.ndim(b));
        self.require((1..=2).contains(&na) && (1..=2).contains(&nb), || "dot: operands must be 1-d or 2-d".into())?;
        self.require(self.ndim(output) == na + nb - 2, || "dot: output ndim is wrong".into())?;
        self.push(Op::Dot, vec![a, b], vec![output]);
        Ok(())
    }

    #[pyo3(signature = (input, output, axis, stable=false))]
    fn sort(&mut self, input: usize, output: usize, axis: usize, stable: bool) -> PyResult<()> {
        self.check_inputs("sort", &[input])?;
        self.check_outputs("sort", &[output])?;
        self.require(axis < self.ndim(input), || "sort: axis out of range".into())?;
        self.push(Op::Sort { axis, stable }, vec![input], vec![output]);
        Ok(())
    }

    fn copy(&mut self, input: usize, output: usize) -> PyResult<()> {
        self.check_inputs("copy", &[input])?;
        self.check_outputs("copy", &[output])?;
        self.require(self.ndim(input) == self.ndim(output) && self.dt(input) == self.dt(output), || "copy: type mismatch".into())?;
        self.push(Op::Copy, vec![input], vec![output]);
        Ok(())
    }

    fn shape_i(&mut self, input: usize, output: usize, i: usize) -> PyResult<()> {
        self.check_inputs("shape_i", &[input])?;
        self.check_outputs("shape_i", &[output])?;
        self.require(i < self.ndim(input), || "shape_i: dim out of range".into())?;
        self.require(self.ndim(output) == 0 && self.dt(output) == DType::Int64, || "shape_i: output must be 0-d int64".into())?;
        self.push(Op::ShapeI { i }, vec![input], vec![output]);
        Ok(())
    }

    fn shape(&mut self, input: usize, output: usize) -> PyResult<()> {
        self.check_inputs("shape", &[input])?;
        self.check_outputs("shape", &[output])?;
        self.require(self.ndim(output) == 1 && self.dt(output) == DType::Int64, || "shape: output must be 1-d int64".into())?;
        self.push(Op::Shape, vec![input], vec![output]);
        Ok(())
    }

    fn make_vector(&mut self, inputs: Vec<usize>, output: usize) -> PyResult<()> {
        self.check_inputs("make_vector", &inputs)?;
        self.check_outputs("make_vector", &[output])?;
        self.require(self.ndim(output) == 1, || "make_vector: output must be 1-d".into())?;
        self.push(Op::MakeVector, inputs, vec![output]);
        Ok(())
    }

    /// Output is a declared view of `input`; fails with `msg` if any 0-d condition is false.
    fn check_and_raise(&mut self, input: usize, conds: Vec<usize>, output: usize, msg: String) -> PyResult<()> {
        let mut ins = vec![input];
        ins.extend(&conds);
        self.check_inputs("check_and_raise", &ins)?;
        self.check_outputs("check_and_raise", &[output])?;
        self.require(self.ndim(input) == self.ndim(output), || "check_and_raise: type mismatch".into())?;
        self.push(Op::Assert { msg }, ins, vec![output]);
        Ok(())
    }

    /// numpy.arange; the kernel allocates its output.
    fn arange(&mut self, start: usize, stop: usize, step: usize, output: usize) -> PyResult<()> {
        self.check_inputs("arange", &[start, stop, step])?;
        self.check_outputs("arange", &[output])?;
        for v in [start, stop, step] {
            self.require(self.ndim(v) == 0, || "arange: inputs must be 0-d".into())?;
        }
        self.require(self.ndim(output) == 1, || "arange: output must be 1-d".into())?;
        self.push(Op::Arange, vec![start, stop, step], vec![output]);
        Ok(())
    }

    /// Broadcast `value` to the shape given by 0-d integer variables; the kernel allocates.
    fn alloc(&mut self, value: usize, shape: Vec<usize>, output: usize) -> PyResult<()> {
        let mut ins = vec![value];
        ins.extend(&shape);
        self.check_inputs("alloc", &ins)?;
        self.check_outputs("alloc", &[output])?;
        self.require(shape.len() <= crate::abi::MAX_NDIM, || "too many dims".into())?;
        self.require(shape.iter().all(|&s| self.ndim(s) == 0 && !self.dt(s).is_float()), || "alloc: shape entries must be 0-d integers".into())?;
        self.require(self.ndim(output) == shape.len() && self.ndim(value) <= shape.len(), || "alloc: ndim mismatch".into())?;
        self.require(self.dt(value) == self.dt(output), || "alloc: dtype mismatch".into())?;
        self.push(Op::Alloc, ins, vec![output]);
        Ok(())
    }

    /// A kernel compiled elsewhere, called through the kernel ABI at `address`
    /// (`int32 kernel(PtCtx*, PtArray*, const void*)`). Every output must be allocated by
    /// the kernel. `owner` keeps the code alive (e.g. a numba cfunc object).
    fn foreign(&mut self, address: usize, inputs: Vec<usize>, outputs: Vec<usize>, desc: String, owner: Py<PyAny>) -> PyResult<()> {
        self.check_inputs("foreign", &inputs)?;
        self.check_outputs("foreign", &outputs)?;
        self.require(address != 0, || "foreign: null kernel address".into())?;
        self.keepalive.push(owner);
        self.push(Op::Foreign { addr: address, desc }, inputs, outputs);
        Ok(())
    }

    /// Basic indexing; the output is a declared view of `x`. `idx_list` uses PyTensor's
    /// encoding (ints and (start, stop, step) triples of numbers into `index_vars`).
    fn subtensor(&mut self, x: usize, idx_list: Vec<Bound<'_, PyAny>>, index_vars: Vec<usize>, output: usize) -> PyResult<()> {
        let mut ins = vec![x];
        ins.extend(&index_vars);
        self.check_inputs("subtensor", &ins)?;
        self.check_outputs("subtensor", &[output])?;
        self.check_index_vars("subtensor", &index_vars)?;
        let entries = parse_idx_list(&idx_list, 1, index_vars.len())?;
        let n_int = entries.iter().filter(|e| matches!(e, IndexEntry::Int(_))).count();
        self.require(entries.len() <= self.ndim(x), || "subtensor: too many indices".into())?;
        self.require(self.ndim(output) == self.ndim(x) - n_int && self.dt(output) == self.dt(x), || "subtensor: output type is wrong".into())?;
        self.push(Op::Subtensor { entries }, ins, vec![output]);
        Ok(())
    }

    /// out = x with out[idx] = y (set) or += y; y broadcasts against the indexed region.
    #[pyo3(signature = (x, y, idx_list, index_vars, output, set=false))]
    fn inc_subtensor(&mut self, x: usize, y: usize, idx_list: Vec<Bound<'_, PyAny>>, index_vars: Vec<usize>, output: usize, set: bool) -> PyResult<()> {
        let mut ins = vec![x, y];
        ins.extend(&index_vars);
        self.check_inputs("inc_subtensor", &ins)?;
        self.check_outputs("inc_subtensor", &[output])?;
        self.check_index_vars("inc_subtensor", &index_vars)?;
        let entries = parse_idx_list(&idx_list, 2, index_vars.len())?;
        self.require(entries.len() <= self.ndim(x), || "inc_subtensor: too many indices".into())?;
        self.require(self.dt(y) == self.dt(x), || "inc_subtensor: y must have x's dtype (cast it first)".into())?;
        self.require(self.ndim(output) == self.ndim(x) && self.dt(output) == self.dt(x), || "inc_subtensor: output type is wrong".into())?;
        self.push(Op::IncSubtensor { entries, set }, ins, vec![output]);
        Ok(())
    }

    /// x[..., idx, ...] with an integer array `idx` on `axis` (numpy.take).
    fn take(&mut self, x: usize, idx: usize, axis: usize, output: usize) -> PyResult<()> {
        self.check_inputs("take", &[x, idx])?;
        self.check_outputs("take", &[output])?;
        self.require(axis < self.ndim(x) && !self.dt(idx).is_float() && self.dt(idx) != DType::Bool, || "take: bad axis or index dtype".into())?;
        self.require(self.ndim(output) == self.ndim(x) - 1 + self.ndim(idx) && self.dt(output) == self.dt(x), || "take: output type is wrong".into())?;
        self.push(Op::AdvancedTake { axis }, vec![x, idx], vec![output]);
        Ok(())
    }

    /// out = x with out[..., idx, ...] = y (set) or += y; duplicate indices accumulate.
    #[pyo3(signature = (x, y, idx, axis, output, set=false))]
    fn inc_take(&mut self, x: usize, y: usize, idx: usize, axis: usize, output: usize, set: bool) -> PyResult<()> {
        self.check_inputs("inc_take", &[x, y, idx])?;
        self.check_outputs("inc_take", &[output])?;
        self.require(axis < self.ndim(x) && !self.dt(idx).is_float() && self.dt(idx) != DType::Bool, || "inc_take: bad axis or index dtype".into())?;
        self.require(self.dt(y) == self.dt(x), || "inc_take: y must have x's dtype (cast it first)".into())?;
        self.require(self.ndim(output) == self.ndim(x) && self.dt(output) == self.dt(x), || "inc_take: output type is wrong".into())?;
        self.push(Op::AdvancedIncTake { axis, set }, vec![x, y, idx], vec![output]);
        Ok(())
    }

    /// One Elemwise loop over `ndim` dims with indexed reads, indexed writes and reductions
    /// (PyTensor's FusedElemwise). `inputs[:n_elem]` are the Elemwise inputs, the rest
    /// index arrays and write targets. `reads[j]`: None or (index position, axis) to read
    /// input j as `input[.., idx[i], ..]`. `outs[k]`: ("plain",), ("reduce", op, axes) or
    /// ("write", index position, axis, target position, set). Positions refer to `inputs`.
    #[allow(clippy::too_many_arguments)]
    fn fused_elemwise(
        &mut self,
        scalar: &ScalarGraph,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
        ndim: usize,
        n_elem: usize,
        reads: Vec<Option<(usize, usize)>>,
        outs: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.check_inputs("fused_elemwise", &inputs)?;
        self.check_outputs("fused_elemwise", &outputs)?;
        let name = "fused_elemwise";
        self.require(ndim <= crate::abi::MAX_NDIM && n_elem <= inputs.len(), || format!("{name}: bad ndim or n_elem"))?;
        self.require(scalar.has_outputs && scalar.g.inputs.len() == n_elem && reads.len() == n_elem, || format!("{name}: ScalarGraph/reads don't match n_elem"))?;
        let is_index = |b: &Self, p: usize| p < inputs.len() && p >= n_elem && b.ndim(inputs[p]) == 1 && !b.dt(inputs[p]).is_float() && b.dt(inputs[p]) != DType::Bool;
        let mut ir_reads = Vec::new();
        for j in 0..n_elem {
            let v = inputs[j];
            self.require(scalar.g.inputs[j] == self.dt(v), || format!("{name}: scalar input {j} dtype differs"))?;
            match reads[j] {
                Some((idx, axis)) => {
                    self.require(is_index(self, idx) && axis < ndim && self.ndim(v) == ndim, || format!("{name}: bad indexed read for input {j}"))?;
                    ir_reads.push(Some(IndexedAxis { idx, axis }));
                }
                None => {
                    self.require(self.ndim(v) == ndim || self.ndim(v) == 0, || format!("{name}: input {j} must have {ndim} dims or be 0-d"))?;
                    ir_reads.push(None);
                }
            }
        }
        let odts = scalar.output_dtypes();
        self.require(odts.len() == outputs.len() && outs.len() == outputs.len(), || format!("{name}: output count differs"))?;
        let mut ir_outs = Vec::new();
        for (k, spec) in outs.iter().enumerate() {
            let o = outputs[k];
            self.require(odts[k] == self.dt(o).numpy_name(), || format!("{name}: output {k} dtype differs from the scalar output"))?;
            let kind: String = spec.get_item(0)?.extract()?;
            ir_outs.push(match kind.as_str() {
                "plain" => {
                    self.require(self.ndim(o) == ndim, || format!("{name}: plain output {k} needs {ndim} dims"))?;
                    FusedOut::Plain
                }
                "reduce" => {
                    let op: String = spec.get_item(1)?.extract()?;
                    let mut axes: Vec<usize> = spec.get_item(2)?.extract()?;
                    axes.sort();
                    axes.dedup();
                    let op = match op.as_str() {
                        "add" => ReduceOp::Add,
                        "mul" => ReduceOp::Mul,
                        "max" => ReduceOp::Max,
                        "min" => ReduceOp::Min,
                        "and" => ReduceOp::And,
                        "or" => ReduceOp::Or,
                        _ => return Err(PyNotImplementedError::new_err(format!("{name}: reduction '{op}'"))),
                    };
                    self.require(axes.iter().all(|&a| a < ndim) && self.ndim(o) == ndim - axes.len(), || format!("{name}: bad reduction for output {k}"))?;
                    FusedOut::Reduce { op, axes }
                }
                "write" => {
                    let (_, idx, axis, target, set): (String, usize, usize, usize, bool) = spec.extract()?;
                    self.require(is_index(self, idx) && axis < ndim, || format!("{name}: bad index for write {k}"))?;
                    self.require(target < inputs.len() && target >= n_elem, || format!("{name}: bad target for write {k}"))?;
                    let t = inputs[target];
                    self.require(self.ndim(t) == ndim && self.ndim(o) == ndim && self.dt(t) == self.dt(o), || format!("{name}: write target {k} type"))?;
                    FusedOut::Write { at: IndexedAxis { idx, axis }, target, set }
                }
                _ => return err(format!("{name}: unknown output kind '{kind}'")),
            });
        }
        self.push(Op::FusedElemwise { scalar: scalar.g.clone(), ndim, n_elem, reads: ir_reads, outs: ir_outs }, inputs, outputs);
        Ok(())
    }

    fn join(&mut self, inputs: Vec<usize>, axis: usize, output: usize) -> PyResult<()> {
        self.check_inputs("join", &inputs)?;
        self.check_outputs("join", &[output])?;
        self.require(!inputs.is_empty(), || "join: no inputs".into())?;
        let nd = self.ndim(output);
        self.require(axis < nd && inputs.iter().all(|&v| self.ndim(v) == nd && self.dt(v) == self.dt(output)), || "join: inputs need the output's ndim and dtype".into())?;
        self.push(Op::Join { axis }, inputs, vec![output]);
        Ok(())
    }

    fn reshape(&mut self, x: usize, shape: usize, output: usize) -> PyResult<()> {
        self.check_inputs("reshape", &[x, shape])?;
        self.check_outputs("reshape", &[output])?;
        self.require(self.ndim(shape) == 1 && !self.dt(shape).is_float(), || "reshape: shape must be a 1-d integer vector".into())?;
        self.require(self.dt(output) == self.dt(x) && self.ndim(output) <= crate::abi::MAX_NDIM, || "reshape: output type is wrong".into())?;
        self.push(Op::Reshape, vec![x, shape], vec![output]);
        Ok(())
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Plan buffers, generate and compile kernels. The builder can't be reused afterwards.
    #[pyo3(signature = (fastmath=true, opt_level=3, cache=true, dump_ir=None, segments=true, segment_max_elems=64))]
    #[allow(clippy::too_many_arguments)]
    fn compile(
        &mut self,
        py: Python<'_>,
        fastmath: bool,
        opt_level: u32,
        cache: bool,
        dump_ir: Option<String>,
        segments: bool,
        segment_max_elems: i64,
    ) -> PyResult<CompiledFunction> {
        if self.graph.outputs.is_empty() && self.graph.nodes.is_empty() && self.graph.inputs.is_empty() {
            return err("empty graph");
        }
        let graph = std::mem::take(&mut self.graph);
        let consts = std::mem::take(&mut self.consts);
        self.defined.clear();
        let opts = Options { fastmath, segments, segment_max_elems, opt_level, cache, dump_ir: dump_ir.map(Into::into) };
        let mut prog = py
            .detach(move || Program::new(graph, consts, &opts))
            .map_err(PyNotImplementedError::new_err)?;
        for obj in std::mem::take(&mut self.keepalive) {
            prog.keep_alive(Box::new(obj));
        }
        CompiledFunction::new(py, prog)
    }
}
