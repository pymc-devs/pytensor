//! Python API of the new backend: `pytensor_llvm.v2`.
//!
//! Ops are typed objects (`Elemwise(scalar=Add())`, `Dot()`, ...); a `Graph` is built
//! from them and compiled into a `Function`.

use super::graph::{ConstData, Graph as RGraph, ValueId};
use super::index::{Idx, IndexVec};
use super::interp::{Options, Program};
use super::op::TensorOp as RTensorOp;
use super::ops;
use super::scalar::{self as sc, ScalarIn, SRef};
use super::types::{Axis, AxisLen, InputIdx, OutputIdx, TensorTy};
use crate::abi::PtBuffer;
use crate::dtype::DType;
use numpy::npyffi::{npy_intp, NpyTypes, NPY_ARRAY_WRITEABLE, PY_ARRAY_API};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyClassInitializer;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};
use std::sync::Arc;

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

fn dtype_of(d: &Bound<'_, PyArrayDescr>) -> PyResult<DType> {
    DType::from_numpy(d.kind(), d.itemsize()).ok_or_else(|| PyTypeError::new_err(format!("dtype {d} is not supported")))
}

fn descr<'py>(py: Python<'py>, dt: DType) -> Bound<'py, PyArrayDescr> {
    match dt {
        DType::Float64 => numpy::dtype::<f64>(py),
        DType::Float32 => numpy::dtype::<f32>(py),
        DType::Int64 => numpy::dtype::<i64>(py),
        DType::Int32 => numpy::dtype::<i32>(py),
        DType::Int16 => numpy::dtype::<i16>(py),
        DType::Int8 => numpy::dtype::<i8>(py),
        DType::Uint8 => numpy::dtype::<u8>(py),
        DType::Bool => numpy::dtype::<bool>(py),
    }
}

fn tensor_ty(dtype: &Bound<'_, PyArrayDescr>, shape: Vec<Option<u64>>) -> PyResult<TensorTy> {
    Ok(TensorTy {
        dtype: dtype_of(dtype)?,
        shape: shape.into_iter().map(|s| s.map_or(AxisLen::Dynamic, AxisLen::Static)).collect(),
        layout: None,
    })
}

// ------------------------------------------------------------------ scalar ops

/// Base class of all scalar ops.
#[pyclass(module = "pytensor_llvm.v2", subclass, frozen)]
pub struct ScalarOp {
    op: Arc<dyn sc::ScalarOp>,
}

#[pymethods]
impl ScalarOp {
    fn __repr__(&self) -> String {
        format!("{:?}", self.op)
    }
}

macro_rules! py_scalar_ops {
    ($($name:ident),* $(,)?) => {
        mod scalar_classes {
            use super::*;
            $(
                #[pyclass(module = "pytensor_llvm.v2", extends = ScalarOp, frozen)]
                pub struct $name;

                #[pymethods]
                impl $name {
                    #[new]
                    fn new() -> PyClassInitializer<Self> {
                        PyClassInitializer::from(ScalarOp { op: Arc::new(sc::$name) }).add_subclass($name)
                    }
                }
            )*

            pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
                $(m.add_class::<$name>()?;)*
                Ok(())
            }
        }
    };
}

py_scalar_ops!(
    Add, Mul, Sub, TrueDiv, IntDiv, Mod, Pow, Maximum, Minimum, Clip, Neg, Abs, Sign, Sqr, Reciprocal,
    Exp, Log, Sqrt, Sin, Cos, Floor, Ceil, Trunc, RoundHalfToEven, RoundHalfAwayFromZero,
    Exp2, Log2, Log10, Expm1, Log1p, Tan, Tanh, Sinh, Cosh, ArcSin, ArcCos, ArcTan, ArcSinh, ArcCosh,
    ArcTanh, ArcTan2, Erf, Erfc, GammaLn, Gamma, Psi, Sigmoid, Softplus, Log1mexp,
    LT, GT, LE, GE, EQ, NEQ, IsNan, IsInf, Switch, AND, OR, XOR, Invert, Identity, Cast, Second,
);

/// A value in a `ScalarGraph`.
#[pyclass(module = "pytensor_llvm.v2", frozen, from_py_object)]
#[derive(Clone)]
pub struct ScalarValue {
    r: SRef,
    graph: usize,
}

/// Builder for a Composite's scalar graph.
#[pyclass(module = "pytensor_llvm.v2")]
pub struct ScalarGraph {
    g: Option<sc::Composite>,
}

impl ScalarGraph {
    fn id(&self) -> usize {
        self as *const _ as usize
    }

    fn graph(&mut self) -> PyResult<&mut sc::Composite> {
        self.g.as_mut().ok_or_else(|| PyValueError::new_err("this ScalarGraph was already used"))
    }

    fn check(&self, v: &ScalarValue) -> PyResult<SRef> {
        if v.graph != self.id() && !matches!(v.r, SRef::Const { .. }) {
            return err("ScalarValue belongs to a different ScalarGraph");
        }
        Ok(v.r)
    }
}

#[pymethods]
impl ScalarGraph {
    #[new]
    fn new(input_dtypes: Vec<Bound<'_, PyArrayDescr>>) -> PyResult<Self> {
        let inputs = input_dtypes.iter().map(dtype_of).collect::<PyResult<Vec<_>>>()?;
        Ok(ScalarGraph { g: Some(sc::Composite { inputs: IndexVec::from_vec(inputs), ..Default::default() }) })
    }

    fn input(&mut self, i: usize) -> PyResult<ScalarValue> {
        let id = self.id();
        let g = self.graph()?;
        if i >= g.inputs.len() {
            return err(format!("scalar input {i} out of range"));
        }
        Ok(ScalarValue { r: SRef::Input(ScalarIn::new(i)), graph: id })
    }

    fn constant(&self, value: f64, dtype: Bound<'_, PyArrayDescr>) -> PyResult<ScalarValue> {
        let dt = dtype_of(&dtype)?;
        let i = if value.is_finite() { value as i64 } else { 0 };
        Ok(ScalarValue { r: SRef::Const { dtype: dt, f: value, i }, graph: self.id() })
    }

    fn apply(
        &mut self,
        op: PyRef<'_, ScalarOp>,
        inputs: Vec<ScalarValue>,
        output_dtypes: Vec<Bound<'_, PyArrayDescr>>,
    ) -> PyResult<Vec<ScalarValue>> {
        let refs = inputs.iter().map(|v| self.check(v)).collect::<PyResult<Vec<_>>>()?;
        let outs = output_dtypes.iter().map(dtype_of).collect::<PyResult<Vec<_>>>()?;
        let n_out = outs.len();
        let id = self.id();
        let node = self.graph()?.add(op.op.clone(), refs, outs).map_err(PyValueError::new_err)?;
        Ok((0..n_out).map(|k| ScalarValue { r: SRef::Node(node, k as u32), graph: id }).collect())
    }

    fn set_outputs(&mut self, outputs: Vec<ScalarValue>) -> PyResult<()> {
        let refs = outputs.iter().map(|v| self.check(v)).collect::<PyResult<Vec<_>>>()?;
        if refs.is_empty() {
            return err("a scalar graph needs outputs");
        }
        self.graph()?.outputs = refs;
        Ok(())
    }
}

/// PyTensor's Composite, from a finished `ScalarGraph`.
#[pyclass(module = "pytensor_llvm.v2", extends = ScalarOp, frozen)]
pub struct Composite;

#[pymethods]
impl Composite {
    #[new]
    fn new(mut graph: PyRefMut<'_, ScalarGraph>) -> PyResult<PyClassInitializer<Self>> {
        let g = graph.g.take().ok_or_else(|| PyValueError::new_err("this ScalarGraph was already used"))?;
        if g.outputs.is_empty() {
            return err("the ScalarGraph has no outputs");
        }
        Ok(PyClassInitializer::from(ScalarOp { op: Arc::new(g) }).add_subclass(Composite))
    }
}

// ------------------------------------------------------------------ tensor ops

/// Base class of all tensor ops.
#[pyclass(module = "pytensor_llvm.v2", subclass, frozen)]
pub struct TensorOp {
    op: Arc<dyn RTensorOp>,
}

#[pymethods]
impl TensorOp {
    fn __repr__(&self) -> String {
        format!("{:?}", self.op)
    }
}

fn base(op: impl RTensorOp + 'static) -> TensorOp {
    TensorOp { op: Arc::new(op) }
}

#[pyclass(module = "pytensor_llvm.v2", extends = TensorOp, frozen)]
pub struct Elemwise;

#[pymethods]
impl Elemwise {
    #[new]
    fn new(scalar: PyRef<'_, ScalarOp>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(base(ops::elemwise::Elemwise { scalar: scalar.op.clone() })).add_subclass(Elemwise)
    }
}

#[pyclass(module = "pytensor_llvm.v2", extends = TensorOp, frozen)]
pub struct CAReduce;

#[pymethods]
impl CAReduce {
    /// `identity` is the start value, as (float value, integer value).
    #[new]
    fn new(scalar: PyRef<'_, ScalarOp>, axes: Vec<u32>, identity: (f64, i64), acc_dtype: Bound<'_, PyArrayDescr>) -> PyResult<PyClassInitializer<Self>> {
        let op = ops::elemwise::CAReduce {
            scalar: scalar.op.clone(),
            axes: axes.into_iter().map(Axis).collect(),
            identity,
            acc: dtype_of(&acc_dtype)?,
        };
        Ok(PyClassInitializer::from(base(op)).add_subclass(CAReduce))
    }
}

#[pyclass(module = "pytensor_llvm.v2", extends = TensorOp, frozen)]
pub struct DimShuffle;

#[pymethods]
impl DimShuffle {
    /// `order[k]`: input axis of output axis k, or None for a new axis.
    #[new]
    fn new(order: Vec<Option<u32>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(base(ops::shape::DimShuffle { order: order.into_iter().map(|a| a.map(Axis)).collect() })).add_subclass(DimShuffle)
    }
}

#[pyclass(module = "pytensor_llvm.v2", extends = TensorOp, frozen)]
pub struct DeepCopy;

#[pymethods]
impl DeepCopy {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(base(ops::shape::DeepCopy)).add_subclass(DeepCopy)
    }
}

#[pyclass(module = "pytensor_llvm.v2", extends = TensorOp, frozen)]
pub struct Dot;

#[pymethods]
impl Dot {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(base(ops::blas::Dot)).add_subclass(Dot)
    }
}

/// A kernel compiled elsewhere (numba fallback) with the old kernel ABI.
#[pyclass(module = "pytensor_llvm.v2", extends = TensorOp, frozen)]
pub struct Foreign;

#[pymethods]
impl Foreign {
    #[new]
    fn new(address: usize, description: String, keepalive: Py<PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(base(ops::foreign::Foreign { addr: address, desc: description, keepalive: Some(keepalive) })).add_subclass(Foreign)
    }
}

#[pyclass(module = "pytensor_llvm.v2", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    Compiled,
    Interpreted,
}

#[pyclass(module = "pytensor_llvm.v2", extends = TensorOp, frozen)]
pub struct OpFromGraph;

#[pymethods]
impl OpFromGraph {
    #[new]
    #[pyo3(signature = (graph, kind, fastmath = true, cache = true, opt_level = 3))]
    fn new(mut graph: PyRefMut<'_, Graph>, kind: UnitKind, fastmath: bool, cache: bool, opt_level: u32) -> PyResult<PyClassInitializer<Self>> {
        let g = graph.take()?;
        let kind = match kind {
            UnitKind::Compiled => ops::ofg::UnitKind::Compiled,
            UnitKind::Interpreted => ops::ofg::UnitKind::Interpreted,
        };
        let opts = Options { fastmath, cache, opt_level };
        Ok(PyClassInitializer::from(base(ops::ofg::OpFromGraph { graph: Arc::new(g), kind, opts })).add_subclass(OpFromGraph))
    }
}

// ------------------------------------------------------------------ graphs

/// A variable of a `Graph`.
#[pyclass(module = "pytensor_llvm.v2", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct Value {
    id: ValueId,
    graph: usize,
}

#[pymethods]
impl Value {
    fn __repr__(&self) -> String {
        format!("Value({})", self.id.0)
    }
}

#[pyclass(module = "pytensor_llvm.v2")]
pub struct Graph {
    g: Option<RGraph>,
    has_outputs: bool,
}

impl Graph {
    fn id(&self) -> usize {
        self as *const _ as usize
    }

    fn graph(&mut self) -> PyResult<&mut RGraph> {
        self.g.as_mut().ok_or_else(|| PyValueError::new_err("this Graph was already used"))
    }

    fn value(&self, v: &Value) -> PyResult<ValueId> {
        if v.graph != self.id() {
            return err("Value belongs to a different Graph");
        }
        Ok(v.id)
    }

    fn take(&mut self) -> PyResult<RGraph> {
        if !self.has_outputs {
            return err("set the graph's outputs first");
        }
        self.g.take().ok_or_else(|| PyValueError::new_err("this Graph was already used"))
    }
}

#[pymethods]
impl Graph {
    #[new]
    fn new() -> Self {
        Graph { g: Some(RGraph::default()), has_outputs: false }
    }

    /// A graph input of the given dtype and static shape (None: unknown length).
    fn add_input(&mut self, dtype: Bound<'_, PyArrayDescr>, shape: Vec<Option<u64>>) -> PyResult<Value> {
        let ty = tensor_ty(&dtype, shape)?;
        let id = self.id();
        Ok(Value { id: self.graph()?.add_input(ty), graph: id })
    }

    fn add_constant(&mut self, array: &Bound<'_, PyUntypedArray>) -> PyResult<Value> {
        let dt = dtype_of(&array.dtype())?;
        let np = array.py().import("numpy")?;
        let native = np.call_method1("ascontiguousarray", (array,))?;
        let native = native.call_method1("astype", (array.dtype().call_method1("newbyteorder", ("=",))?,))?;
        let bytes: Vec<u8> = native.call_method0("tobytes")?.extract()?;
        let shape: Vec<i64> = array.shape().iter().map(|&n| n as i64).collect();
        let data = ConstData::new(dt, shape, &bytes).map_err(PyValueError::new_err)?;
        let id = self.id();
        Ok(Value { id: self.graph()?.add_const(Arc::new(data)), graph: id })
    }

    /// Add a node. `output_types` are (dtype, static shape); `view_map` lists
    /// (output, input) pairs that may share memory (PyTensor's view_map).
    #[pyo3(signature = (op, inputs, output_types, view_map = vec![]))]
    fn add_node(
        &mut self,
        op: PyRef<'_, TensorOp>,
        inputs: Vec<Value>,
        output_types: Vec<(Bound<'_, PyArrayDescr>, Vec<Option<u64>>)>,
        view_map: Vec<(u32, u32)>,
    ) -> PyResult<Vec<Value>> {
        let ins = inputs.iter().map(|v| self.value(v)).collect::<PyResult<Vec<_>>>()?;
        let tys = output_types.into_iter().map(|(d, s)| tensor_ty(&d, s)).collect::<PyResult<Vec<_>>>()?;
        let views = view_map.into_iter().map(|(o, i)| (OutputIdx(o), InputIdx(i))).collect();
        let id = self.id();
        let outs = self.graph()?.add_node(op.op.clone(), ins, tys, views).map_err(PyValueError::new_err)?;
        Ok(outs.into_iter().map(|v| Value { id: v, graph: id }).collect())
    }

    fn set_outputs(&mut self, outputs: Vec<Value>) -> PyResult<()> {
        let outs = outputs.iter().map(|v| self.value(v)).collect::<PyResult<Vec<_>>>()?;
        self.graph()?.outputs = IndexVec::from_vec(outs);
        self.has_outputs = true;
        Ok(())
    }
}

// ------------------------------------------------------------------ functions

struct SendPtr(*mut PtBuffer);
// SAFETY: moves one buffer reference into a capsule destructor; refcounts are atomic.
unsafe impl Send for SendPtr {}

fn wrap_output<'py>(py: Python<'py>, dt: DType, o: &super::interp::OutArray) -> PyResult<Bound<'py, PyAny>> {
    let owner = SendPtr(o.owner);
    // SAFETY: the capsule holds the one reference of `o`; numpy calls the destructor
    // once, when the array (its base) is freed.
    let capsule = PyCapsule::new_with_value_and_destructor(py, owner, c"pytensor_llvm.buffer", |b: SendPtr, _| unsafe {
        PtBuffer::decref(b.0)
    })?;
    let mut dims: Vec<npy_intp> = o.shape.iter().map(|&s| s as npy_intp).collect();
    let mut strides: Vec<npy_intp> = o.strides.iter().map(|&s| s as npy_intp).collect();
    // SAFETY: `o` describes a valid array in the buffer the capsule keeps alive; the
    // descr and capsule references are stolen by the numpy calls (also on failure).
    unsafe {
        let ty = numpy::npyffi::get_type_object(py, NpyTypes::PyArray_Type);
        let arr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            ty,
            descr(py, dt).into_ptr() as *mut _,
            dims.len() as i32,
            dims.as_mut_ptr(),
            strides.as_mut_ptr(),
            o.data as *mut _,
            NPY_ARRAY_WRITEABLE,
            std::ptr::null_mut(),
        );
        let arr = Bound::from_owned_ptr_or_err(py, arr)?;
        if PY_ARRAY_API.PyArray_SetBaseObject(py, arr.as_ptr() as *mut _, capsule.into_ptr()) != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(arr)
    }
}

fn runtime_error(msg: String) -> PyErr {
    if msg.contains("out of bounds") {
        PyIndexError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

#[pyclass(module = "pytensor_llvm.v2", frozen)]
pub struct Function {
    prog: Arc<Program>,
    out_dtypes: Vec<DType>,
    in_dtypes: Vec<DType>,
    numpy: Py<PyModule>,
}

/// Compile a graph into a callable.
#[pyfunction]
#[pyo3(signature = (graph, fastmath = true, cache = true, opt_level = 3))]
fn compile(py: Python<'_>, mut graph: PyRefMut<'_, Graph>, fastmath: bool, cache: bool, opt_level: u32) -> PyResult<Function> {
    let g = graph.take()?;
    let in_dtypes = g.inputs.iter().map(|&v| g.ty(v).dtype).collect();
    let out_dtypes = g.outputs.iter().map(|&v| g.ty(v).dtype).collect();
    let opts = Options { fastmath, cache, opt_level };
    let prog = py.detach(|| Program::build(Arc::new(g), &opts)).map_err(PyValueError::new_err)?;
    Ok(Function { prog: Arc::new(prog), out_dtypes, in_dtypes, numpy: py.import("numpy")?.unbind() })
}

#[pymethods]
impl Function {
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyList>> {
        let prog = &*self.prog;
        if args.len() != prog.n_inputs() {
            return err(format!("expected {} inputs, got {}", prog.n_inputs(), args.len()));
        }
        let np = self.numpy.bind(py);
        let mut st = prog.take_state();
        let mut keep = Vec::with_capacity(args.len());
        let mut work = 0usize;
        let res = (|| -> PyResult<Vec<super::interp::OutArray>> {
            for (k, a) in args.iter().enumerate() {
                // kernels read native-endian, aligned data of exactly the input's dtype
                let exact = match a.cast::<PyUntypedArray>() {
                    Ok(arr) => {
                        let d = arr.dtype();
                        DType::from_numpy(d.kind(), d.itemsize()) == Some(self.in_dtypes[k]) && d.is_native_byteorder() != Some(false)
                    }
                    Err(_) => false,
                };
                let a = if exact {
                    a
                } else {
                    let want = descr(py, self.in_dtypes[k]);
                    np.call_method("asarray", (a,), Some(&[("dtype", want)].into_py_dict(py)?))?
                };
                let arr = a.cast::<PyUntypedArray>()?;
                // SAFETY: reading the flags of a live array.
                let aligned = unsafe { (*arr.as_array_ptr()).fields.flags } & numpy::npyffi::NPY_ARRAY_ALIGNED != 0;
                let a = if !aligned { np.call_method1("ascontiguousarray", (a,))?.call_method0("copy")? } else { a };
                let arr = a.cast::<PyUntypedArray>()?;
                // SAFETY: the data pointer of a live array, kept alive by `keep` until
                // the call is finished.
                let data = unsafe { (*arr.as_array_ptr()).fields.data as *mut u8 };
                let shape: Vec<i64> = arr.shape().iter().map(|&n| n as i64).collect();
                let strides: Vec<i64> = arr.strides().iter().map(|&s| s as i64).collect();
                prog.set_input(&mut st, InputIdx::new(k), data, &shape, &strides).map_err(PyValueError::new_err)?;
                work += arr.len();
                keep.push(a);
            }
            let mut ctx = std::mem::take(&mut st.ctx);
            let run = |st: &mut super::interp::CallState, ctx: &mut crate::abi::PtCtx| -> Result<Vec<super::interp::OutArray>, String> {
                let none: IndexVec<OutputIdx, Option<*mut u8>> = prog.sig().outputs.indices().map(|_| None).collect();
                prog.execute(st, ctx, &none)?;
                let outs = prog.sig().outputs.indices().map(|o| prog.take_output(st, ctx, o)).collect::<Result<Vec<_>, _>>();
                prog.finish(st);
                outs
            };
            // Releasing the GIL costs ~100ns; only worth it when there's real work.
            let r = if work > 20_000 {
                struct S<'a>(&'a mut super::interp::CallState, &'a mut crate::abi::PtCtx);
                // SAFETY: the state and context are exclusively ours for this call.
                unsafe impl Send for S<'_> {}
                let s = S(&mut st, &mut ctx);
                py.detach(move || {
                    let s = s;
                    run(s.0, s.1)
                })
            } else {
                run(&mut st, &mut ctx)
            };
            st.ctx = ctx;
            r.map_err(runtime_error)
        })();
        drop(keep);
        prog.put_state(st);
        let outs = res?;
        let arrays: Vec<Bound<'_, PyAny>> =
            outs.iter().zip(&self.out_dtypes).map(|(o, &dt)| wrap_output(py, dt, o)).collect::<PyResult<_>>()?;
        Ok(PyList::new(py, arrays)?.unbind())
    }

    /// Compile statistics and the step list.
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let s = &self.prog.stats;
        d.set_item("n_steps", self.prog.n_steps())?;
        d.set_item("n_units", s.n_units)?;
        d.set_item("n_compiled", s.n_compiled)?;
        d.set_item("n_disk_cache", s.n_disk_cache)?;
        d.set_item("n_in_process", s.n_in_process)?;
        d.set_item("llvm_cpu_s", s.llvm_cpu_s)?;
        d.set_item("compile_wall_s", s.wall_s)?;
        d.set_item("build_s", s.build_s)?;
        d.set_item("arena_bytes", self.prog.arena_bytes())?;
        d.set_item("steps", self.prog.describe_steps())?;
        Ok(d)
    }
}

use pyo3::types::IntoPyDict;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ScalarOp>()?;
    scalar_classes::register(m)?;
    m.add_class::<ScalarValue>()?;
    m.add_class::<ScalarGraph>()?;
    m.add_class::<Composite>()?;
    m.add_class::<TensorOp>()?;
    m.add_class::<Elemwise>()?;
    m.add_class::<CAReduce>()?;
    m.add_class::<DimShuffle>()?;
    m.add_class::<DeepCopy>()?;
    m.add_class::<Dot>()?;
    m.add_class::<Foreign>()?;
    m.add_class::<UnitKind>()?;
    m.add_class::<OpFromGraph>()?;
    m.add_class::<Value>()?;
    m.add_class::<Graph>()?;
    m.add_class::<Function>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}
