//! `pytensor_llvm.interp`: compiled functions and Rust kernels for Python.

use super::function::{ArgType, Destination, Function as RFunction};
use super::layout::Layout;
use crate::abi::OutputMode;
use crate::runtime::PtCtx;
use crate::array::{OwnedArray, PtArrayRef};
use crate::dtype::DType;
use crate::kernel::Kernel;
use crate::linalg;
use numpy::npyffi::{NPY_ARRAY_WRITEABLE, NpyTypes, PY_ARRAY_API, npy_intp};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::PyClassInitializer;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};
use std::marker::PhantomData;
use std::sync::Arc;

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

fn dtype_of(d: &Bound<'_, PyArrayDescr>) -> PyResult<DType> {
    DType::from_numpy(d.kind(), d.itemsize())
        .ok_or_else(|| PyTypeError::new_err(format!("dtype {d} is not supported")))
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
        DType::Uint16 => numpy::dtype::<u16>(py),
        DType::Uint32 => numpy::dtype::<u32>(py),
        DType::Uint64 => numpy::dtype::<u64>(py),
    }
}

// ------------------------------------------------------------------ kernels

/// A Rust kernel: its address and its config (owned by this object).
///
/// Every kernel is made for the layouts of its inputs (`in_layouts`, None where
/// unknown) and the number of axes of its outputs (`out_ndims`), the last two arguments
/// of each constructor: it chooses the layouts of its outputs then. The planner chooses
/// whether each output is provided or allocated (the output modes, the first argument).
/// `outputs` says both.
#[pyclass(module = "pytensor_llvm.interp", subclass, frozen)]
pub struct RustKernel {
    address: usize,
    config: Box<dyn std::any::Any + Send + Sync>,
    config_ptr: usize,
    desc: String,
    /// the mode of each output
    modes: Vec<OutputMode>,
    /// the layout of each output (`Kernel::output_layouts`)
    layouts: Vec<Layout>,
}

impl RustKernel {
    /// A kernel written with `crate::kernel`: its own function, and itself as config.
    /// `modes`: what the kernel was made with.
    fn of<K: Kernel>(kernel: K, modes: Vec<OutputMode>, desc: String) -> PyResult<Self> {
        let layouts = kernel.output_layouts().to_vec();
        let n = kernel.arity().1;
        if modes.len() != n || layouts.len() != n {
            return err(format!(
                "{desc}: {} output modes and {} layouts for {n} outputs",
                modes.len(),
                layouts.len()
            ));
        }
        if modes.contains(&OutputMode::View) {
            return err(format!("{desc} can't produce a view"));
        }
        let config: Box<K> = Box::new(kernel);
        let config_ptr = &*config as *const K as usize;
        Ok(RustKernel {
            address: crate::kernel::address::<K>(),
            config,
            config_ptr,
            desc,
            modes,
            layouts,
        })
    }
}

/// The layouts of the outputs of a faer kernel (`linalg::faer_layouts`).
fn faer(in_layouts: &[Option<Layout>], out_ndims: &[usize]) -> Vec<Layout> {
    linalg::faer_layouts(out_ndims, in_layouts)
}

/// Exactly `N` output modes.
fn modes<const N: usize>(what: &str, modes: &[OutputMode]) -> PyResult<[OutputMode; N]> {
    modes
        .try_into()
        .or_else(|_| err(format!("{what}: {N} outputs, got {} modes", modes.len())))
}

#[pymethods]
impl RustKernel {
    #[getter]
    fn address(&self) -> usize {
        self.address
    }

    /// Pointer to the kernel's config, valid while this object lives.
    #[getter]
    fn config(&self) -> usize {
        let _ = &self.config;
        self.config_ptr
    }

    fn __repr__(&self) -> String {
        self.desc.clone()
    }

    /// How the kernel produces each output: `[(mode, layout)]`, mode "provided"
    /// (written into memory provided in that layout) or "allocated" (the kernel
    /// allocates it in that layout).
    fn outputs(&self) -> Vec<(&'static str, Layout)> {
        self.modes
            .iter()
            .zip(&self.layouts)
            .map(|(&mode, layout)| {
                let mode = if mode == OutputMode::Provided {
                    "provided"
                } else {
                    "allocated"
                };
                (mode, layout.clone())
            })
            .collect()
    }
}

/// `$make` with the type `$t` the float type of `dtype` (float64 or float32).
macro_rules! for_float {
    ($dtype:expr, $what:expr, $t:ident => $make:expr) => {
        match dtype_of(&$dtype)? {
            DType::Float64 => {
                type $t = f64;
                $make
            }
            DType::Float32 => {
                type $t = f32;
                $make
            }
            dt => return err(format!("{} of {dt:?}", $what)),
        }
    };
}

/// numpy's dot of vectors and matrices (faer).
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Dot;

#[pymethods]
impl Dot {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "Dot", T => RustKernel::of(
            linalg::Dot::<T> { parallel, mode, layouts, element: PhantomData },
            vec![mode],
            "Dot".into(),
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(Dot))
    }
}

/// Sort along `axis` (float64, float32, int64), NaNs last.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Sort;

#[pymethods]
impl Sort {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        axis: usize,
        stable: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("Sort(axis={axis})");
        let [ndim] = out_ndims[..] else {
            return err(format!("{desc}: one output, got {out_ndims:?}"));
        };
        let layout = in_layouts.first().and_then(|l| l.as_ref());
        fn sort<T: crate::sort::Sortable>(
            mode: OutputMode,
            axis: usize,
            stable: bool,
            layout: Option<&Layout>,
            ndim: usize,
            desc: String,
        ) -> PyResult<RustKernel> {
            RustKernel::of(
                crate::sort::Sort::<T>::new(mode, axis, stable, layout, ndim),
                vec![mode],
                desc,
            )
        }
        let k = match dtype_of(&dtype)? {
            DType::Float64 => sort::<f64>(mode, axis, stable, layout, ndim, desc)?,
            DType::Float32 => sort::<f32>(mode, axis, stable, layout, ndim, desc)?,
            DType::Int64 => sort::<i64>(mode, axis, stable, layout, ndim, desc)?,
            dt => return err(format!("Sort of {dt:?}")),
        };
        Ok(PyClassInitializer::from(k).add_subclass(Sort))
    }
}

/// Cholesky factor (faer), LAPACK potrf semantics: NaN if not positive definite.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Cholesky;

#[pymethods]
impl Cholesky {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        lower: bool,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("Cholesky(lower={lower})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "Cholesky", T => RustKernel::of(
            linalg::Cholesky::<T> { parallel, mode, layouts, lower, element: PhantomData },
            vec![mode],
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(Cholesky))
    }
}

/// Triangular solve (faer), LAPACK trtrs semantics: NaN if the diagonal has a zero.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct SolveTriangular;

#[pymethods]
impl SolveTriangular {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        lower: bool,
        unit_diagonal: bool,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("SolveTriangular(lower={lower}, unit_diagonal={unit_diagonal})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "SolveTriangular", T => RustKernel::of(
            linalg::SolveTriangular::<T> {
                parallel,
                mode,
                layouts,
                lower,
                unit_diagonal,
                element: PhantomData,
            },
            vec![mode],
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(SolveTriangular))
    }
}

/// Solve (faer), scipy.linalg.solve semantics: NaN if A is singular. `assume_a`:
/// "gen", "sym" or "pos".
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Solve;

#[pymethods]
impl Solve {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        assume_a: &str,
        lower: bool,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let assume = match assume_a {
            "gen" => linalg::Assume::General,
            "sym" => linalg::Assume::Symmetric { lower },
            "pos" => linalg::Assume::PositiveDefinite { lower },
            other => return err(format!("Solve: assume_a={other:?} is not supported")),
        };
        let desc = format!("Solve(assume_a={assume_a}, lower={lower})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "Solve", T => RustKernel::of(
            linalg::Solve::<T> { parallel, mode, layouts, assume, element: PhantomData },
            vec![mode],
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(Solve))
    }
}

/// Solve with a Cholesky factor (faer), LAPACK potrs semantics.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct CholeskySolve;

#[pymethods]
impl CholeskySolve {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        lower: bool,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("CholeskySolve(lower={lower})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "CholeskySolve", T => RustKernel::of(
            linalg::CholeskySolve::<T> { parallel, mode, layouts, lower, element: PhantomData },
            vec![mode],
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(CholeskySolve))
    }
}

/// Matrix inverse (faer LU): fails for a singular matrix, as numpy.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct MatrixInverse;

#[pymethods]
impl MatrixInverse {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "MatrixInverse", T => RustKernel::of(
            linalg::MatrixInverse::<T> { parallel, mode, layouts, element: PhantomData },
            vec![mode],
            "MatrixInverse".into(),
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(MatrixInverse))
    }
}

/// Determinant (faer LU).
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Det;

#[pymethods]
impl Det {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "Det", T => RustKernel::of(
            linalg::Det::<T> { parallel, mode, layouts, element: PhantomData },
            vec![mode],
            "Det".into(),
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(Det))
    }
}

/// Sign and log of the absolute determinant (faer LU).
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct SLogDet;

#[pymethods]
impl SLogDet {
    #[new]
    fn new(
        modes: Vec<OutputMode>,
        dtype: Bound<'_, PyArrayDescr>,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let layouts = faer(&in_layouts, &out_ndims);
        let fixed = self::modes::<2>("SLogDet", &modes)?;
        let k = for_float!(dtype, "SLogDet", T => RustKernel::of(
            linalg::SLogDet::<T> { parallel, modes: fixed, layouts, element: PhantomData },
            modes,
            "SLogDet".into(),
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(SLogDet))
    }
}

/// LU factorization as LAPACK getrf (faer): (LU, pivots), NaN if singular.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct LUFactor;

#[pymethods]
impl LUFactor {
    #[new]
    fn new(
        modes: Vec<OutputMode>,
        dtype: Bound<'_, PyArrayDescr>,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let layouts = faer(&in_layouts, &out_ndims);
        let fixed = self::modes::<2>("LUFactor", &modes)?;
        let k = for_float!(dtype, "LUFactor", T => RustKernel::of(
            linalg::LUFactor::<T> { parallel, modes: fixed, layouts, element: PhantomData },
            modes,
            "LUFactor".into(),
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(LUFactor))
    }
}

/// LU decomposition as scipy.linalg.lu (faer). `form`: "matrix" (P, L, U), "indices"
/// (p, L, U) or "permute_l" (PL, U).
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct LU;

#[pymethods]
impl LU {
    #[new]
    fn new(
        modes: Vec<OutputMode>,
        dtype: Bound<'_, PyArrayDescr>,
        form: &str,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let form = match form {
            "matrix" => linalg::LuForm::Matrix,
            "indices" => linalg::LuForm::Indices,
            "permute_l" => linalg::LuForm::PermuteL,
            other => return err(format!("LU: unknown form {other:?}")),
        };
        let desc = format!("LU({form:?})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "LU", T => RustKernel::of(
            linalg::LU::<T> { parallel, modes: modes.clone(), layouts, form, element: PhantomData },
            modes,
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(LU))
    }
}

/// The permutation of LAPACK pivots (int32 or int64), as int64.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct PivotToPermutations;

#[pymethods]
impl PivotToPermutations {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        inverse: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("PivotToPermutations(inverse={inverse})");
        // (a vector: there is only one order)
        let layouts = faer(&in_layouts, &out_ndims);
        fn make<I: crate::kernel::Element + TryInto<usize>>(
            mode: OutputMode,
            layouts: Vec<Layout>,
            inverse: bool,
            desc: String,
        ) -> PyResult<RustKernel> {
            RustKernel::of(
                linalg::PivotToPermutations::<I> {
                    mode,
                    layouts,
                    inverse,
                    element: PhantomData,
                },
                vec![mode],
                desc,
            )
        }
        let k = match dtype_of(&dtype)? {
            DType::Int32 => make::<i32>(mode, layouts, inverse, desc)?,
            DType::Int64 => make::<i64>(mode, layouts, inverse, desc)?,
            dt => return err(format!("PivotToPermutations of {dt:?}")),
        };
        Ok(PyClassInitializer::from(k).add_subclass(PivotToPermutations))
    }
}

/// Symmetric eigendecomposition (faer): eigenvalues ascending, and eigenvectors if
/// `vectors` (Eigh) or not (Eigvalsh).
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Eigh;

#[pymethods]
impl Eigh {
    #[new]
    fn new(
        modes: Vec<OutputMode>,
        dtype: Bound<'_, PyArrayDescr>,
        lower: bool,
        vectors: bool,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("Eigh(lower={lower}, vectors={vectors})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "Eigh", T => RustKernel::of(
            linalg::Eigh::<T> {
                parallel,
                modes: modes.clone(),
                layouts,
                lower,
                vectors,
                element: PhantomData,
            },
            modes,
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(Eigh))
    }
}

/// Singular value decomposition (faer), numpy.linalg.svd semantics.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct SVD;

#[pymethods]
impl SVD {
    #[new]
    fn new(
        modes: Vec<OutputMode>,
        dtype: Bound<'_, PyArrayDescr>,
        full_matrices: bool,
        compute_uv: bool,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("SVD(full_matrices={full_matrices}, compute_uv={compute_uv})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "SVD", T => RustKernel::of(
            linalg::Svd::<T> {
                parallel,
                modes: modes.clone(),
                layouts,
                full_matrices,
                compute_uv,
                element: PhantomData,
            },
            modes,
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(SVD))
    }
}

/// QR decomposition without pivoting (faer). `mode`: "full", "economic" or "r".
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct QR;

#[pymethods]
impl QR {
    #[new]
    fn new(
        modes: Vec<OutputMode>,
        dtype: Bound<'_, PyArrayDescr>,
        mode: &str,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let qr_mode = match mode {
            "full" => linalg::QrMode::Full,
            "economic" => linalg::QrMode::Economic,
            "r" => linalg::QrMode::R,
            other => return err(format!("QR: mode={other:?} is not supported")),
        };
        let desc = format!("QR(mode={mode})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "QR", T => RustKernel::of(
            linalg::Qr::<T> {
                parallel,
                modes: modes.clone(),
                layouts,
                mode: qr_mode,
                element: PhantomData,
            },
            modes,
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(QR))
    }
}

/// Pseudo-inverse (faer SVD, or symmetric eigendecomposition if `hermitian`), as
/// numpy.linalg.pinv.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct MatrixPinv;

#[pymethods]
impl MatrixPinv {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        hermitian: bool,
        parallel: bool,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("MatrixPinv(hermitian={hermitian})");
        let layouts = faer(&in_layouts, &out_ndims);
        let k = for_float!(dtype, "MatrixPinv", T => RustKernel::of(
            linalg::MatrixPinv::<T> { parallel, mode, layouts, hermitian, element: PhantomData },
            vec![mode],
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(MatrixPinv))
    }
}

// ------------------------------------------------------------------ programs

/// A numpy array of `array` (not copied: the numpy array holds its buffer).
fn wrap_output<'py>(py: Python<'py>, array: OwnedArray) -> PyResult<Bound<'py, PyAny>> {
    let dt = array.dtype();
    let (buf, data, shape, strides) = array.into_parts();
    // the capsule holds the buffer reference until numpy frees the array (its base)
    let capsule = PyCapsule::new_with_value(py, buf, c"pytensor_llvm.buffer")?;
    let mut dims: Vec<npy_intp> = shape.iter().map(|&s| s as npy_intp).collect();
    let mut strides: Vec<npy_intp> = strides.iter().map(|&s| s as npy_intp).collect();
    // SAFETY: a valid array of `dt` in the buffer the capsule keeps alive (the
    // invariant of `OwnedArray`); the descr and capsule references are stolen by the
    // numpy calls. numpy may write the array: the call has let go of the buffer, so
    // the only other holder there can be is another output of the call that is the
    // same memory.
    unsafe {
        let ty = numpy::npyffi::get_type_object(py, NpyTypes::PyArray_Type);
        let arr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            ty,
            descr(py, dt).into_ptr() as *mut _,
            dims.len() as i32,
            dims.as_mut_ptr(),
            strides.as_mut_ptr(),
            data as *mut _,
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

/// A compiled function: the generated kernel of a graph (`driver.py`), callable with
/// arrays, and from C (`c_function`).
#[pyclass(module = "pytensor_llvm.interp", frozen)]
pub struct Function {
    f: Arc<RFunction>,
    keepalive: Vec<Py<PyAny>>,
    /// the context of Python calls: one per function (one random state); concurrent
    /// calls from several threads take turns
    session: std::sync::Mutex<Option<PtCtx>>,
}

#[pymethods]
impl Function {
    /// `inputs`: `(dtype, shape, contiguous)` per graph input (shape: None where only
    /// known at run time; contiguous: the generated code assumes C-contiguous memory);
    /// `outputs`: `(dtype, shape, provided)` per graph output (provided: written into
    /// memory the caller provides, else returned in a buffer). `keepalive` is kept as
    /// long as the function: what `address` and `config` belong to.
    #[new]
    fn new(
        address: usize,
        config: usize,
        inputs: Vec<(Bound<'_, PyArrayDescr>, Vec<Option<u64>>, bool)>,
        outputs: Vec<(Bound<'_, PyArrayDescr>, Vec<Option<u64>>, bool)>,
        keepalive: Vec<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut in_types = vec![];
        let mut contiguous = vec![];
        for (d, shape, c) in &inputs {
            in_types.push(ArgType {
                dtype: dtype_of(d)?,
                shape: shape.clone(),
            });
            contiguous.push(*c);
        }
        let mut out_types = vec![];
        let mut provided = vec![];
        for (d, shape, p) in &outputs {
            out_types.push(ArgType {
                dtype: dtype_of(d)?,
                shape: shape.clone(),
            });
            provided.push(*p);
        }
        // SAFETY: what Python promises: `address` and `config` are those of a compiled
        // driver for these types (`driver.py`), and `keepalive` holds what they belong
        // to.
        let f = unsafe { RFunction::new(address, config, in_types, contiguous, out_types, provided) }
            .map_err(PyValueError::new_err)?;
        Ok(Function {
            f: Arc::new(f),
            keepalive,
            session: std::sync::Mutex::new(None),
        })
    }

    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyList>> {
        let f = &*self.f;
        if args.len() != f.n_inputs() {
            return err(format!(
                "expected {} inputs, got {}",
                f.n_inputs(),
                args.len()
            ));
        }
        let mut guard = self.lock_session(py);
        let ctx = guard.get_or_insert_with(PtCtx::new);
        // the arrays to call with, kept alive until the call is over, and their
        let mut keep: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
        // data, shapes and strides (per array: its data, and where its lengths start in
        // `dims`; its strides follow them)
        let mut parts: Vec<(usize, usize, usize)> = Vec::with_capacity(args.len());
        let mut dims: Vec<i64> = Vec::with_capacity(8 * args.len());
        let mut work = 0usize;
        let res = (|| -> PyResult<Vec<OwnedArray>> {
            for (k, a) in args.iter().enumerate() {
                let want = f.inputs[k].dtype;
                let exact = match a.cast::<PyUntypedArray>() {
                    Ok(arr) => {
                        let d = arr.dtype();
                        DType::from_numpy(d.kind(), d.itemsize()) == Some(want)
                            && d.is_native_byteorder() != Some(false)
                    }
                    Err(_) => false,
                };
                let a = if exact {
                    a
                } else {
                    let kw = pyo3::types::PyDict::new(py);
                    kw.set_item("dtype", descr(py, want))?;
                    py.import("numpy")?
                        .call_method("asarray", (a,), Some(&kw))?
                };
                let arr = a.cast::<PyUntypedArray>()?;
                // SAFETY: reading the flags of a live array.
                let flags = unsafe { (*arr.as_array_ptr()).fields.flags };
                let aligned = flags & numpy::npyffi::NPY_ARRAY_ALIGNED != 0;
                let a = if !aligned {
                    py.import("numpy")?
                        .call_method1("ascontiguousarray", (a,))?
                        .call_method0("copy")?
                } else if f.contiguous_inputs[k]
                    && flags & numpy::npyffi::NPY_ARRAY_C_CONTIGUOUS == 0
                {
                    py.import("numpy")?
                        .call_method1("ascontiguousarray", (a,))?
                } else {
                    a
                };
                let arr = a.cast::<PyUntypedArray>()?;
                // SAFETY: reading the data pointer of a live array.
                let data = unsafe { (*arr.as_array_ptr()).fields.data as usize };
                parts.push((data, dims.len(), arr.ndim()));
                dims.extend(arr.shape().iter().map(|&n| n as i64));
                dims.extend(arr.strides().iter().map(|&s| s as i64));
                work += arr.len();
                keep.push(a);
            }
            let run = move |ctx: &mut PtCtx| -> Result<Vec<OwnedArray>, String> {
                let mut ins = Vec::with_capacity(parts.len());
                for (k, &(data, start, ndim)) in parts.iter().enumerate() {
                    let (shape, strides) = dims[start..start + 2 * ndim].split_at(ndim);
                    // SAFETY: the memory of a numpy array of the input's dtype
                    // (converted above), which `keep` keeps alive until the call is
                    // over. Nothing here writes it; as everywhere with numpy, that no
                    // other thread does meanwhile is up to the user.
                    ins.push(
                        unsafe {
                            PtArrayRef::checked(data as *const u8, shape, strides, f.inputs[k].dtype)
                        }
                        .map_err(|e| format!("input {k}: {e}"))?,
                    );
                }
                let dests = (0..f.n_outputs()).map(|_| Destination::New).collect();
                Ok(f.call(ctx, &ins, dests)?.into_iter().flatten().collect())
            };
            // releasing the GIL costs ~100ns: only worth it with real work
            let r = if work > 20_000 {
                py.detach(move || run(ctx))
            } else {
                run(ctx)
            };
            r.map_err(runtime_error)
        })();
        drop(keep);
        drop(guard);
        let arrays: Vec<Bound<'_, PyAny>> = res?
            .into_iter()
            .map(|a| wrap_output(py, a))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, arrays)?.unbind())
    }

    /// Restart the random state of Python calls from `seed` (draws are then
    /// reproducible). The C API seeds each of its sessions (`seed` in the table).
    fn seed(&self, py: Python<'_>, seed: u64) {
        self.lock_session(py).get_or_insert_with(PtCtx::new).seed(seed);
    }

    /// Allow or forbid parallel loops for Python calls (results are the same either
    /// way). The C API sets this per session (`set_parallel` in the table).
    fn set_parallel(&self, py: Python<'_>, on: bool) {
        self.lock_session(py).get_or_insert_with(PtCtx::new).parallel = on;
    }

    /// A new reference to this function for the C API, as the address of a
    /// `PtFunction` (release it with `function_decref` from `c_api_table()`). It keeps
    /// the compiled code alive on its own.
    fn c_function(&self, py: Python<'_>) -> usize {
        let keep = self.keepalive.iter().map(|o| o.clone_ref(py)).collect();
        super::capi::Function::new(self.f.clone(), keep).into_handle() as usize
    }

    /// The address of the `PtFunctionApi` table for functions from `c_function()`.
    #[staticmethod]
    fn c_api_table() -> usize {
        &super::capi::API as *const _ as usize
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OutputMode>()?;
    m.add_class::<RustKernel>()?;
    m.add_class::<Dot>()?;
    m.add_class::<Sort>()?;
    m.add_class::<Cholesky>()?;
    m.add_class::<SolveTriangular>()?;
    m.add_class::<Solve>()?;
    m.add_class::<CholeskySolve>()?;
    m.add_class::<MatrixInverse>()?;
    m.add_class::<Det>()?;
    m.add_class::<SLogDet>()?;
    m.add_class::<LUFactor>()?;
    m.add_class::<LU>()?;
    m.add_class::<PivotToPermutations>()?;
    m.add_class::<Eigh>()?;
    m.add_class::<SVD>()?;
    m.add_class::<QR>()?;
    m.add_class::<MatrixPinv>()?;
    m.add_class::<Function>()?;
    m.add_function(wrap_pyfunction!(c_api_table, m)?)?;
    m.add_function(wrap_pyfunction!(numba_entry, m)?)?;
    Ok(())
}

/// The address of the kernel that runs a numba kernel: the task's config is the address
/// of the numba cfunc (`kernel::numba_entry`).
#[pyfunction]
fn numba_entry() -> usize {
    crate::kernel::numba_entry as *const () as usize
}

/// The address of the `PtFunctionApi` table for functions from `Function.c_function()`
/// (header: `pytensor_llvm.c_header()`).
#[pyfunction]
fn c_api_table() -> usize {
    &super::capi::API as *const _ as usize
}

impl Function {
    /// The session, waiting without the GIL if another thread has it.
    fn lock_session(&self, py: Python<'_>) -> std::sync::MutexGuard<'_, Option<PtCtx>> {
        loop {
            match self.session.try_lock() {
                Ok(g) => return g,
                Err(std::sync::TryLockError::Poisoned(p)) => return p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => py.detach(|| drop(self.session.lock())),
            }
        }
    }
}

/// Choose the number of threads parallel loops use (process-wide; only before the
/// first parallel loop runs). Default: `PYTENSOR_LLVM_THREADS`, else one per CPU.
#[pyfunction]
pub fn set_threads(n: usize) -> PyResult<()> {
    crate::runtime::runtime()
        .set_threads(n)
        .map_err(PyRuntimeError::new_err)
}

/// The number of threads parallel loops use.
#[pyfunction]
pub fn threads() -> usize {
    crate::runtime::runtime().threads()
}
