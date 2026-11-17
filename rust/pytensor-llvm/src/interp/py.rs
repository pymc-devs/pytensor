//! `pytensor_llvm.interp`: programs and Rust kernels for Python.

use super::consts::{ConstBuffers, ConstData};
use super::kernels;
use super::layout::Layout;
use super::program::{
    KernelHandle, Program as RProgram, ProgramSpec, Task, ValueId, ValueKind, ValueSpec,
};
use crate::abi::OutputMode;
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
#[pyclass(module = "pytensor_llvm.interp", subclass, frozen)]
pub struct RustKernel {
    address: usize,
    config: Box<dyn std::any::Any + Send + Sync>,
    config_ptr: usize,
    desc: String,
    /// what the kernel produces (see `outputs`)
    produces: Produces,
    /// programs the kernel runs (Scan's loop body), for profiling
    programs: Vec<Py<Program>>,
}

/// How a kernel produces its outputs, declared next to the kernel (the Python planner
/// asks `RustKernel.outputs`).
pub enum Produces {
    /// new memory per output: provided (written as it is given, in the layout the
    /// planner provides: C) or allocated C-contiguous by the kernel
    Fresh(Vec<OutputMode>),
    /// one view of input 0, its axes rearranged (DimShuffle, also identity views)
    DimShuffle(Vec<Option<usize>>),
}

impl RustKernel {
    /// A kernel written with `crate::kernel`: its own function, and itself as config.
    fn of<K: Kernel>(kernel: K, desc: String, produces: Produces) -> Self {
        RustKernel::new(crate::kernel::address::<K>(), kernel, desc, produces)
    }

    /// One that writes a single output, in new memory.
    fn writing<K: Kernel>(kernel: K, mode: OutputMode, desc: String) -> PyResult<Self> {
        RustKernel::writing_all(kernel, vec![mode], desc)
    }

    /// One that writes all its outputs, in new memory.
    fn writing_all<K: Kernel>(kernel: K, modes: Vec<OutputMode>, desc: String) -> PyResult<Self> {
        if modes.contains(&OutputMode::View) {
            return err(format!("{desc} can't produce a view"));
        }
        if modes.len() != kernel.arity().1 {
            return err(format!(
                "{desc}: {} output modes for {} outputs",
                modes.len(),
                kernel.arity().1
            ));
        }
        Ok(RustKernel::of(kernel, desc, Produces::Fresh(modes)))
    }

    fn new<T: Send + Sync + 'static>(
        f: usize,
        config: T,
        desc: String,
        produces: Produces,
    ) -> Self {
        let config: Box<T> = Box::new(config);
        let config_ptr = &*config as *const T as usize;
        RustKernel {
            address: f,
            config,
            config_ptr,
            desc,
            produces,
            programs: vec![],
        }
    }
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

    /// How each output comes to be, given the inputs' layouts (`(order, strides)`, None
    /// where unknown): `[(mode, view_of, layout)]` with mode "provided", "allocated" or
    /// "view" (the Python side's `llvm_outputs`, for this kernel).
    /// `out_ndims`: the number of axes of each output.
    fn outputs(
        &self,
        in_layouts: Vec<Option<Layout>>,
        out_ndims: Vec<usize>,
    ) -> PyResult<Vec<(&'static str, Option<usize>, Layout)>> {
        match &self.produces {
            Produces::Fresh(modes) => {
                if modes.len() != out_ndims.len() {
                    return err(format!(
                        "{}: {} outputs, got {} ndims",
                        self.desc,
                        modes.len(),
                        out_ndims.len()
                    ));
                }
                Ok(modes
                    .iter()
                    .zip(&out_ndims)
                    .map(|(m, &nd)| {
                        (
                            if *m == OutputMode::Provided {
                                "provided"
                            } else {
                                "allocated"
                            },
                            None,
                            Layout::c(nd),
                        )
                    })
                    .collect())
            }
            Produces::DimShuffle(order) => {
                let x = match in_layouts.first() {
                    Some(Some(l)) => l.clone(),
                    _ => {
                        let nd = order.iter().flatten().map(|a| a + 1).max().unwrap_or(0);
                        Layout::dynamic(nd)
                    }
                };
                Ok(vec![("view", Some(0), x.dimshuffle(order))])
            }
        }
    }

    /// Programs this kernel runs (e.g. Scan's loop body).
    #[getter]
    fn programs(&self, py: Python<'_>) -> Vec<Py<Program>> {
        self.programs.iter().map(|p| p.clone_ref(py)).collect()
    }
}

#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct DimShuffle;

#[pymethods]
impl DimShuffle {
    /// `order[k]`: input axis of output axis k, or None for a new axis. Its output is a
    /// view of the input.
    #[new]
    fn new(order: Vec<Option<usize>>) -> PyClassInitializer<Self> {
        let desc = format!("DimShuffle{order:?}");
        let produces = Produces::DimShuffle(order.clone());
        let k = RustKernel::of(kernels::DimShuffle { order }, desc, produces);
        PyClassInitializer::from(k).add_subclass(DimShuffle)
    }
}

#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Copy;

#[pymethods]
impl Copy {
    #[new]
    fn new(mode: OutputMode, dtype: Bound<'_, PyArrayDescr>) -> PyResult<PyClassInitializer<Self>> {
        let k = RustKernel::writing(
            kernels::Copy {
                mode,
                dtype: dtype_of(&dtype)?,
            },
            mode,
            "Copy".into(),
        )?;
        Ok(PyClassInitializer::from(k).add_subclass(Copy))
    }
}

#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Dot;

#[pymethods]
impl Dot {
    #[new]
    fn new(
        mode: OutputMode,
        dtype: Bound<'_, PyArrayDescr>,
        parallel: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = "Dot".to_string();
        let k = match dtype_of(&dtype)? {
            DType::Float64 => RustKernel::writing(
                linalg::Dot::<f64> {
                    parallel,
                    mode,
                    element: PhantomData,
                },
                mode,
                desc,
            )?,
            DType::Float32 => RustKernel::writing(
                linalg::Dot::<f32> {
                    parallel,
                    mode,
                    element: PhantomData,
                },
                mode,
                desc,
            )?,
            dt => return err(format!("Dot of {dt:?}")),
        };
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("Sort(axis={axis})");
        fn sort<T: crate::sort::Sortable>(
            mode: OutputMode,
            axis: usize,
            stable: bool,
            desc: String,
        ) -> PyResult<RustKernel> {
            RustKernel::writing(
                crate::sort::Sort::<T> {
                    mode,
                    axis,
                    stable,
                    element: PhantomData,
                },
                mode,
                desc,
            )
        }
        let k = match dtype_of(&dtype)? {
            DType::Float64 => sort::<f64>(mode, axis, stable, desc)?,
            DType::Float32 => sort::<f32>(mode, axis, stable, desc)?,
            DType::Int64 => sort::<i64>(mode, axis, stable, desc)?,
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("Cholesky(lower={lower})");
        let k = match dtype_of(&dtype)? {
            DType::Float64 => RustKernel::writing(
                linalg::Cholesky::<f64> {
                    parallel,
                    mode,
                    lower,
                    element: PhantomData,
                },
                mode,
                desc,
            )?,
            DType::Float32 => RustKernel::writing(
                linalg::Cholesky::<f32> {
                    parallel,
                    mode,
                    lower,
                    element: PhantomData,
                },
                mode,
                desc,
            )?,
            dt => return err(format!("Cholesky of {dt:?}")),
        };
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("SolveTriangular(lower={lower}, unit_diagonal={unit_diagonal})");
        fn solve<T: linalg::Float>(
            mode: OutputMode,
            lower: bool,
            unit_diagonal: bool,
            parallel: bool,
            desc: String,
        ) -> PyResult<RustKernel> {
            RustKernel::writing(
                linalg::SolveTriangular::<T> {
                    parallel,
                    mode,
                    lower,
                    unit_diagonal,
                    element: PhantomData,
                },
                mode,
                desc,
            )
        }
        let k = match dtype_of(&dtype)? {
            DType::Float64 => solve::<f64>(mode, lower, unit_diagonal, parallel, desc)?,
            DType::Float32 => solve::<f32>(mode, lower, unit_diagonal, parallel, desc)?,
            dt => return err(format!("SolveTriangular of {dt:?}")),
        };
        Ok(PyClassInitializer::from(k).add_subclass(SolveTriangular))
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

/// Exactly `N` output modes.
fn modes<const N: usize>(what: &str, modes: Vec<OutputMode>) -> PyResult<[OutputMode; N]> {
    let n = modes.len();
    modes
        .try_into()
        .or_else(|_| err(format!("{what}: {N} outputs, got {n} modes")))
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let assume = match assume_a {
            "gen" => linalg::Assume::General,
            "sym" => linalg::Assume::Symmetric { lower },
            "pos" => linalg::Assume::PositiveDefinite { lower },
            other => return err(format!("Solve: assume_a={other:?} is not supported")),
        };
        let desc = format!("Solve(assume_a={assume_a}, lower={lower})");
        let k = for_float!(dtype, "Solve", T => RustKernel::writing(
            linalg::Solve::<T> { parallel, mode, assume, element: PhantomData },
            mode,
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("CholeskySolve(lower={lower})");
        let k = for_float!(dtype, "CholeskySolve", T => RustKernel::writing(
            linalg::CholeskySolve::<T> { parallel, mode, lower, element: PhantomData },
            mode,
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let k = for_float!(dtype, "MatrixInverse", T => RustKernel::writing(
            linalg::MatrixInverse::<T> { parallel, mode, element: PhantomData },
            mode,
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let k = for_float!(dtype, "Det", T => RustKernel::writing(
            linalg::Det::<T> { parallel, mode, element: PhantomData },
            mode,
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let fixed = self::modes::<2>("SLogDet", modes.clone())?;
        let k = for_float!(dtype, "SLogDet", T => RustKernel::writing_all(
            linalg::SLogDet::<T> { parallel, modes: fixed, element: PhantomData },
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let fixed = self::modes::<2>("LUFactor", modes.clone())?;
        let k = for_float!(dtype, "LUFactor", T => RustKernel::writing_all(
            linalg::LUFactor::<T> { parallel, modes: fixed, element: PhantomData },
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let form = match form {
            "matrix" => linalg::LuForm::Matrix,
            "indices" => linalg::LuForm::Indices,
            "permute_l" => linalg::LuForm::PermuteL,
            other => return err(format!("LU: unknown form {other:?}")),
        };
        let desc = format!("LU({form:?})");
        let k = for_float!(dtype, "LU", T => RustKernel::writing_all(
            linalg::LU::<T> { parallel, modes: modes.clone(), form, element: PhantomData },
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("PivotToPermutations(inverse={inverse})");
        fn make<I: crate::kernel::Element + TryInto<usize>>(
            mode: OutputMode,
            inverse: bool,
            desc: String,
        ) -> PyResult<RustKernel> {
            RustKernel::writing(
                linalg::PivotToPermutations::<I> {
                    mode,
                    inverse,
                    element: PhantomData,
                },
                mode,
                desc,
            )
        }
        let k = match dtype_of(&dtype)? {
            DType::Int32 => make::<i32>(mode, inverse, desc)?,
            DType::Int64 => make::<i64>(mode, inverse, desc)?,
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("Eigh(lower={lower}, vectors={vectors})");
        let k = for_float!(dtype, "Eigh", T => RustKernel::writing_all(
            linalg::Eigh::<T> { parallel, modes: modes.clone(), lower, vectors, element: PhantomData },
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("SVD(full_matrices={full_matrices}, compute_uv={compute_uv})");
        let k = for_float!(dtype, "SVD", T => RustKernel::writing_all(
            linalg::Svd::<T> { parallel,
                modes: modes.clone(),
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let qr_mode = match mode {
            "full" => linalg::QrMode::Full,
            "economic" => linalg::QrMode::Economic,
            "r" => linalg::QrMode::R,
            other => return err(format!("QR: mode={other:?} is not supported")),
        };
        let desc = format!("QR(mode={mode})");
        let k = for_float!(dtype, "QR", T => RustKernel::writing_all(
            linalg::Qr::<T> { parallel, modes: modes.clone(), mode: qr_mode, element: PhantomData },
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
    ) -> PyResult<PyClassInitializer<Self>> {
        let desc = format!("MatrixPinv(hermitian={hermitian})");
        let k = for_float!(dtype, "MatrixPinv", T => RustKernel::writing(
            linalg::MatrixPinv::<T> { parallel, mode, hermitian, element: PhantomData },
            mode,
            desc,
        )?);
        Ok(PyClassInitializer::from(k).add_subclass(MatrixPinv))
    }
}

/// Scan: a loop around its loop body (a subprogram). Every output is allocated by the kernel.
#[pyclass(module = "pytensor_llvm.interp", extends = RustKernel, frozen)]
pub struct Scan;

#[pymethods]
impl Scan {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (n_seqs, mit_mot_in, mit_mot_out, mit_sot_in, sit_sot_in, n_untraced, n_nit_sot, n_non_seqs, as_while, body, outer_in_dtypes, inner_out_dtypes))]
    fn new(
        py: Python<'_>,
        n_seqs: usize,
        mit_mot_in: Vec<Vec<i64>>,
        mit_mot_out: Vec<Vec<i64>>,
        mit_sot_in: Vec<Vec<i64>>,
        sit_sot_in: Vec<Vec<i64>>,
        n_untraced: usize,
        n_nit_sot: usize,
        n_non_seqs: usize,
        as_while: bool,
        body: Bound<'_, Program>,
        outer_in_dtypes: Vec<Bound<'_, PyArrayDescr>>,
        inner_out_dtypes: Vec<Bound<'_, PyArrayDescr>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let info = super::scan::ScanInfo {
            n_seqs,
            mit_mot_in,
            mit_mot_out,
            mit_sot_in,
            sit_sot_in,
            n_untraced,
            n_nit_sot,
            n_non_seqs,
            as_while,
        };
        let (n_mm, n_ms, n_ss) = (
            info.mit_mot_in.len(),
            info.mit_sot_in.len(),
            info.sit_sot_in.len(),
        );
        let n_in = 1
            + n_seqs
            + info.mit_mot_in.len()
            + info.mit_sot_in.len()
            + info.sit_sot_in.len()
            + n_untraced
            + n_nit_sot
            + n_non_seqs;
        if outer_in_dtypes.len() != n_in {
            return err(format!(
                "Scan: {n_in} outer inputs, got {} dtypes",
                outer_in_dtypes.len()
            ));
        }
        let prog = body.get().prog.clone();
        let n_body_in = n_seqs
            + info.mit_mot_in.iter().map(|t| t.len()).sum::<usize>()
            + info.mit_sot_in.iter().map(|t| t.len()).sum::<usize>()
            + info.sit_sot_in.len()
            + n_untraced
            + n_non_seqs;
        let n_body_out = info.mit_mot_out.iter().map(|t| t.len()).sum::<usize>()
            + info.mit_sot_in.len()
            + info.sit_sot_in.len()
            + n_nit_sot
            + n_untraced
            + as_while as usize;
        if prog.n_inputs() != n_body_in
            || prog.n_outputs() != n_body_out
            || inner_out_dtypes.len() != n_body_out
        {
            return err("Scan: the loop body doesn't match the scan's layout");
        }
        let config = super::scan::ScanConfig {
            info,
            body: prog,
            outer_in_dtypes: outer_in_dtypes
                .iter()
                .map(dtype_of)
                .collect::<PyResult<_>>()?,
            inner_out_dtypes: inner_out_dtypes
                .iter()
                .map(dtype_of)
                .collect::<PyResult<_>>()?,
            keepalive: body.clone().into_any().unbind(),
        };
        // every outer output is one of the kernel's C-contiguous buffers
        let n_outer_out = n_mm + n_ms + n_ss + n_nit_sot + n_untraced;
        let produces = Produces::Fresh(vec![OutputMode::Allocated; n_outer_out]);
        let mut k = RustKernel::of(config, "Scan".into(), produces);
        k.programs.push(body.clone().unbind());
        let _ = py;
        Ok(PyClassInitializer::from(k).add_subclass(Scan))
    }
}

// ------------------------------------------------------------------ programs

/// Builds a program: values, then tasks in execution order, then outputs.
#[pyclass(module = "pytensor_llvm.interp")]
pub struct ProgramBuilder {
    values: Vec<ValueSpec>,
    n_inputs: usize,
    tasks: Vec<Task>,
    keepalive: Vec<Py<PyAny>>,
    arena_bytes: u64,
    contiguous_inputs: Vec<bool>,
    const_buffers: ConstBuffers,
}

impl ProgramBuilder {
    fn add(
        &mut self,
        dtype: &Bound<'_, PyArrayDescr>,
        shape: Vec<Option<u64>>,
        kind: ValueKind,
    ) -> PyResult<u32> {
        self.values.push(ValueSpec {
            dtype: dtype_of(dtype)?,
            shape,
            kind,
            order: vec![],
        });
        Ok((self.values.len() - 1) as u32)
    }
}

#[pymethods]
impl ProgramBuilder {
    #[new]
    fn new() -> Self {
        ProgramBuilder {
            values: vec![],
            n_inputs: 0,
            tasks: vec![],
            keepalive: vec![],
            arena_bytes: 0,
            contiguous_inputs: vec![],
            const_buffers: ConstBuffers::default(),
        }
    }

    /// The next graph input. Returns its value id.
    /// A graph input; `contiguous`: the generated code assumes it is C-contiguous
    /// (Python calls copy other arrays, the C API rejects them).
    #[pyo3(signature = (dtype, shape, contiguous = false))]
    fn input(
        &mut self,
        dtype: Bound<'_, PyArrayDescr>,
        shape: Vec<Option<u64>>,
        contiguous: bool,
    ) -> PyResult<u32> {
        let k = self.n_inputs;
        self.n_inputs += 1;
        self.contiguous_inputs.push(contiguous);
        self.add(&dtype, shape, ValueKind::Input(k))
    }

    /// `order`: the memory order (outermost axis first) to store it in; default C.
    #[pyo3(signature = (array, order = None))]
    fn constant(
        &mut self,
        array: &Bound<'_, PyUntypedArray>,
        order: Option<Vec<usize>>,
    ) -> PyResult<u32> {
        let dt = dtype_of(&array.dtype())?;
        let np = array.py().import("numpy")?;
        let order = order.unwrap_or_default();
        let native = match order.is_empty() {
            true => array.clone().into_any(),
            false => np.call_method1("transpose", (array, order.clone()))?,
        };
        let native = native.call_method1(
            "astype",
            (array.dtype().call_method1("newbyteorder", ("=",))?,),
        )?;
        let bytes: Vec<u8> = native.call_method0("tobytes")?.extract()?;
        let shape: Vec<i64> = array.shape().iter().map(|&n| n as i64).collect();
        let c = ConstData::new(dt, shape.clone(), &order, &bytes, &mut self.const_buffers)
            .map_err(PyValueError::new_err)?;
        self.values.push(ValueSpec {
            dtype: dt,
            shape: shape.iter().map(|&n| Some(n as u64)).collect(),
            kind: ValueKind::Const(Arc::new(c)),
            order: vec![],
        });
        Ok((self.values.len() - 1) as u32)
    }

    /// A static temporary at `offset` in the arena (provided to its producer).
    /// `order`: the memory order (outermost axis first) the value is packed in; default C.
    #[pyo3(signature = (dtype, shape, offset, order = None))]
    fn arena(
        &mut self,
        dtype: Bound<'_, PyArrayDescr>,
        shape: Vec<u64>,
        offset: u64,
        order: Option<Vec<usize>>,
    ) -> PyResult<u32> {
        let k = self.add(
            &dtype,
            shape.into_iter().map(Some).collect(),
            ValueKind::Arena(offset),
        )?;
        self.values[k as usize].order = order.unwrap_or_default();
        Ok(k)
    }

    /// A static graph output: a fresh buffer per call, provided to its producer.
    fn fresh(&mut self, dtype: Bound<'_, PyArrayDescr>, shape: Vec<u64>) -> PyResult<u32> {
        self.add(
            &dtype,
            shape.into_iter().map(Some).collect(),
            ValueKind::Fresh,
        )
    }

    /// A value its producer returns: a buffer it allocates, or a view into the memory
    /// of one of the values `may_alias`.
    fn returned(
        &mut self,
        dtype: Bound<'_, PyArrayDescr>,
        shape: Vec<Option<u64>>,
        may_alias: Vec<u32>,
    ) -> PyResult<u32> {
        self.add(
            &dtype,
            shape,
            ValueKind::Returned(may_alias.into_iter().map(ValueId).collect()),
        )
    }

    /// Append a task. `keepalive` is kept alive as long as the program (the compiled
    /// module or kernel object the address and config belong to).
    #[pyo3(signature = (address, config, inputs, outputs, release, description, keepalive))]
    fn task(
        &mut self,
        address: usize,
        config: usize,
        inputs: Vec<u32>,
        outputs: Vec<u32>,
        release: Vec<u32>,
        description: String,
        keepalive: Py<PyAny>,
    ) -> PyResult<()> {
        // SAFETY: what Python promises when it adds a task: `address` and `config`
        // are those of a compiled function (a kernel wrapper), of a kernel object
        // (`RustKernel`, `Program`: its `address` and `config`) or of a numba kernel,
        // made for this node, and `keepalive` is what they belong to.
        let kernel = unsafe { KernelHandle::from_raw(address, config) }
            .ok_or_else(|| PyValueError::new_err(format!("task {description}: null function")))?;
        self.tasks.push(Task {
            kernel,
            inputs: inputs.into_iter().map(ValueId).collect(),
            outputs: outputs.into_iter().map(ValueId).collect(),
            release: release.into_iter().map(ValueId).collect(),
            desc: description,
        });
        self.keepalive.push(keepalive);
        Ok(())
    }

    fn set_arena_bytes(&mut self, n: u64) {
        self.arena_bytes = n;
    }

    /// Finish with these outputs. `modes` say how each output is produced when the
    /// program runs as a task of another program (Provided or Allocated).
    fn build(&mut self, outputs: Vec<u32>, modes: Vec<OutputMode>) -> PyResult<Program> {
        if modes.contains(&OutputMode::View) {
            return Err(PyValueError::new_err("program outputs can't be views"));
        }
        let output_modes = modes;
        let spec = ProgramSpec {
            values: std::mem::take(&mut self.values),
            n_inputs: self.n_inputs,
            tasks: std::mem::take(&mut self.tasks),
            outputs: outputs.into_iter().map(ValueId).collect(),
            output_modes,
            arena_bytes: self.arena_bytes,
            contiguous_inputs: std::mem::take(&mut self.contiguous_inputs),
        };
        let prog = RProgram::new(spec).map_err(PyValueError::new_err)?;
        Ok(Program {
            prog: Arc::new(prog),
            keepalive: std::mem::take(&mut self.keepalive),
            session: std::sync::Mutex::new(None),
        })
    }
}

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

/// A runnable program: call it with arrays, or use it as a task of another program
/// (`address` / `config`).
#[pyclass(module = "pytensor_llvm.interp", frozen)]
pub struct Program {
    prog: Arc<RProgram>,
    keepalive: Vec<Py<PyAny>>,
    /// the session of Python calls: one per function (one random state);
    /// concurrent calls from several threads take turns
    session: std::sync::Mutex<Option<Session>>,
}

/// A session of Python calls: the program's call memory and the context.
struct Session {
    memory: super::program::CallMemory,
    ctx: crate::runtime::PtCtx,
}

#[pymethods]
impl Program {
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyList>> {
        let prog = &*self.prog;
        if args.len() != prog.n_inputs() {
            return err(format!(
                "expected {} inputs, got {}",
                prog.n_inputs(),
                args.len()
            ));
        }
        let mut guard = self.lock_session(py);
        let Session { memory: st, ctx } = guard.get_or_insert_with(|| Session {
            memory: prog.take_memory(),
            ctx: crate::runtime::PtCtx::new(),
        });
        // the arrays to call with, kept alive until the call is over, and their
        let mut keep: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
        // data, shapes and strides (per array: its data, and where its lengths start in
        // `dims`; its strides follow them)
        let mut parts: Vec<(*const u8, usize, usize)> = Vec::with_capacity(args.len());
        let mut dims: Vec<i64> = Vec::with_capacity(8 * args.len());
        let mut work = 0usize;
        let res = (|| -> PyResult<Vec<OwnedArray>> {
            for (k, a) in args.iter().enumerate() {
                let want = prog.input_dtype(k);
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
                } else if prog.input_contiguous(k)
                    && flags & numpy::npyffi::NPY_ARRAY_C_CONTIGUOUS == 0
                {
                    py.import("numpy")?
                        .call_method1("ascontiguousarray", (a,))?
                } else {
                    a
                };
                let arr = a.cast::<PyUntypedArray>()?;
                // SAFETY: reading the data pointer of a live array.
                let data = unsafe { (*arr.as_array_ptr()).fields.data as *const u8 };
                parts.push((data, dims.len(), arr.ndim()));
                dims.extend(arr.shape().iter().map(|&n| n as i64));
                dims.extend(arr.strides().iter().map(|&s| s as i64));
                work += arr.len();
                keep.push(a);
            }
            let mut call = prog.begin(st).map_err(PyValueError::new_err)?;
            for (k, &(data, start, ndim)) in parts.iter().enumerate() {
                let (shape, strides) = dims[start..start + 2 * ndim].split_at(ndim);
                // SAFETY: the memory of a numpy array of the input's dtype (converted
                // above), which `keep` keeps alive until the call is over. Nothing
                // here writes it; as everywhere with numpy, that no other thread
                // does meanwhile is up to the user.
                unsafe { PtArrayRef::checked(data, shape, strides, prog.input_dtype(k)) }
                    .and_then(|a| call.set_input(k, a))
                    .map_err(PyValueError::new_err)?;
            }
            let run = move |ctx: &mut crate::runtime::PtCtx| -> Result<Vec<OwnedArray>, String> {
                let done = call.run(ctx)?;
                (0..prog.n_outputs()).map(|o| done.take_output(o)).collect()
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

    /// The kernel ABI entry for running this program as a subprogram.
    #[getter]
    fn address(&self) -> usize {
        crate::kernel::address::<RProgram>()
    }

    /// The environment for `address`: this program (valid while this object lives).
    #[getter]
    fn config(&self) -> usize {
        Arc::as_ptr(&self.prog) as usize
    }

    /// The program as text (values, storage, tasks, releases).
    fn dump(&self) -> String {
        self.prog.dump()
    }

    fn __str__(&self) -> String {
        self.prog.dump()
    }

    #[getter]
    fn n_tasks(&self) -> usize {
        self.prog.spec().tasks.len()
    }

    #[getter]
    fn arena_bytes(&self) -> u64 {
        self.prog.spec().arena_bytes
    }

    /// Turn per-task timing on or off, here and in subprograms (a task that is a
    /// subprogram, Scan's loop body). Timing costs two clock reads per task.
    fn set_profiling(&self, py: Python<'_>, on: bool) {
        self.prog.set_profiling(on);
        for c in self.children(py).into_iter().flatten() {
            c.get().set_profiling(py, on);
        }
    }

    fn reset_profile(&self, py: Python<'_>) {
        self.prog.reset_profile();
        for c in self.children(py).into_iter().flatten() {
            c.get().reset_profile(py);
        }
    }

    /// Rows `(depth, description, calls, seconds)`: each task with its total time,
    /// followed by the tasks of the programs it runs at depth + 1; `calls` is the number
    /// of runs of the program the task belongs to.
    fn profile(&self, py: Python<'_>) -> Vec<(usize, String, u64, f64)> {
        let mut rows = vec![];
        self.profile_rows(py, 0, &mut rows);
        rows
    }

    /// The profile as a table, slowest first within each program.
    fn profile_report(&self, py: Python<'_>) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let (calls, _) = self.prog.profile();
        let total: f64 = self.prog.profile().1.iter().sum();
        let _ = writeln!(
            s,
            "{calls} calls, {:.3} ms in tasks ({:.2} us per call)",
            total * 1e3,
            if calls > 0 {
                total / calls as f64 * 1e6
            } else {
                0.0
            }
        );
        let _ = writeln!(
            s,
            "{:>7} {:>10} {:>10}  task (subprogram tasks indented)",
            "%", "total ms", "us/call"
        );
        self.report_into(py, 0, total, calls, &mut s);
        s
    }

    /// Restart the random state of Python calls from `seed` (draws are then
    /// reproducible). The C API seeds each of its sessions (`seed` in the table).
    fn seed(&self, py: Python<'_>, seed: u64) {
        let prog = &self.prog;
        let mut guard = self.lock_session(py);
        let s = guard.get_or_insert_with(|| Session {
            memory: prog.take_memory(),
            ctx: crate::runtime::PtCtx::new(),
        });
        s.ctx.seed(seed);
    }

    /// Allow or forbid parallel loops for Python calls (results are the same either
    /// way). The C API sets this per session (`set_parallel` in the table).
    fn set_parallel(&self, py: Python<'_>, on: bool) {
        let prog = &self.prog;
        let mut guard = self.lock_session(py);
        let s = guard.get_or_insert_with(|| Session {
            memory: prog.take_memory(),
            ctx: crate::runtime::PtCtx::new(),
        });
        s.ctx.parallel = on;
    }

    /// A new reference to this program for the C API, as the address of a
    /// `PtFunction` (release it with `function_decref` from `c_api_table()`). It keeps
    /// the compiled code alive on its own.
    fn c_function(&self, py: Python<'_>) -> usize {
        let keep = self.keepalive.iter().map(|o| o.clone_ref(py)).collect();
        super::capi::Function::new(self.prog.clone(), keep).into_handle() as usize
    }

    /// The address of the `PtFunctionApi` table for functions from `c_function()`.
    #[staticmethod]
    fn c_api_table() -> usize {
        &super::capi::API as *const _ as usize
    }

    #[getter]
    fn tasks(&self) -> Vec<String> {
        let _ = &self.keepalive;
        self.prog
            .spec()
            .tasks
            .iter()
            .map(|s| s.desc.clone())
            .collect()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OutputMode>()?;
    m.add_class::<RustKernel>()?;
    m.add_class::<DimShuffle>()?;
    m.add_class::<Copy>()?;
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
    m.add_class::<Scan>()?;
    m.add_class::<ProgramBuilder>()?;
    m.add_class::<Program>()?;
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

/// The address of the `PtFunctionApi` table for functions from `Program.c_function()`
/// (header: `pytensor_llvm.c_header()`).
#[pyfunction]
fn c_api_table() -> usize {
    &super::capi::API as *const _ as usize
}

impl Program {
    /// The session, waiting without the GIL if another thread has it.
    fn lock_session(&self, py: Python<'_>) -> std::sync::MutexGuard<'_, Option<Session>> {
        loop {
            match self.session.try_lock() {
                Ok(g) => return g,
                Err(std::sync::TryLockError::Poisoned(p)) => return p.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => py.detach(|| drop(self.session.lock())),
            }
        }
    }

    /// For each task, the programs it runs.
    fn children(&self, py: Python<'_>) -> Vec<Vec<Py<Program>>> {
        self.keepalive
            .iter()
            .map(|k| {
                let k = k.bind(py);
                if let Ok(p) = k.cast::<Program>() {
                    vec![p.clone().unbind()]
                } else if let Ok(kern) = k.cast::<RustKernel>() {
                    kern.get()
                        .programs
                        .iter()
                        .map(|p| p.clone_ref(py))
                        .collect()
                } else {
                    vec![]
                }
            })
            .collect()
    }

    fn profile_rows(
        &self,
        py: Python<'_>,
        depth: usize,
        rows: &mut Vec<(usize, String, u64, f64)>,
    ) {
        let (calls, secs) = self.prog.profile();
        let children = self.children(py);
        for (k, task) in self.prog.spec().tasks.iter().enumerate() {
            rows.push((depth, task.desc.clone(), calls, secs[k]));
            for c in children.get(k).into_iter().flatten() {
                c.get().profile_rows(py, depth + 1, rows);
            }
        }
    }

    /// `calls`: of the top-level program (us/call is per top-level call).
    fn report_into(&self, py: Python<'_>, depth: usize, total: f64, calls: u64, s: &mut String) {
        use std::fmt::Write;
        let (_, secs) = self.prog.profile();
        let children = self.children(py);
        let mut order: Vec<usize> = (0..secs.len()).collect();
        order.sort_by(|&a, &b| secs[b].total_cmp(&secs[a]));
        for k in order {
            let pct = if total > 0.0 {
                100.0 * secs[k] / total
            } else {
                0.0
            };
            let per = if calls > 0 {
                secs[k] / calls as f64 * 1e6
            } else {
                0.0
            };
            let desc: String = self.prog.spec().tasks[k].desc.chars().take(100).collect();
            let _ = writeln!(
                s,
                "{pct:>6.1}% {:>10.3} {per:>10.2}  {}{desc}",
                secs[k] * 1e3,
                "  ".repeat(depth)
            );
            for c in children.get(k).into_iter().flatten() {
                c.get().report_into(py, depth + 1, total, calls, s);
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
