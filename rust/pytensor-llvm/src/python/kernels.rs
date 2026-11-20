//! The Rust kernels for Python: `RustKernel` and a class per kernel, whose constructor
//! makes the kernel for the layouts of its inputs.

use super::layout::Layout;
use super::{dtype_of, err};
use crate::abi::OutputMode;
use crate::dtype::DType;
use crate::kernels::{Kernel, linalg};
use numpy::PyArrayDescr;
use pyo3::PyClassInitializer;
use pyo3::prelude::*;
use std::marker::PhantomData;

/// A Rust kernel: its address and its config (owned by this object).
///
/// Every kernel is made for the layouts of its inputs (`in_layouts`, None where
/// unknown) and the number of axes of its outputs (`out_ndims`), the last two arguments
/// of each constructor: it chooses the layouts of its outputs then. The planner chooses
/// whether each output is provided or allocated (the output modes, the first argument).
/// `outputs` says both.
#[pyclass(module = "pytensor_llvm.runtime", subclass, frozen)]
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
    /// A kernel written with `crate::kernels`: its own function, and itself as config.
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
            address: crate::kernels::address::<K>(),
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
        fn sort<T: crate::kernels::sort::Sortable>(
            mode: OutputMode,
            axis: usize,
            stable: bool,
            layout: Option<&Layout>,
            ndim: usize,
            desc: String,
        ) -> PyResult<RustKernel> {
            RustKernel::of(
                crate::kernels::sort::Sort::<T>::new(mode, axis, stable, layout, ndim),
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
        fn make<I: crate::kernels::Element + TryInto<usize>>(
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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
#[pyclass(module = "pytensor_llvm.runtime", extends = RustKernel, frozen)]
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

