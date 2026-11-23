//! Compiled functions for Python: `Function`, called with numpy arrays.

use super::{descr, dtype_of, err};
use crate::array::{OwnedArray, PtArrayRef};
use crate::dtype::DType;
use crate::function::{ArgType, Destination, Function as RFunction, Session};
use numpy::npyffi::{NPY_ARRAY_WRITEABLE, NpyTypes, PY_ARRAY_API, npy_intp};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};
use std::sync::Arc;

/// A numpy array of `array` (not copied: the numpy array holds its buffer).
fn wrap_output<'py>(py: Python<'py>, array: OwnedArray) -> PyResult<Bound<'py, PyAny>> {
    let dt = array.dtype();
    let (buf, data, shape, strides) = array.into_parts();
    // the capsule holds the buffer reference until numpy frees the array (its base)
    let capsule = PyCapsule::new_with_value(py, buf, c"pytensor_llvm.buffer")?;
    let mut lengths: Vec<npy_intp> = shape.iter().map(|&s| s as npy_intp).collect();
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
            lengths.len() as i32,
            lengths.as_mut_ptr(),
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
#[pyclass(module = "pytensor_llvm.runtime", frozen)]
pub struct Function {
    f: Arc<RFunction>,
    keepalive: Vec<Py<PyAny>>,
    /// the session of Python calls: one per function (one random state); concurrent
    /// calls from several threads take turns
    session: std::sync::Mutex<Option<Session>>,
}

#[pymethods]
impl Function {
    /// `inputs`: `(dtype, shape, contiguous)` per graph input (shape: None where only
    /// known at run time; contiguous: the generated code assumes C-contiguous memory);
    /// `outputs`: `(dtype, shape, provided)` per graph output (provided: written into
    /// memory the caller provides, else returned in a buffer). `keepalive` is kept as
    /// long as the function: what `address` and `config` belong to. `uses_pool`: may
    /// the function run work on the runtime's thread pool?
    #[new]
    fn new(
        address: usize,
        config: usize,
        inputs: Vec<(Bound<'_, PyArrayDescr>, Vec<Option<u64>>, bool)>,
        outputs: Vec<(Bound<'_, PyArrayDescr>, Vec<Option<u64>>, bool)>,
        keepalive: Vec<Py<PyAny>>,
        uses_pool: bool,
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
        let f =
            unsafe {
                RFunction::new(
                    address, config, in_types, contiguous, out_types, provided, uses_pool,
                )
            }
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
        let session = guard.get_or_insert_with(Session::new);
        // the arrays to call with, kept alive until the call is over, and their
        let mut keep: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
        // data, shapes and strides (per array: its data, and where its lengths start in
        // `headers`; its strides follow them)
        let mut parts: Vec<(usize, usize, usize)> = Vec::with_capacity(args.len());
        let mut headers: Vec<i64> = Vec::with_capacity(8 * args.len());
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
                parts.push((data, headers.len(), arr.ndim()));
                headers.extend(arr.shape().iter().map(|&n| n as i64));
                headers.extend(arr.strides().iter().map(|&s| s as i64));
                keep.push(a);
            }
            let run = move |session: &mut Session| -> Result<Vec<OwnedArray>, String> {
                let ins = parts.iter().enumerate().map(|(k, &(data, start, ndim))| {
                    let (shape, strides) = headers[start..start + 2 * ndim].split_at(ndim);
                    // SAFETY: the memory of a numpy array of the input's dtype
                    // (converted above), which `keep` keeps alive until the call is
                    // over. Nothing here writes it; as everywhere with numpy, that no
                    // other thread does meanwhile is up to the user.
                    unsafe {
                        PtArrayRef::checked(data as *const u8, shape, strides, f.inputs[k].dtype)
                    }
                    .map_err(|e| format!("input {k}: {e}"))
                });
                let outs = (0..f.n_outputs()).map(|_| Ok(Destination::New));
                Ok(f.call(session, ins, outs)?.map(|(_, a)| a).collect())
            };
            // without the GIL: other Python threads run meanwhile, and what the call
            // runs on other threads may need it (numba kernels in object mode)
            py.detach(move || run(session)).map_err(runtime_error)
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
        self.lock_session(py)
            .get_or_insert_with(Session::new)
            .ctx
            .seed(seed);
    }

    /// Allow or forbid parallel loops for Python calls (results are the same either
    /// way). The C API sets this per session (`set_parallel` in the table).
    fn set_parallel(&self, py: Python<'_>, on: bool) {
        self.lock_session(py)
            .get_or_insert_with(Session::new)
            .ctx
            .parallel = on;
    }

    /// A new reference to this function for the C API, as the address of a
    /// `PtFunction` (release it with `function_decref` from `c_api_table()`). It keeps
    /// the compiled code alive on its own.
    fn c_function(&self, py: Python<'_>) -> usize {
        let keep = self.keepalive.iter().map(|o| o.clone_ref(py)).collect();
        crate::function::capi::Function::new(self.f.clone(), keep).into_handle() as usize
    }

    /// The address of the `PtFunctionApi` table for functions from `c_function()`.
    #[staticmethod]
    fn c_api_table() -> usize {
        &crate::function::capi::API as *const _ as usize
    }
}

impl Function {
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
}
