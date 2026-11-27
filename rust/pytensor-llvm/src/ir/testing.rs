//! A runner for kernels (tests): `ir.call_kernel`.

use crate::abi::{OwnedBuf, PtArray, PtBuffer};
use crate::dtype::DType;
use crate::runtime::PtCtx;
use numpy::npyffi::{NPY_ARRAY_WRITEABLE, NpyTypes, PY_ARRAY_API, npy_intp};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

// Generated code calls the runtime functions of `crate::runtime::runtime()` (`ptl_rt_*`).

// ------------------------------------------------------------------ running C ABI functions

use crate::abi::KernelFn;

fn dtype_of(d: &Bound<'_, PyArrayDescr>) -> PyResult<DType> {
    DType::from_numpy(d.kind(), d.itemsize())
        .ok_or_else(|| PyTypeError::new_err(format!("dtype {d} is not supported")))
}

/// A numpy array over `data`; owns `owner` (one reference) if not null, else copies.
fn to_numpy<'py>(
    py: Python<'py>,
    dtype: &Bound<'py, PyArrayDescr>,
    data: *mut u8,
    shape: &[i64],
    strides: &[i64],
    owner: *mut PtBuffer,
) -> PyResult<Bound<'py, PyAny>> {
    let mut lengths: Vec<npy_intp> = shape.iter().map(|&s| s as npy_intp).collect();
    let mut st: Vec<npy_intp> = strides.iter().map(|&s| s as npy_intp).collect();
    // SAFETY: `data` with shape/strides is a valid array (a kernel output); the descr
    // reference is stolen by numpy; the capsule holds the buffer reference until the
    // array is freed.
    unsafe {
        let ty = numpy::npyffi::get_type_object(py, NpyTypes::PyArray_Type);
        let arr = PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            ty,
            dtype.clone().into_ptr() as *mut _,
            lengths.len() as i32,
            lengths.as_mut_ptr(),
            st.as_mut_ptr(),
            data as *mut _,
            NPY_ARRAY_WRITEABLE,
            std::ptr::null_mut(),
        );
        let arr = Bound::from_owned_ptr_or_err(py, arr)?;
        if owner.is_null() {
            return arr.call_method0("copy");
        }
        // SAFETY: the reference the kernel handed over in its output (the caller's
        // promise about the kernel).
        let buf = OwnedBuf::from_raw(owner);
        let capsule = PyCapsule::new_with_value(py, buf, c"pytensor_llvm.buffer")?;
        if PY_ARRAY_API.PyArray_SetBaseObject(py, arr.as_ptr() as *mut _, capsule.into_ptr()) != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(arr)
    }
}

/// Call a kernel (`int32 f(ctx, args, n_inputs, n_outputs, config)`), for tests.
///
/// `inputs` are arrays; `outputs` are arrays (provided memory, written in place) or
/// `(dtype, ndim)` for outputs the function sets (allocated or views; views of inputs are
/// copied). Returns the outputs.
///
/// # Safety (for the caller)
/// `address` must be a function with that ABI expecting exactly these arguments.
#[pyfunction]
#[pyo3(signature = (address, inputs, outputs, config = 0))]
pub fn call_kernel<'py>(
    py: Python<'py>,
    address: usize,
    inputs: Vec<Bound<'py, PyAny>>,
    outputs: Vec<Bound<'py, PyAny>>,
    config: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let n = inputs.len() + outputs.len();
    let mut storage: Vec<Vec<i64>> = Vec::with_capacity(n);
    let mut arrays: Vec<PtArray> = Vec::with_capacity(n);
    let mut out_kind: Vec<Option<Bound<'py, PyArrayDescr>>> = Vec::new();
    let mut keep = Vec::new();
    let push_array = |a: &Bound<'py, PyAny>,
                      storage: &mut Vec<Vec<i64>>,
                      arrays: &mut Vec<PtArray>|
     -> PyResult<()> {
        let arr = a.cast::<PyUntypedArray>()?;
        let dtype = dtype_of(&arr.dtype())?;
        let nd = arr.ndim();
        let mut s: Vec<i64> = arr.shape().iter().map(|&x| x as i64).collect();
        s.extend(arr.strides().iter().map(|&x| x as i64));
        storage.push(s);
        let st = storage.last_mut().unwrap().as_mut_ptr();
        // SAFETY: reading the data pointer of a live array, kept alive by `keep`; the
        // inner storage vector holds 2 * ndim entries (its buffer never moves).
        let (data, strides) = unsafe { ((*arr.as_array_ptr()).fields.data as *mut u8, st.add(nd)) };
        arrays.push(PtArray {
            data,
            shape: st,
            strides,
            owner: std::ptr::null_mut(),
            ndim: nd as i32,
            dtype: dtype.code(),
        });
        Ok(())
    };
    for a in &inputs {
        push_array(a, &mut storage, &mut arrays)?;
        keep.push(a.clone());
    }
    for o in &outputs {
        if o.cast::<PyUntypedArray>().is_ok() {
            push_array(o, &mut storage, &mut arrays)?;
            out_kind.push(None);
        } else {
            let (dt, nd): (Bound<'py, PyArrayDescr>, usize) = o.extract()?;
            let dtype = dtype_of(&dt)?;
            storage.push(vec![0; 2 * nd.max(1)]);
            let st = storage.last_mut().unwrap().as_mut_ptr();
            // SAFETY: the storage holds 2 * ndim entries.
            let strides = unsafe { st.add(nd) };
            arrays.push(PtArray {
                data: std::ptr::null_mut(),
                shape: st,
                strides,
                owner: std::ptr::null_mut(),
                ndim: nd as i32,
                dtype: dtype.code(),
            });
            out_kind.push(Some(dt));
        }
    }
    let mut ctx = PtCtx::new();
    // SAFETY: the caller promises `address` has the C ABI for these arguments.
    let f: KernelFn = unsafe { std::mem::transmute::<usize, KernelFn>(address) };
    let mark = ctx.scratch_mark();
    // SAFETY: as above; arrays describe live numpy memory or empty output slots.
    let rc = unsafe {
        f(
            &mut ctx,
            arrays.as_mut_ptr(),
            inputs.len() as i32,
            outputs.len() as i32,
            config as *const u8,
        )
    };
    ctx.scratch_release(mark);
    let n_in = inputs.len();
    if rc != 0 {
        for a in &arrays[n_in..] {
            if !a.owner.is_null() {
                // SAFETY: the callee handed this reference over in its output slot.
                drop(unsafe { OwnedBuf::from_raw(a.owner) });
            }
        }
        let msg = ctx
            .error
            .take()
            .map(|(_, m)| m)
            .unwrap_or_else(|| format!("error {rc}"));
        return Err(PyRuntimeError::new_err(msg));
    }
    let mut res = Vec::new();
    for (k, kind) in out_kind.into_iter().enumerate() {
        let a = &arrays[n_in + k];
        match kind {
            None => res.push(outputs[k].clone()),
            Some(dt) => {
                // (the callee set the slot's shape and strides, in our storage)
                let nd = a.ndim as usize;
                let (shape, strides) = storage[n_in + k].split_at(nd);
                let (shape, strides) = (shape.to_vec(), strides[..nd].to_vec());
                if a.data.is_null() && shape.iter().product::<i64>() != 0 {
                    return Err(PyValueError::new_err(format!("output {k} was not set")));
                }
                res.push(to_numpy(py, &dt, a.data, &shape, &strides, a.owner)?);
            }
        }
    }
    drop(keep);
    Ok(res)
}
