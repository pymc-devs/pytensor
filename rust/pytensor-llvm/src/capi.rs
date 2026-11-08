//! C API for calling compiled functions without Python. See `include/pytensor_llvm.h`.
//!
//! A `PtFunction*` is `Arc::into_raw` of the `Program`; a `PtCallState*` owns a clone of
//! that Arc plus the per-thread `CallState`. Every entry point catches panics so they
//! never unwind into C.

use crate::abi::{PtArray, PtBuffer, ABI_VERSION};
use crate::runtime::{CallState, Program};
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

pub const API_VERSION: u32 = 1;

#[repr(C)]
pub struct PtTypeInfo {
    pub dtype: i32,
    pub ndim: i32,
    pub static_shape: *const i64,
}

pub struct PtCallState {
    prog: Arc<Program>,
    st: CallState,
    err: CString,
}

#[repr(C)]
pub struct PtFunctionApi {
    pub abi_version: u32,
    pub api_version: u32,
    pub function_incref: unsafe extern "C" fn(*const Program),
    pub function_decref: unsafe extern "C" fn(*const Program),
    pub n_inputs: unsafe extern "C" fn(*const Program) -> usize,
    pub n_outputs: unsafe extern "C" fn(*const Program) -> usize,
    pub input_type: unsafe extern "C" fn(*const Program, usize, *mut PtTypeInfo) -> i32,
    pub output_type: unsafe extern "C" fn(*const Program, usize, *mut PtTypeInfo) -> i32,
    pub state_new: unsafe extern "C" fn(*const Program) -> *mut PtCallState,
    pub state_free: unsafe extern "C" fn(*mut PtCallState),
    pub call: unsafe extern "C" fn(*mut PtCallState, *const PtArray, *mut PtArray) -> i32,
    pub last_error: unsafe extern "C" fn(*const PtCallState) -> *const c_char,
    pub buffer_decref: unsafe extern "C" fn(*mut PtBuffer),
}

/// Hand out a new reference to `prog` as a `PtFunction*`.
pub fn function_handle(prog: &Arc<Program>) -> *const Program {
    Arc::into_raw(prog.clone())
}

// SAFETY contract of every entry point below (the C API's preconditions, as documented
// in pytensor_llvm.h; the caller's responsibility, not checkable here):
// - `f` is a live `PtFunction*`: from `c_function()` / `function_incref`, not yet
//   released. It is `Arc::into_raw` of a `Program`.
// - `s` is a live `PtCallState*` from `state_new`, used by one thread at a time.
// - out-pointers point to writable memory of the right type.
// Within that contract the functions are sound: everything they read through the
// arrays passed to `call` is validated first (see `call`).

unsafe extern "C" fn function_incref(f: *const Program) {
    // SAFETY: `f` is a live PtFunction (contract above).
    unsafe { Arc::increment_strong_count(f) };
}

unsafe extern "C" fn function_decref(f: *const Program) {
    // SAFETY: the caller releases a reference it holds.
    unsafe { Arc::decrement_strong_count(f) };
}

unsafe extern "C" fn n_inputs(f: *const Program) -> usize {
    // SAFETY: `f` is live.
    unsafe { (*f).n_inputs() }
}

unsafe extern "C" fn n_outputs(f: *const Program) -> usize {
    // SAFETY: `f` is live.
    unsafe { (*f).n_outputs() }
}

/// `static_shape` points into the Program, valid while the caller holds `f`.
unsafe fn fill_type(dt: crate::dtype::DType, shape: &[i64], out: *mut PtTypeInfo) {
    // SAFETY: `out` is writable (contract).
    unsafe { *out = PtTypeInfo { dtype: dt.code(), ndim: shape.len() as i32, static_shape: shape.as_ptr() } };
}

unsafe extern "C" fn input_type(f: *const Program, i: usize, out: *mut PtTypeInfo) -> i32 {
    // SAFETY: `f` is live.
    let f = unsafe { &*f };
    if i >= f.n_inputs() {
        return 1;
    }
    let (dt, sh) = f.input_type(i);
    // SAFETY: `out` is writable (contract).
    unsafe { fill_type(dt, sh, out) };
    0
}

unsafe extern "C" fn output_type(f: *const Program, i: usize, out: *mut PtTypeInfo) -> i32 {
    // SAFETY: `f` is live.
    let f = unsafe { &*f };
    if i >= f.n_outputs() {
        return 1;
    }
    let (dt, sh) = f.output_type(i);
    // SAFETY: `out` is writable (contract).
    unsafe { fill_type(dt, sh, out) };
    0
}

unsafe extern "C" fn state_new(f: *const Program) -> *mut PtCallState {
    // a new strong reference, owned by the state
    // SAFETY: `f` is live; the state owns the new reference.
    let prog = unsafe {
        Arc::increment_strong_count(f);
        Arc::from_raw(f)
    };
    let st = prog.new_state();
    Box::into_raw(Box::new(PtCallState { prog, st, err: CString::default() }))
}

unsafe extern "C" fn state_free(s: *mut PtCallState) {
    if !s.is_null() {
        // SAFETY: `s` came from state_new and is freed once.
        drop(unsafe { Box::from_raw(s) });
    }
}

/// Run the function: see `call` in pytensor_llvm.h. Before anything reads through a
/// caller pointer, the input/output counts come from the Program and each array's
/// `ndim` and shape/stride pointers are checked; `set_input_raw` and `execute` then
/// validate data, shapes, strides, alignment and output overlap.
unsafe extern "C" fn call(s: *mut PtCallState, inputs: *const PtArray, outputs: *mut PtArray) -> i32 {
    // SAFETY: `s` is live and used by this thread only (contract).
    let s = unsafe { &mut *s };
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let prog = &*s.prog;
        if (inputs.is_null() && prog.n_inputs() > 0) || (outputs.is_null() && prog.n_outputs() > 0) {
            return Err("inputs and outputs must not be NULL".into());
        }
        // Before building any slice from caller pointers: ndim must be the expected one
        // (so shape/strides are read with a length we know), and they must be non-null.
        let check = |what: &str, k: usize, a: &PtArray, nd: usize| -> Result<(), String> {
            if a.ndim < 0 || a.ndim as usize != nd {
                return Err(format!("{what} {k}: expected {nd} dims, got {}", a.ndim));
            }
            if nd > 0 && (a.shape.is_null() || a.strides.is_null()) {
                return Err(format!("{what} {k}: shape and strides must not be NULL"));
            }
            Ok(())
        };
        for k in 0..prog.n_inputs() {
            // SAFETY: `inputs` holds n_inputs arrays (contract); shape/strides checked above
            let a = unsafe { &*inputs.add(k) };
            check("input", k, a, prog.input_type(k).1.len())?;
            // SAFETY: ndim and the shape/stride pointers were checked just above.
            prog.set_input_raw(&mut s.st, k, a.data, unsafe { a.shape() }, unsafe { a.strides() })?;
        }
        for k in 0..prog.n_outputs() {
            // SAFETY: `outputs` holds n_outputs arrays (contract).
            check("output", k, unsafe { &*outputs.add(k) }, prog.output_type(k).1.len())?;
        }
        // SAFETY: n_outputs arrays (contract), each with a checked ndim and pointers
        let outs = unsafe { std::slice::from_raw_parts_mut(outputs, prog.n_outputs()) };
        prog.execute(&mut s.st, Some(outs))?;
        for (o, r) in outs.iter_mut().zip(&s.st.outs) {
            if o.data.is_null() {
                // runtime-allocated: the caller takes over `r.owner`
                o.data = r.data;
                o.owner = r.owner;
                // SAFETY: o's shape/strides point to caller arrays of length ndim
                // (checked equal to the output's ndim above).
                unsafe {
                    o.shape_mut().copy_from_slice(r.shape());
                    o.strides_mut().copy_from_slice(r.strides());
                }
            }
        }
        Ok(())
    }));
    let msg = match res {
        Ok(Ok(())) => return 0,
        Ok(Err(m)) => m,
        Err(_) => "internal error: panic in pytensor_llvm".to_string(),
    };
    s.err = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    1
}

unsafe extern "C" fn last_error(s: *const PtCallState) -> *const c_char {
    // SAFETY: `s` is live; the string lives until the next call on `s`.
    unsafe { (*s).err.as_ptr() }
}

unsafe extern "C" fn buffer_decref(b: *mut PtBuffer) {
    if !b.is_null() {
        // SAFETY: the caller releases a reference it got from `call`
        unsafe { PtBuffer::decref(b) }
    }
}

pub static API: PtFunctionApi = PtFunctionApi {
    abi_version: ABI_VERSION,
    api_version: API_VERSION,
    function_incref,
    function_decref,
    n_inputs,
    n_outputs,
    input_type,
    output_type,
    state_new,
    state_free,
    call,
    last_error,
    buffer_decref,
};
