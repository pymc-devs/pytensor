//! The C API (`include/pytensor_llvm.h`) for compiled programs.
//!
//! Get the table with `pytensor_llvm.runtime.c_api_table()` (or `program.c_api_table()`), and
//! a function with `program.c_function()`.
//!
//! A `PtFunction*` is `Arc::into_raw` of a [`Function`]: the compiled function plus the
//! Python objects that own its machine code and kernel environments, so the function
//! outlives the Python object it came from. A `PtSession*` owns a clone of that Arc and
//! a context. Every entry point catches panics so they never unwind into C.

use super::{Destination, Function as Compiled, Session};
use crate::abi::{ABI_VERSION, OwnedBuf, PtArray, PtBuffer};
use crate::array::{OutSlot, PtArrayMut, PtArrayRef};
use crate::dtype::DType;
use pyo3::prelude::*;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

pub const API_VERSION: u32 = 3;

/// The type of an input or output (`input_type` / `output_type`).
#[repr(C)]
pub struct PtTypeInfo {
    pub dtype: i32,
    pub ndim: i32,
    pub static_shape: *const i64,
}

/// What a `PtFunction*` points to.
pub struct Function {
    pub prog: Arc<Compiled>,
    /// compiled modules, kernels and link tables the function's code points into;
    /// dropped (possibly without the GIL: pyo3 defers the decref) after the last
    /// reference
    pub keepalive: Vec<Py<PyAny>>,
    /// static shape per input, then per output (-1: known at run time), handed out by
    /// `input_type` / `output_type`
    shapes: Vec<Vec<i64>>,
}

impl Function {
    pub fn new(prog: Arc<Compiled>, keepalive: Vec<Py<PyAny>>) -> Function {
        let shapes = prog
            .inputs
            .iter()
            .chain(&prog.outputs)
            .map(|t| t.shape_or_unknown())
            .collect();
        Function {
            prog,
            keepalive,
            shapes,
        }
    }

    /// A new reference as a `PtFunction*`.
    pub fn into_handle(self) -> *const Function {
        Arc::into_raw(Arc::new(self))
    }
}

pub struct PtSession {
    f: Arc<Function>,
    session: Session,
    err: CString,
}

#[repr(C)]
pub struct PtFunctionApi {
    pub abi_version: u32,
    pub api_version: u32,
    pub function_incref: unsafe extern "C" fn(*const Function),
    pub function_decref: unsafe extern "C" fn(*const Function),
    pub n_inputs: unsafe extern "C" fn(*const Function) -> usize,
    pub n_outputs: unsafe extern "C" fn(*const Function) -> usize,
    pub input_type: unsafe extern "C" fn(*const Function, usize, *mut PtTypeInfo) -> i32,
    pub output_type: unsafe extern "C" fn(*const Function, usize, *mut PtTypeInfo) -> i32,
    pub session_new: unsafe extern "C" fn(*const Function) -> *mut PtSession,
    pub session_free: unsafe extern "C" fn(*mut PtSession),
    pub call: unsafe extern "C" fn(*mut PtSession, *const PtArray, *mut PtArray) -> i32,
    pub last_error: unsafe extern "C" fn(*const PtSession) -> *const c_char,
    pub buffer_decref: unsafe extern "C" fn(*mut PtBuffer),
    pub seed: unsafe extern "C" fn(*mut PtSession, u64),
    pub set_parallel: unsafe extern "C" fn(*mut PtSession, i32),
}

// SAFETY contract of every entry point below (pytensor_llvm.h; the caller's
// responsibility, not checkable here):
// - `f` is a live `PtFunction*` from `c_function()` / `function_incref`, not yet released.
// - `s` is a live `PtSession*` from `session_new`, used by one thread at a time.
// - out-pointers point to writable memory of the right type.
// Within that contract the functions are sound: everything `call` reads through the
// arrays it gets is validated first. What can't be validated, and so is the caller's
// to get right, is that the memory the arrays describe is there (see `run`).

unsafe extern "C" fn function_incref(f: *const Function) {
    // SAFETY: `f` is live (contract).
    unsafe { Arc::increment_strong_count(f) };
}

unsafe extern "C" fn function_decref(f: *const Function) {
    // SAFETY: the caller releases a reference it holds.
    unsafe { Arc::decrement_strong_count(f) };
}

unsafe extern "C" fn n_inputs(f: *const Function) -> usize {
    // SAFETY: `f` is live.
    unsafe { (*f).prog.n_inputs() }
}

unsafe extern "C" fn n_outputs(f: *const Function) -> usize {
    // SAFETY: `f` is live.
    unsafe { (*f).prog.n_outputs() }
}

unsafe extern "C" fn input_type(f: *const Function, i: usize, out: *mut PtTypeInfo) -> i32 {
    // SAFETY: `f` is live.
    let f = unsafe { &*f };
    if i >= f.prog.n_inputs() {
        return 1;
    }
    let sh = &f.shapes[i];
    // SAFETY: `out` is writable; the shape lives as long as `f`.
    unsafe {
        *out = PtTypeInfo {
            dtype: f.prog.inputs[i].dtype.code(),
            ndim: sh.len() as i32,
            static_shape: sh.as_ptr(),
        }
    };
    0
}

unsafe extern "C" fn output_type(f: *const Function, o: usize, out: *mut PtTypeInfo) -> i32 {
    // SAFETY: `f` is live.
    let f = unsafe { &*f };
    if o >= f.prog.n_outputs() {
        return 1;
    }
    let sh = &f.shapes[f.prog.n_inputs() + o];
    // SAFETY: `out` is writable; the shape lives as long as `f`.
    unsafe {
        *out = PtTypeInfo {
            dtype: f.prog.outputs[o].dtype.code(),
            ndim: sh.len() as i32,
            static_shape: sh.as_ptr(),
        }
    };
    0
}

unsafe extern "C" fn session_new(f: *const Function) -> *mut PtSession {
    // SAFETY: `f` is live; the session owns the new reference.
    let f = unsafe {
        Arc::increment_strong_count(f);
        Arc::from_raw(f)
    };
    let s = PtSession {
        f,
        session: Session::new(),
        err: CString::default(),
    };
    Box::into_raw(Box::new(s))
}

unsafe extern "C" fn session_free(s: *mut PtSession) {
    if !s.is_null() {
        // SAFETY: `s` came from session_new and is freed once.
        drop(unsafe { Box::from_raw(s) });
    }
}

/// Run the function: see `call` in pytensor_llvm.h.
unsafe extern "C" fn call(s: *mut PtSession, inputs: *const PtArray, outputs: *mut PtArray) -> i32 {
    // SAFETY: `s` is live and used by this thread only (contract).
    let s = unsafe { &mut *s };
    let res = catch_unwind(AssertUnwindSafe(|| run(s, inputs, outputs)));
    let msg = match res {
        Ok(Ok(())) => return 0,
        Ok(Err(m)) => m,
        Err(_) => "internal error: panic in pytensor_llvm".to_string(),
    };
    s.err = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    1
}

/// The header of a caller array against the function's type, for errors that say what
/// was expected. (The views check the rest.)
fn check_header(what: &str, k: usize, a: &PtArray, nd: usize, dtype: DType) -> Result<(), String> {
    if a.ndim < 0 || a.ndim as usize != nd {
        return Err(format!("{what} {k}: expected {nd} axes, got {}", a.ndim));
    }
    if nd > 0 && (a.shape.is_null() || a.strides.is_null()) {
        return Err(format!("{what} {k}: shape and strides must not be NULL"));
    }
    if a.dtype != dtype.code() {
        return Err(format!(
            "{what} {k}: expected dtype {} (code {}), got code {}",
            dtype.numpy_name(),
            dtype.code(),
            a.dtype
        ));
    }
    Ok(())
}

fn run(s: &mut PtSession, inputs: *const PtArray, outputs: *mut PtArray) -> Result<(), String> {
    let prog = &*s.f.prog;
    let (n_in, n_out) = (prog.n_inputs(), prog.n_outputs());
    if (inputs.is_null() && n_in > 0) || (outputs.is_null() && n_out > 0) {
        return Err("inputs and outputs must not be NULL".into());
    }
    let inputs: &[PtArray] = match n_in {
        0 => &[],
        // SAFETY: `inputs` holds n_inputs arrays, not written during the call (the
        // contract of `call`).
        _ => unsafe { std::slice::from_raw_parts(inputs, n_in) },
    };
    let outputs: &mut [PtArray] = match n_out {
        0 => &mut [],
        // SAFETY: `outputs` holds n_outputs arrays, ours during the call (the contract
        // of `call`).
        _ => unsafe { std::slice::from_raw_parts_mut(outputs, n_out) },
    };
    let ins = inputs.iter().enumerate().map(|(k, a)| {
        let t = &prog.inputs[k];
        check_header("input", k, a, t.ndim(), t.dtype)?;
        // SAFETY: the header was just checked against the function's type; what the
        // constructor can't check is the caller's side of the contract of `call`: the
        // shape and strides arrays have `ndim` entries, and the memory they describe
        // exists and isn't written during the call.
        unsafe { PtArrayRef::from_raw_checked(a) }.map_err(|e| format!("input {k}: {e}"))
    });
    let outs = outputs.iter().enumerate().map(|(o, a)| {
        let t = &prog.outputs[o];
        check_header("output", o, a, t.ndim(), t.dtype)?;
        if a.data.is_null() {
            return Ok(Destination::New);
        }
        // SAFETY: as for the inputs, for memory to write: the caller's side of the
        // contract is that it exists and that nobody else uses it during the call.
        // That it overlaps none of the arrays of this call is checked (by `call`)
        // before anything is written.
        unsafe { PtArrayMut::from_raw_checked(a) }
            .map(Destination::Into)
            .map_err(|e| format!("provided output {o}: {e}"))
    });
    let results = prog
        .call(&mut s.session, ins, outs)
        .map_err(|e| e.message)?;
    // outputs we hand over, taken back if a later one fails
    let mut taken: Vec<usize> = vec![];
    let mut r = Ok(());
    for (o, out) in results {
        // SAFETY: shape and strides point to caller arrays of length ndim, ours during
        // the call (the contract of `call`; the header was checked above), and no
        // memory is provided.
        r = unsafe { OutSlot::from_raw(&mut outputs[o]) }
            .and_then(|slot| slot.set(out))
            .map_err(|e| format!("output {o}: {e}"));
        if r.is_err() {
            break;
        }
        taken.push(o);
    }
    if r.is_err() {
        for o in taken {
            // SAFETY: the reference we just put there; take it back.
            drop(unsafe { OwnedBuf::from_raw(outputs[o].owner) });
            outputs[o].data = std::ptr::null_mut();
            outputs[o].owner = std::ptr::null_mut();
        }
    }
    r
}

unsafe extern "C" fn last_error(s: *const PtSession) -> *const c_char {
    // SAFETY: `s` is live; the string lives until the next call on `s`.
    unsafe { (*s).err.as_ptr() }
}

unsafe extern "C" fn seed(s: *mut PtSession, seed: u64) {
    // SAFETY: `s` is live and used by this thread only (contract).
    unsafe { (*s).session.ctx.seed(seed) }
}

unsafe extern "C" fn set_parallel(s: *mut PtSession, on: i32) {
    // SAFETY: `s` is live and used by this thread only (contract).
    unsafe { (*s).session.ctx.parallel = on != 0 }
}

unsafe extern "C" fn buffer_decref(b: *mut PtBuffer) {
    if !b.is_null() {
        // SAFETY: the caller releases a reference it got from `call`.
        drop(unsafe { OwnedBuf::from_raw(b) })
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
    session_new,
    session_free,
    call,
    last_error,
    buffer_decref,
    seed,
    set_parallel,
};
