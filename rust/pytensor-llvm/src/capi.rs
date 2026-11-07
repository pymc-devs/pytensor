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

unsafe extern "C" fn function_incref(f: *const Program) {
    Arc::increment_strong_count(f);
}

unsafe extern "C" fn function_decref(f: *const Program) {
    Arc::decrement_strong_count(f);
}

unsafe extern "C" fn n_inputs(f: *const Program) -> usize {
    (*f).n_inputs()
}

unsafe extern "C" fn n_outputs(f: *const Program) -> usize {
    (*f).n_outputs()
}

unsafe fn fill_type(dt: crate::dtype::DType, shape: &[i64], out: *mut PtTypeInfo) {
    *out = PtTypeInfo { dtype: dt.code(), ndim: shape.len() as i32, static_shape: shape.as_ptr() };
}

unsafe extern "C" fn input_type(f: *const Program, i: usize, out: *mut PtTypeInfo) -> i32 {
    if i >= (*f).n_inputs() {
        return 1;
    }
    let (dt, sh) = (*f).input_type(i);
    fill_type(dt, sh, out);
    0
}

unsafe extern "C" fn output_type(f: *const Program, i: usize, out: *mut PtTypeInfo) -> i32 {
    if i >= (*f).n_outputs() {
        return 1;
    }
    let (dt, sh) = (*f).output_type(i);
    fill_type(dt, sh, out);
    0
}

unsafe extern "C" fn state_new(f: *const Program) -> *mut PtCallState {
    Arc::increment_strong_count(f);
    let prog = Arc::from_raw(f);
    let st = prog.new_state();
    Box::into_raw(Box::new(PtCallState { prog, st, err: CString::default() }))
}

unsafe extern "C" fn state_free(s: *mut PtCallState) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

unsafe extern "C" fn call(s: *mut PtCallState, inputs: *const PtArray, outputs: *mut PtArray) -> i32 {
    let s = &mut *s;
    let res = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        let prog = &*s.prog;
        for k in 0..prog.n_inputs() {
            let a = &*inputs.add(k);
            prog.set_input_raw(&mut s.st, k, a.data as *mut u8, a.shape(), a.strides())?;
        }
        let outs = std::slice::from_raw_parts_mut(outputs, prog.n_outputs());
        prog.execute(&mut s.st, Some(outs))?;
        for (o, r) in outs.iter_mut().zip(&s.st.outs) {
            if o.data.is_null() {
                o.data = r.data;
                o.owner = r.owner;
                o.ndim = r.ndim as i64;
                o.shape_mut().copy_from_slice(r.shape());
                o.strides_mut().copy_from_slice(r.strides());
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
    (*s).err.as_ptr()
}

unsafe extern "C" fn buffer_decref(b: *mut PtBuffer) {
    if !b.is_null() {
        PtBuffer::decref(b)
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
