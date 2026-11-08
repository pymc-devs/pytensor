//! Helpers for precompiled kernels.

use super::op::ArrayRef;
use crate::dtype::DType;

/// Read one element as (f64, i64).
///
/// # Safety
/// `p` points to a valid, aligned element of `dt`.
pub unsafe fn read_scalar(p: *const u8, dt: DType) -> (f64, i64) {
    // SAFETY: the function's contract.
    unsafe {
        match dt {
            DType::Float64 => {
                let v = *(p as *const f64);
                (v, if v.is_finite() { v as i64 } else { 0 })
            }
            DType::Float32 => {
                let v = *(p as *const f32) as f64;
                (v, if v.is_finite() { v as i64 } else { 0 })
            }
            DType::Int64 => {
                let v = *(p as *const i64);
                (v as f64, v)
            }
            DType::Int32 => {
                let v = *(p as *const i32) as i64;
                (v as f64, v)
            }
            DType::Int16 => {
                let v = *(p as *const i16) as i64;
                (v as f64, v)
            }
            DType::Int8 => {
                let v = *(p as *const i8) as i64;
                (v as f64, v)
            }
            DType::Uint8 | DType::Bool => {
                let v = *p as i64;
                (v as f64, v)
            }
        }
    }
}

/// Copy the elements of `src` into memory with the same shape and `dst_strides`.
///
/// # Safety
/// `src` is a valid array of elements of `itemsize` bytes; `dst` with `dst_strides`
/// describes writable memory of the same shape that doesn't overlap `src`.
pub unsafe fn copy_strided(src: ArrayRef<'_>, dst: *mut u8, dst_strides: &[i64], itemsize: usize) {
    let nd = src.ndim();
    if src.shape.contains(&0) {
        return;
    }
    let mut idx = vec![0i64; nd];
    loop {
        let (mut so, mut d) = (0isize, 0isize);
        for a in 0..nd {
            so += (idx[a] * src.strides[a]) as isize;
            d += (idx[a] * dst_strides[a]) as isize;
        }
        // SAFETY: the offsets stay inside both arrays (index < shape on every axis).
        unsafe { std::ptr::copy_nonoverlapping(src.data.offset(so), dst.offset(d), itemsize) };
        let mut a = nd;
        loop {
            if a == 0 {
                return;
            }
            a -= 1;
            idx[a] += 1;
            if idx[a] < src.shape[a] {
                break;
            }
            idx[a] = 0;
        }
    }
}
