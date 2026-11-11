//! Constant arrays owned by programs.

use crate::dtype::DType;
use zerocopy::{FromBytes, Immutable, IntoBytes};

#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

/// Constant data: native byte order, C-contiguous, 64-byte aligned.
pub struct ConstData {
    pub dtype: DType,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    buf: Vec<Align64>,
}

impl ConstData {
    pub fn new(dtype: DType, shape: Vec<i64>, bytes: &[u8]) -> Result<Self, String> {
        let n = crate::abi::checked_nbytes(&shape, dtype.itemsize()).ok_or_else(|| crate::abi::bad_size("constant", &shape))?;
        if n != bytes.len() {
            return Err(format!("constant of shape {shape:?} needs {n} bytes, got {}", bytes.len()));
        }
        let mut buf = vec![Align64([0; 64]); n.div_ceil(64).max(1)];
        buf.as_mut_bytes()[..n].copy_from_slice(bytes);
        let mut strides = vec![0i64; shape.len()];
        let mut acc = dtype.itemsize() as i64;
        for d in (0..shape.len()).rev() {
            strides[d] = acc;
            acc *= shape[d].max(1);
        }
        Ok(ConstData { dtype, shape, strides, buf })
    }

    pub fn data(&self) -> *const u8 {
        self.buf.as_bytes().as_ptr()
    }
}

impl std::fmt::Debug for ConstData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConstData({:?}, {:?})", self.dtype, self.shape)
    }
}

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
