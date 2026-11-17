//! Constant arrays owned by programs.

use crate::dtype::DType;
use zerocopy::{FromBytes, Immutable, IntoBytes};

#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

/// Constant data: native byte order, packed in `order` (outermost axis first; empty is
/// C), 64-byte aligned. Equal buffers are shared (a matrix and its transpose).
pub struct ConstData {
    pub dtype: DType,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    buf: std::sync::Arc<Vec<Align64>>,
}

#[derive(Default)]
pub struct ConstBuffers(std::collections::HashMap<u64, Vec<std::sync::Arc<Vec<Align64>>>>);

impl ConstData {
    pub fn new(
        dtype: DType,
        shape: Vec<i64>,
        order: &[usize],
        bytes: &[u8],
        shared: &mut ConstBuffers,
    ) -> Result<Self, String> {
        let n = crate::abi::checked_nbytes(&shape, dtype.itemsize())
            .ok_or_else(|| crate::abi::bad_size("constant", &shape))?;
        if n != bytes.len() {
            return Err(format!(
                "constant of shape {shape:?} needs {n} bytes, got {}",
                bytes.len()
            ));
        }
        let key = std::hash::BuildHasher::hash_one(
            &std::hash::BuildHasherDefault::<std::hash::DefaultHasher>::default(),
            bytes,
        );
        let same = shared.0.entry(key).or_default();
        let buf = match same.iter().find(|b| &b.as_bytes()[..n] == bytes) {
            Some(buf) => buf.clone(),
            None => {
                let mut buf = vec![Align64([0; 64]); n.div_ceil(64).max(1)];
                buf.as_mut_bytes()[..n].copy_from_slice(bytes);
                same.push(std::sync::Arc::new(buf));
                same.last().unwrap().clone()
            }
        };
        let strides = super::program::packed_strides(&shape, dtype.itemsize() as i64, order);
        Ok(ConstData {
            dtype,
            shape,
            strides,
            buf,
        })
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
            DType::Uint16 => {
                let v = *(p as *const u16) as i64;
                (v as f64, v)
            }
            DType::Uint32 => {
                let v = *(p as *const u32) as i64;
                (v as f64, v)
            }
            DType::Uint64 => {
                // the i64 wraps above i64::MAX
                let v = *(p as *const u64);
                (v as f64, v as i64)
            }
        }
    }
}
