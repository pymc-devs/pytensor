//! Rust kernels, with the kernel ABI:
//! `int32 f(PtCtx*, PtArray* args /* inputs, outputs */, const void* config, void* frame)`.
//!
//! A kernel's config holds its parameters and the mode of each output, which Python
//! chooses per node (the kernel never infers it from the arguments).

use crate::abi::{checked_nbytes, PtArray};
use crate::runtime::{PtCtx};
use crate::dtype::DType;

pub use crate::abi::{LegacyFn, OutputMode};

/// Inputs and outputs of one call.
pub struct Args<'a> {
    arrays: &'a mut [PtArray],
    n_in: usize,
}

impl<'a> Args<'a> {
    /// # Safety
    /// `args` holds `n_in + n_out` arrays: valid inputs, outputs as their modes say
    /// (provided: valid writable memory of the output's shape; otherwise writable
    /// shape/strides storage of the output's ndim).
    pub unsafe fn new(args: *mut PtArray, n_in: usize, n_out: usize) -> Self {
        // SAFETY: the function's contract.
        Args { arrays: unsafe { std::slice::from_raw_parts_mut(args, n_in + n_out) }, n_in }
    }

    pub fn input(&self, i: usize) -> &PtArray {
        &self.arrays[i]
    }

    pub fn shape(&self, i: usize) -> &[i64] {
        // SAFETY: inputs are valid arrays (the constructor's contract).
        unsafe { self.arrays[i].shape() }
    }

    pub fn strides(&self, i: usize) -> &[i64] {
        // SAFETY: as above.
        unsafe { self.arrays[i].strides() }
    }

    pub fn out(&mut self, o: usize) -> &mut PtArray {
        let k = self.n_in + o;
        &mut self.arrays[k]
    }

    pub fn raw(&mut self) -> *mut PtArray {
        self.arrays.as_mut_ptr()
    }

    /// Memory for output `o` of `shape`: the provided memory (whose shape must match)
    /// or a fresh C-contiguous buffer. Returns (data, byte strides).
    pub fn output(&mut self, ctx: &mut PtCtx, o: usize, mode: OutputMode, dtype: DType, shape: &[i64]) -> Result<(*mut u8, Vec<i64>), String> {
        let a = self.out(o);
        match mode {
            OutputMode::Provided => {
                // SAFETY: provided outputs are valid arrays (the constructor's contract).
                let (have, strides) = unsafe { (a.shape(), a.strides().to_vec()) };
                if have != shape {
                    return Err(format!("output {o} has shape {have:?}, expected {shape:?}"));
                }
                Ok((a.data, strides))
            }
            OutputMode::Allocated | OutputMode::View => {
                let n = checked_nbytes(shape, dtype.itemsize()).ok_or_else(|| crate::abi::bad_size("output", shape))?;
                // SAFETY: ctx is the live context.
                let (data, owner) = unsafe { PtCtx::alloc(ctx, n) };
                if data.is_null() {
                    return Err(format!("could not allocate {n} bytes"));
                }
                let mut strides = vec![0i64; shape.len()];
                let mut acc = dtype.itemsize() as i64;
                for d in (0..shape.len()).rev() {
                    strides[d] = acc;
                    acc *= shape[d].max(1);
                }
                // SAFETY: the output has shape/strides storage of its ndim.
                unsafe {
                    a.shape_mut().copy_from_slice(shape);
                    a.strides_mut().copy_from_slice(&strides);
                }
                a.data = data;
                a.owner = owner;
                Ok((data, strides))
            }
        }
    }
}

fn finish(ctx: *mut PtCtx, r: Result<(), String>) -> i32 {
    match r {
        Ok(()) => 0,
        // SAFETY: ctx is the live context.
        Err(m) => unsafe { PtCtx::fail(ctx, 1, &m) },
    }
}

// ------------------------------------------------------------------ DimShuffle

pub struct DimShuffleConfig {
    /// output axis -> input axis, or None for a new length-1 axis
    pub order: Vec<Option<usize>>,
}

/// # Safety
/// The kernel ABI; `config` is a `DimShuffleConfig`; one input, one view output.
pub unsafe extern "C-unwind" fn k_dimshuffle(ctx: *mut PtCtx, args: *mut PtArray, config: *const u8, _frame: *mut u8) -> i32 {
    // SAFETY: the function's contract.
    let (e, mut a) = unsafe { (&*(config as *const DimShuffleConfig), Args::new(args, 1, 1)) };
    let (shape, strides): (Vec<i64>, Vec<i64>) = e
        .order
        .iter()
        .map(|o| match o {
            Some(ax) => (a.shape(0)[*ax], a.strides(0)[*ax]),
            None => (1, 0),
        })
        .unzip();
    let (data, owner) = (a.input(0).data, a.input(0).owner);
    let out = a.out(0);
    if out.ndim as usize != shape.len() {
        return finish(ctx, Err("DimShuffle: output ndim mismatch".into()));
    }
    // SAFETY: the output has shape/strides storage of its ndim (checked above).
    unsafe {
        out.shape_mut().copy_from_slice(&shape);
        out.strides_mut().copy_from_slice(&strides);
    }
    out.data = data;
    out.owner = owner;
    0
}

// ------------------------------------------------------------------ DeepCopy

pub struct CopyConfig {
    pub mode: OutputMode,
    pub dtype: DType,
}

/// # Safety
/// The kernel ABI; `config` is a `CopyConfig`.
pub unsafe extern "C-unwind" fn k_copy(ctx: *mut PtCtx, args: *mut PtArray, config: *const u8, _frame: *mut u8) -> i32 {
    // SAFETY: the function's contract.
    let (e, mut a, c) = unsafe { (&*(config as *const CopyConfig), Args::new(args, 1, 1), &mut *ctx) };
    let r = (|| {
        let shape = a.shape(0).to_vec();
        let src_strides = a.strides(0).to_vec();
        let src = a.input(0).data;
        let (dst, dst_strides) = a.output(c, 0, e.mode, e.dtype, &shape)?;
        // SAFETY: a valid input and output memory of the same shape (provided outputs
        // don't overlap inputs; fresh ones are new).
        unsafe { super::program::copy_strided(src, &shape, &src_strides, dst, &dst_strides, e.dtype.itemsize()) };
        Ok(())
    })();
    finish(ctx, r)
}

// ------------------------------------------------------------------ Dot

pub struct DotConfig {
    pub mode: OutputMode,
    pub dtype: DType,
}

/// numpy dot of 1-d/2-d float operands, via faer.
///
/// # Safety
/// The kernel ABI; `config` is a `DotConfig`.
pub unsafe extern "C-unwind" fn k_dot(ctx: *mut PtCtx, args: *mut PtArray, config: *const u8, _frame: *mut u8) -> i32 {
    // SAFETY: the function's contract.
    let (e, mut a, c) = unsafe { (&*(config as *const DotConfig), Args::new(args, 2, 1), &mut *ctx) };
    let r = (|| {
        let (sa, sb) = (a.shape(0).to_vec(), a.shape(1).to_vec());
        if sa.last() != sb.first() {
            return Err(format!("Dot: shapes {sa:?} and {sb:?} are not aligned"));
        }
        let mut shape = Vec::new();
        if sa.len() == 2 {
            shape.push(sa[0]);
        }
        if sb.len() == 2 {
            shape.push(sb[1]);
        }
        a.output(c, 0, e.mode, e.dtype, &shape)?;
        let f = match e.dtype {
            DType::Float64 => crate::legacy_kernels::ptl_matmul_f64,
            DType::Float32 => crate::legacy_kernels::ptl_matmul_f32,
            _ => return Err("Dot: unsupported dtype".into()),
        };
        // SAFETY: args = [a, b, out] with the output memory set above; the matmul kernel
        // reads data/shape/strides only.
        if unsafe { f(c, a.raw(), std::ptr::null()) } != 0 {
            return Err(c.error.take().map(|(_, m)| m).unwrap_or_else(|| "Dot failed".into()));
        }
        Ok(())
    })();
    finish(ctx, r)
}

// ------------------------------------------------------------------ adapters

/// Runs a legacy kernel (`abi::LegacyFn`): allocates the output (shaped like input
/// `shape_from`) unless it is provided, then calls the kernel with its parameters.
pub struct LegacyConfig {
    pub f: LegacyFn,
    pub params: Vec<i64>,
    pub n_in: usize,
    pub shape_from: usize,
    pub mode: OutputMode,
    pub dtype: DType,
}

/// # Safety
/// The kernel ABI; `config` is a `LegacyConfig` whose function expects these arguments.
pub unsafe extern "C-unwind" fn k_legacy(ctx: *mut PtCtx, args: *mut PtArray, config: *const u8, _frame: *mut u8) -> i32 {
    // SAFETY: the function's contract.
    let (e, mut a, c) = unsafe { (&*(config as *const LegacyConfig), Args::new(args, (*(config as *const LegacyConfig)).n_in, 1), &mut *ctx) };
    let shape = a.shape(e.shape_from).to_vec();
    if let Err(m) = a.output(c, 0, e.mode, e.dtype, &shape) {
        return finish(ctx, Err(m));
    }
    // SAFETY: the inputs are valid and the output now has memory of the input's shape,
    // which is what the wrapped kernel expects.
    unsafe { (e.f)(ctx, a.raw(), e.params.as_ptr() as *const std::ffi::c_void) }
}
