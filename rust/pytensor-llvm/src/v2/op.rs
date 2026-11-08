//! What an op implementation provides.
//!
//! An op is a Rust type implementing `TensorOp`. It can offer two implementations:
//!
//! * a `Kernel`: precompiled Rust, called directly by the interpreter (and through one
//!   generic trampoline from generated code);
//! * a `Generate`: LLVM IR emitted into a compiled unit with the internal convention.
//!
//! The interpreter uses the kernel if there is one, and otherwise compiles the node as a
//! one-node unit. A compiled unit inlines everything that can be generated and calls the
//! kernels of the rest.

use super::types::{InputIdx, OutputAlloc, OutputIdx, Signature};
use super::unit::{ArrVal, EnvBuilder, UnitCx};
use crate::abi::{PtArray, PtBuffer, PtCtx};
use crate::v2::index::{Idx, IndexVec};
use sha2::Digest;

pub type R<T> = Result<T, String>;

pub trait TensorOp: Send + Sync + std::fmt::Debug {
    /// Check that a node of this op with these types is supported.
    fn check(&self, _sig: &Signature) -> R<()> {
        Ok(())
    }

    /// A precompiled implementation for these types, if there is one.
    fn kernel(&self, _sig: &Signature) -> R<Option<Box<dyn Kernel>>> {
        Ok(None)
    }

    /// A generated implementation, if there is one.
    fn generate(&self) -> Option<&dyn Generate> {
        None
    }
}

/// Precompiled implementation.
pub trait Kernel: Send + Sync {
    fn call(&self, ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()>;
}

/// Generated implementation.
pub trait Generate {
    /// Emit the node. `provided[o]` is the memory for `Caller` outputs; `Callee` outputs
    /// are allocated with `cx.alloc_output`, and `ViewOf` outputs are built from inputs.
    fn emit<'c>(
        &self,
        cx: &mut UnitCx<'c>,
        sig: &Signature,
        inputs: &IndexVec<InputIdx, ArrVal<'c>>,
        provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>>;

    /// Everything the generated code depends on besides the signature (cache key).
    fn key(&self, k: &mut Key);

    /// Register the callees and constants `emit` will use, in the order it uses them.
    /// Only ops that emit nested graphs need this.
    fn collect_env(&self, _env: &mut EnvBuilder, _sig: &Signature) -> R<()> {
        Ok(())
    }
}

/// Cache key builder.
pub struct Key(sha2::Sha256);

impl Key {
    pub fn new() -> Self {
        Key(sha2::Sha256::new())
    }

    /// Mark the start of something of type `T` (an op, a scalar op).
    pub fn tag<T: ?Sized>(&mut self) {
        self.str(std::any::type_name::<T>());
    }

    pub fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.update(s.as_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.0.update(v.to_le_bytes());
    }

    pub fn i64(&mut self, v: i64) {
        self.0.update(v.to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        self.0.update(v.to_bits().to_le_bytes());
    }

    pub fn bool(&mut self, v: bool) {
        self.0.update([v as u8]);
    }

    pub fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.0.update(b);
    }

    pub fn debug<T: std::fmt::Debug>(&mut self, v: &T) {
        self.str(&format!("{v:?}"));
    }

    pub fn finish(self) -> String {
        hex::encode(&self.0.finalize()[..16])
    }
}

impl Default for Key {
    fn default() -> Self {
        Self::new()
    }
}

/// A borrowed input array.
#[derive(Clone, Copy)]
pub struct ArrayRef<'a> {
    pub data: *const u8,
    pub shape: &'a [i64],
    pub strides: &'a [i64],
    pub owner: *mut PtBuffer,
}

impl ArrayRef<'_> {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> i64 {
        self.shape.iter().product()
    }
}

/// The arguments of a kernel call: inputs then outputs, as in the ABI.
pub struct KernelArgs<'a> {
    sig: &'a Signature,
    arrays: &'a mut [PtArray],
}

impl<'a> KernelArgs<'a> {
    /// # Safety
    /// `arrays` holds `sig.inputs.len() + sig.outputs.len()` entries. Inputs describe
    /// valid arrays of their signature type; `Caller` outputs describe valid, writable
    /// memory of their (static) type that doesn't overlap any input; every output has
    /// writable shape and strides storage of its ndim.
    pub unsafe fn new(sig: &'a Signature, arrays: &'a mut [PtArray]) -> Self {
        debug_assert_eq!(arrays.len(), sig.inputs.len() + sig.outputs.len());
        KernelArgs { sig, arrays }
    }

    pub fn sig(&self) -> &Signature {
        self.sig
    }

    pub fn n_inputs(&self) -> usize {
        self.sig.inputs.len()
    }

    pub fn input(&self, i: InputIdx) -> ArrayRef<'_> {
        let a = &self.arrays[i.index()];
        // SAFETY: the constructor's contract.
        unsafe { ArrayRef { data: a.data, shape: a.shape(), strides: a.strides(), owner: a.owner } }
    }

    /// Raw access for kernels that pass the ABI on (the numba fallback).
    pub fn raw(&mut self) -> &mut [PtArray] {
        self.arrays
    }

    pub fn alloc_kind(&self, o: OutputIdx) -> OutputAlloc {
        self.sig.outputs[o].alloc
    }

    fn out(&mut self, o: OutputIdx) -> &mut PtArray {
        let k = self.sig.arg_of_output(o);
        &mut self.arrays[k]
    }

    /// The memory of a `Caller` output, or a fresh C-contiguous buffer of `shape` for
    /// a `Callee` (or `ViewOf`) output. Returns the data pointer and byte strides.
    pub fn output(&mut self, ctx: &mut PtCtx, o: OutputIdx, shape: &[i64]) -> R<(*mut u8, Vec<i64>)> {
        let ty = self.sig.output(o).clone();
        if shape.len() != ty.ndim() {
            return Err(format!("output {} has {} axes, the type says {}", o.0, shape.len(), ty.ndim()));
        }
        for (a, (&n, len)) in shape.iter().zip(&ty.shape).enumerate() {
            if let Some(s) = len.known() {
                if n as u64 != s {
                    return Err(format!("output {} has length {n} on axis {a}, the type says {s}", o.0));
                }
            }
        }
        match self.alloc_kind(o) {
            OutputAlloc::Caller => {
                let a = self.out(o);
                // SAFETY: the constructor's contract (Caller outputs are valid memory).
                let st = unsafe { a.strides() }.to_vec();
                Ok((a.data, st))
            }
            OutputAlloc::Callee | OutputAlloc::ViewOf(_) => {
                let isz = ty.dtype.itemsize();
                let n = crate::abi::checked_nbytes(shape, isz).ok_or_else(|| crate::abi::bad_size("output", shape))?;
                // SAFETY: ctx is a live context.
                let (data, owner) = unsafe { PtCtx::alloc(ctx, n) };
                if data.is_null() {
                    return Err(format!("could not allocate {n} bytes"));
                }
                let strides = super::types::c_strides(shape, isz as i64);
                self.set_output(o, data, shape, &strides, owner);
                Ok((data, strides))
            }
        }
    }

    /// Point an output at memory: a view (owner = the viewed array's owner) or a buffer
    /// whose reference is handed over.
    pub fn set_output(&mut self, o: OutputIdx, data: *mut u8, shape: &[i64], strides: &[i64], owner: *mut PtBuffer) {
        let a = self.out(o);
        // SAFETY: the constructor's contract: shape/strides storage of the output's ndim.
        unsafe {
            a.shape_mut().copy_from_slice(shape);
            a.strides_mut().copy_from_slice(strides);
        }
        a.data = data;
        a.owner = owner;
    }
}
