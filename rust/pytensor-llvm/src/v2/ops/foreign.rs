//! Kernels compiled elsewhere (the numba fallback), with the old kernel ABI:
//! `int32 f(PtCtx*, PtArray* args, void* params)`, all outputs kernel-allocated.

use crate::abi::{KernelFn, PtArray, PtBuffer, PtCtx};
use crate::v2::index::Idx;
use crate::v2::op::{Kernel, KernelArgs, TensorOp, R};
use crate::v2::types::{OutputAlloc, OutputIdx, Signature};

pub struct Foreign {
    pub addr: usize,
    pub desc: String,
    /// keeps the compiled code alive (e.g. the numba cfunc)
    pub keepalive: Option<pyo3::Py<pyo3::PyAny>>,
}

impl std::fmt::Debug for Foreign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Foreign({})", self.desc)
    }
}

impl TensorOp for Foreign {
    fn kernel(&self, _sig: &Signature) -> R<Option<Box<dyn Kernel>>> {
        Ok(Some(Box::new(ForeignKernel { addr: self.addr, desc: self.desc.clone() })))
    }
}

struct ForeignKernel {
    addr: usize,
    desc: String,
}

impl Kernel for ForeignKernel {
    fn call(&self, ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()> {
        let sig = args.sig().clone();
        let n_in = sig.inputs.len();
        // the foreign code allocates every output: give it fresh slots
        let mut storage: Vec<Vec<i64>> = sig.outputs.iter().map(|o| vec![0; 2 * o.ty.ndim().max(1)]).collect();
        let mut arrays: Vec<PtArray> = args.raw()[..n_in].to_vec();
        for (o, os) in sig.outputs.iter().enumerate() {
            let st = storage[o].as_mut_ptr();
            let nd = os.ty.ndim();
            // SAFETY: `storage[o]` holds 2 * ndim entries.
            let strides = unsafe { st.add(nd) };
            arrays.push(PtArray { data: std::ptr::null_mut(), shape: st, strides, owner: std::ptr::null_mut(), ndim: nd as i64 });
        }
        // SAFETY: `addr` is a kernel with the ABI above (the numba fallback wrapper);
        // inputs are valid arrays and every output has shape/strides storage.
        let f: KernelFn = unsafe { std::mem::transmute(self.addr) };
        // SAFETY: as above.
        let rc = unsafe { f(ctx, arrays.as_mut_ptr(), std::ptr::null()) };
        if rc != 0 {
            let msg = ctx.error.take().map(|(_, m)| m).unwrap_or_else(|| "the numba implementation raised an exception".into());
            for a in &arrays[n_in..] {
                // SAFETY: references the foreign code handed over (or NULL).
                unsafe { crate::v2::unit::ptl_v2_decref(a.owner) };
            }
            return Err(format!("{}: {msg}", self.desc));
        }
        for (k, a) in arrays[n_in..].iter().enumerate() {
            let o = OutputIdx::new(k);
            // SAFETY: the foreign code set shape and strides of every output.
            let (shape, strides) = unsafe { (a.shape().to_vec(), a.strides().to_vec()) };
            match sig.outputs[o].alloc {
                OutputAlloc::Caller => {
                    let isz = sig.output(o).dtype.itemsize();
                    let (dst, dst_strides) = args.output(ctx, o, &shape)?;
                    let src = crate::v2::op::ArrayRef { data: a.data, shape: &shape, strides: &strides, owner: a.owner };
                    // SAFETY: a fresh, valid array from the foreign code and provided
                    // memory of the same shape.
                    unsafe {
                        crate::v2::kernels::copy_strided(src, dst, &dst_strides, isz);
                        PtBuffer::decref(a.owner);
                    }
                }
                OutputAlloc::Callee | OutputAlloc::ViewOf(_) => {
                    args.set_output(o, a.data, &shape, &strides, a.owner);
                }
            }
        }
        Ok(())
    }
}
