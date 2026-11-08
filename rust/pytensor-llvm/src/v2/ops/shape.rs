//! Ops that only change how memory is viewed, or copy it.

use crate::abi::PtCtx;
use crate::v2::index::{Idx, IndexVec};
use crate::v2::op::{Generate, Kernel, KernelArgs, Key, TensorOp, R};
use crate::v2::types::{Axis, InputIdx, OutputAlloc, OutputIdx, Signature};
use crate::v2::unit::{ArrVal, UnitCx};

/// PyTensor's DimShuffle: `order[k]` is the input axis of output axis `k`, or None for
/// a new length-1 axis. Input axes not in `order` are dropped (they have length 1).
#[derive(Debug, Clone)]
pub struct DimShuffle {
    pub order: Vec<Option<Axis>>,
}

impl DimShuffle {
    fn dropped(&self, ndim: usize) -> Vec<Axis> {
        (0..ndim).map(Axis::new).filter(|a| !self.order.contains(&Some(*a))).collect()
    }
}

impl TensorOp for DimShuffle {
    fn check(&self, sig: &Signature) -> R<()> {
        let x = sig.input(InputIdx(0));
        if sig.output(OutputIdx(0)).ndim() != self.order.len() {
            return Err("output ndim doesn't match the order".into());
        }
        for a in self.order.iter().flatten() {
            if a.index() >= x.ndim() {
                return Err(format!("axis {} out of range", a.0));
            }
        }
        if self.dropped(x.ndim()).iter().any(|&a| !x.broadcastable(a)) {
            return Err("dropped axes must have static length 1".into());
        }
        if sig.outputs[OutputIdx(0)].alloc != OutputAlloc::ViewOf(InputIdx(0)) {
            return Err("DimShuffle must be declared a view of its input".into());
        }
        Ok(())
    }

    fn kernel(&self, _sig: &Signature) -> R<Option<Box<dyn Kernel>>> {
        Ok(Some(Box::new(self.clone())))
    }

    fn generate(&self) -> Option<&dyn Generate> {
        Some(self)
    }
}

impl Kernel for DimShuffle {
    fn call(&self, _ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()> {
        let x = args.input(InputIdx(0));
        let mut shape = Vec::with_capacity(self.order.len());
        let mut strides = Vec::with_capacity(self.order.len());
        for a in &self.order {
            match a {
                Some(a) => {
                    shape.push(x.shape[a.index()]);
                    strides.push(x.strides[a.index()]);
                }
                None => {
                    shape.push(1);
                    strides.push(0);
                }
            }
        }
        let (data, owner) = (x.data as *mut u8, x.owner);
        args.set_output(OutputIdx(0), data, &shape, &strides, owner);
        Ok(())
    }
}

impl Generate for DimShuffle {
    fn key(&self, k: &mut Key) {
        k.tag::<Self>();
        k.debug(&self.order);
    }

    fn emit<'c>(
        &self,
        cx: &mut UnitCx<'c>,
        _sig: &Signature,
        ins: &IndexVec<InputIdx, ArrVal<'c>>,
        _provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>> {
        let x = &ins[InputIdx(0)];
        let (mut shape, mut strides) = (vec![], vec![]);
        for a in &self.order {
            match a {
                Some(a) => {
                    shape.push(x.shape[a.index()]);
                    strides.push(x.strides[a.index()]);
                }
                None => {
                    shape.push(cx.e.i64c(1));
                    strides.push(cx.e.i64c(0));
                }
            }
        }
        let identity_order = self.order.iter().enumerate().all(|(k, a)| matches!(a, Some(a) if a.index() == k));
        let layout = if identity_order { x.layout } else { None };
        Ok(IndexVec::from_vec(vec![ArrVal { dtype: x.dtype, data: x.data, shape, strides, layout, known: x.known }]))
    }
}

/// PyTensor's DeepCopyOp.
#[derive(Debug, Clone)]
pub struct DeepCopy;

impl TensorOp for DeepCopy {
    fn check(&self, sig: &Signature) -> R<()> {
        if sig.inputs.len() != 1 || sig.outputs.len() != 1 || sig.input(InputIdx(0)).dtype != sig.output(OutputIdx(0)).dtype {
            return Err("DeepCopy copies one array".into());
        }
        Ok(())
    }

    fn kernel(&self, _sig: &Signature) -> R<Option<Box<dyn Kernel>>> {
        Ok(Some(Box::new(DeepCopy)))
    }

    fn generate(&self) -> Option<&dyn Generate> {
        Some(self)
    }
}

impl Kernel for DeepCopy {
    fn call(&self, ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()> {
        let x = args.input(InputIdx(0));
        let shape = x.shape.to_vec();
        let itemsize = args.sig().input(InputIdx(0)).dtype.itemsize();
        let (data, strides) = args.output(ctx, OutputIdx(0), &shape)?;
        let x = args.input(InputIdx(0));
        // SAFETY: `x` is a valid input; the output memory has the same shape and is
        // separate from it (provided outputs don't overlap inputs, fresh ones are new).
        unsafe { crate::v2::kernels::copy_strided(x, data, &strides, itemsize) };
        Ok(())
    }
}

impl Generate for DeepCopy {
    fn key(&self, k: &mut Key) {
        k.tag::<Self>();
    }

    fn emit<'c>(
        &self,
        cx: &mut UnitCx<'c>,
        _sig: &Signature,
        ins: &IndexVec<InputIdx, ArrVal<'c>>,
        provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>> {
        let x = &ins[InputIdx(0)];
        let out = match &provided[OutputIdx(0)] {
            Some(a) => a.clone(),
            None => cx.alloc_output(OutputIdx(0), x.dtype, &x.shape.clone())?,
        };
        super::elemwise::copy_array(cx, x, &out)?;
        Ok(IndexVec::from_vec(vec![out]))
    }
}
