//! Linear algebra kernels.

use crate::abi::PtCtx;
use crate::dtype::DType;
use crate::v2::op::{Kernel, KernelArgs, TensorOp, R};
use crate::v2::types::{InputIdx, OutputIdx, Signature};

/// PyTensor's Dot on 1-d / 2-d float operands (numpy dot semantics), via faer.
#[derive(Debug, Clone)]
pub struct Dot;

impl TensorOp for Dot {
    fn check(&self, sig: &Signature) -> R<()> {
        let (a, b) = (sig.input(InputIdx(0)), sig.input(InputIdx(1)));
        let out = sig.output(OutputIdx(0));
        if !(1..=2).contains(&a.ndim()) || !(1..=2).contains(&b.ndim()) {
            return Err("Dot needs 1-d or 2-d operands".into());
        }
        if a.dtype != b.dtype || a.dtype != out.dtype || !matches!(a.dtype, DType::Float64 | DType::Float32) {
            return Err(format!("Dot of {:?} and {:?} into {:?}", a.dtype, b.dtype, out.dtype));
        }
        Ok(())
    }

    fn kernel(&self, sig: &Signature) -> R<Option<Box<dyn Kernel>>> {
        Ok(Some(Box::new(DotKernel { dtype: sig.input(InputIdx(0)).dtype })))
    }
}

struct DotKernel {
    dtype: DType,
}

impl Kernel for DotKernel {
    fn call(&self, ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()> {
        let (a, b) = (args.input(InputIdx(0)), args.input(InputIdx(1)));
        let k_a = *a.shape.last().unwrap();
        let k_b = b.shape[0];
        if k_a != k_b {
            return Err(format!("Dot: shapes {:?} and {:?} are not aligned", a.shape, b.shape));
        }
        let mut shape = Vec::new();
        if a.ndim() == 2 {
            shape.push(a.shape[0]);
        }
        if b.ndim() == 2 {
            shape.push(b.shape[1]);
        }
        args.output(ctx, OutputIdx(0), &shape)?;
        let f = match self.dtype {
            DType::Float64 => crate::native::ptl_matmul_f64,
            _ => crate::native::ptl_matmul_f32,
        };
        // SAFETY: args = [a, b, out] with the output allocated above (or provided) with
        // the product's shape; the matmul kernel only reads data/shape/strides.
        let rc = unsafe { f(ctx, args.raw().as_mut_ptr(), std::ptr::null()) };
        if rc != 0 {
            return Err("Dot failed".into());
        }
        Ok(())
    }
}
