//! Rust kernels that move data: views and copies. (`linalg.rs` and `sort.rs` have the
//! others.)
//!
//! A kernel's config holds its parameters and the mode of each output, which Python
//! chooses per node (the kernel never infers it from the arguments).

use crate::abi::OutputMode;
use crate::dtype::DType;
use crate::kernel::{Args, Kernel};
use crate::runtime::PtCtx;

/// A view of the input with its axes rearranged.
pub struct DimShuffle {
    /// output axis -> input axis, or None for a new length-1 axis
    pub order: Vec<Option<usize>>,
}

impl Kernel for DimShuffle {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, _ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        args.outputs.view(0, &args.inputs, 0, &self.order)
    }
}

/// A copy of the input.
pub struct Copy {
    pub mode: OutputMode,
    pub dtype: DType,
}

impl Kernel for Copy {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        let source = args.inputs.get(0)?;
        args.outputs
            .untyped(ctx, 0, self.mode, self.dtype, source.shape())?
            .copy_from(&source)
    }
}
