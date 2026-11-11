//! Sort along an axis, NaNs last.

use crate::abi::OutputMode;
use crate::kernel::{zip_lanes, Args, Element, Kernel};
use crate::runtime::PtCtx;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Elements that can be sorted.
pub trait Sortable: Element {
    fn compare(a: &Self, b: &Self) -> Ordering;
}

macro_rules! sortable_float {
    ($($t:ty),*) => {$(
        impl Sortable for $t {
            fn compare(a: &Self, b: &Self) -> Ordering {
                // NaNs compare equal to each other and greater than everything else
                a.partial_cmp(b).unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
            }
        }
    )*};
}
sortable_float!(f64, f32);

impl Sortable for i64 {
    fn compare(a: &Self, b: &Self) -> Ordering {
        a.cmp(b)
    }
}

pub struct Sort<T> {
    pub mode: OutputMode,
    pub axis: usize,
    pub stable: bool,
    pub element: PhantomData<T>,
}

// SAFETY: one input and one output, both arrays of `T` (`interp::py::Sort` picks `T`
// from the node's dtype).
unsafe impl<T: Sortable> Kernel for Sort<T> {
    fn arity(&self) -> (usize, usize) {
        (1, 1)
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        let source = args.inputs.array::<T>(0);
        let mut sorted = args.outputs.array::<T>(ctx, 0, self.mode, source.shape())?;
        let mut values = Vec::new();
        zip_lanes(&source, &mut sorted, self.axis, |lane, mut target| {
            lane.read_into(&mut values);
            if self.stable {
                values.sort_by(T::compare);
            } else {
                values.sort_unstable_by(T::compare);
            }
            target.write(&values);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::PtArray;
    use crate::kernel::entry;

    fn array(data: &mut [f64], shape: &mut [i64], strides: &mut [i64]) -> PtArray {
        PtArray { data: data.as_mut_ptr() as *mut u8, shape: shape.as_mut_ptr(), strides: strides.as_mut_ptr(), owner: std::ptr::null_mut(), ndim: shape.len() as i64 }
    }

    #[test]
    fn sorts_every_lane_into_a_provided_output_with_other_strides() {
        // 2 x 3, rows are lanes; the output is laid out column by column
        let mut source = [3.0, f64::NAN, 1.0, 6.0, 5.0, 4.0];
        let mut sorted = [0.0; 6];
        let (mut shape, mut out_shape) = ([2i64, 3], [2i64, 3]);
        let (mut strides, mut out_strides) = ([24i64, 8], [8i64, 16]);
        let mut args = [array(&mut source, &mut shape, &mut strides), array(&mut sorted, &mut out_shape, &mut out_strides)];
        let kernel = Sort::<f64> { mode: OutputMode::Provided, axis: 1, stable: true, element: PhantomData };
        let mut ctx = PtCtx::new();
        // SAFETY: a live context, the kernel as its config, and one valid input and
        // one provided output of f64 that don't overlap.
        let code = unsafe { entry::<Sort<f64>>(&mut ctx, args.as_mut_ptr(), &kernel as *const _ as *const u8) };
        assert_eq!(code, 0);
        assert_eq!(sorted[0], 1.0);
        assert_eq!(sorted[2], 3.0);
        assert!(sorted[4].is_nan());
        assert_eq!([sorted[1], sorted[3], sorted[5]], [4.0, 5.0, 6.0]);

        // a provided output of another shape is an error, not a write out of bounds
        out_shape[1] = 2;
        let mut args = [array(&mut source, &mut shape, &mut strides), array(&mut sorted, &mut out_shape, &mut out_strides)];
        // SAFETY: as above.
        let code = unsafe { entry::<Sort<f64>>(&mut ctx, args.as_mut_ptr(), &kernel as *const _ as *const u8) };
        assert_eq!(code, 1);
        assert!(ctx.error.take().unwrap().1.contains("expected [2, 3]"));
    }
}
