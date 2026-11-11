//! Writing Rust kernels.
//!
//! A kernel is a type that holds its parameters and implements `Kernel`: a safe
//! function from the inputs to the outputs. `entry::<K>` is its function with the
//! kernel ABI (`abi::KernelFn`), and a `K` is its config:
//!
//! ```ignore
//! pub struct Negate<T> { mode: OutputMode, _t: PhantomData<T> }
//!
//! impl<T: Element + Neg<Output = T>> Kernel for Negate<T> {
//!     fn arity(&self) -> (usize, usize) { (1, 1) }
//!     fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
//!         let x = args.inputs.array::<T>(0)?;
//!         let mut out = args.outputs.array::<T>(ctx, 0, self.mode, x.shape())?;
//!         zip_lanes(&x, &mut out, 0, |x, mut out| { ... })
//!     }
//! }
//! ```
//!
//! Writing a kernel needs no `unsafe`. `Args` hands out the arguments of the call as
//! the views of `crate::array`, and as typed arrays (`Array<T>`, `ArrayMut<T>`) on top
//! of those. What a kernel asks for is checked against what it was called with: the
//! number of inputs and outputs, the dtype of every array it wants as an array of `T`,
//! the number of axes and whether an output is provided or to be produced. A kernel
//! that was planned for other arguments than it gets fails with an error.
//!
//! What can't be checked is that the memory the arguments describe is there. That is
//! the contract of the call, which `entry` takes on (see there); it is the one place
//! that turns the raw arguments into `Args`.

use crate::abi::{OutputMode, PtArray};
use crate::array::{OutSlot, PtArrayMut, PtArrayRef};
use crate::dtype::DType;
use crate::runtime::PtCtx;
use faer::{MatMut, MatRef};
use std::marker::PhantomData;
use std::mem::MaybeUninit;

/// The element types of arrays.
///
/// # Safety
/// `DTYPE` is the dtype whose elements have exactly this type's representation, and
/// every bit pattern of that size is a valid value.
pub unsafe trait Element: Copy + Send + Sync + 'static {
    const DTYPE: DType;
}

macro_rules! element {
    ($($t:ty => $d:ident),*) => {$(
        // SAFETY: the numpy dtype of this primitive type; all bit patterns are valid.
        unsafe impl Element for $t {
            const DTYPE: DType = DType::$d;
        }
    )*};
}
element!(f64 => Float64, f32 => Float32, i64 => Int64, i32 => Int32, i16 => Int16, i8 => Int8, u8 => Uint8);

/// A Rust kernel: its parameters, and what it computes.
pub trait Kernel: Send + Sync + 'static {
    /// (number of inputs, number of outputs)
    fn arity(&self) -> (usize, usize);

    fn run(&self, ctx: &mut PtCtx, args: Args) -> Result<(), String>;
}

/// The kernel `K` with the kernel ABI (`abi::KernelFn`).
///
/// # Safety
/// The kernel contract: `ctx` is the live context of the call, used by nobody else
/// during it, `config` points to a `K`, and `args` holds `n_inputs` arrays, then
/// `n_outputs` outputs (see `Args::from_raw`).
pub unsafe extern "C-unwind" fn entry<K: Kernel>(
    ctx: *mut PtCtx,
    args: *mut PtArray,
    n_inputs: i32,
    n_outputs: i32,
    config: *const u8,
) -> i32 {
    // SAFETY: the function's contract.
    let (kernel, ctx) = unsafe { (&*(config as *const K), &mut *ctx) };
    // SAFETY: the function's contract.
    let result = unsafe { Args::from_raw(args, n_inputs, n_outputs) }.and_then(|args| {
        let arity = (args.inputs.len(), args.outputs.len());
        if arity != kernel.arity() {
            return Err(format!(
                "the kernel takes {} inputs and {} outputs, but was called with {} and {}",
                kernel.arity().0,
                kernel.arity().1,
                arity.0,
                arity.1
            ));
        }
        kernel.run(ctx, args)
    });
    match result {
        Ok(()) => 0,
        Err(message) => ctx.fail(1, &message),
    }
}

/// The address of `entry::<K>`, for a task.
pub fn address<K: Kernel>() -> usize {
    entry::<K> as *const () as usize
}

/// The arguments of one call.
pub struct Args<'a> {
    pub inputs: Inputs<'a>,
    pub outputs: Outputs<'a>,
}

impl<'a> Args<'a> {
    /// The arguments of a call with the kernel ABI.
    ///
    /// # Safety
    /// `args` points to `n_inputs + n_outputs` `PtArray`s (if that is more than
    /// none), ours alone during `'a`:
    /// * the first `n_inputs` are arrays to read (`PtArrayRef::from_raw`),
    /// * the others are outputs to produce (`OutSlot::from_raw`), and memory that is
    ///   provided for an output overlaps no input and no other output.
    pub unsafe fn from_raw(
        args: *mut PtArray,
        n_inputs: i32,
        n_outputs: i32,
    ) -> Result<Args<'a>, String> {
        if n_inputs < 0 || n_outputs < 0 {
            return Err(format!(
                "a call with {n_inputs} inputs and {n_outputs} outputs"
            ));
        }
        let (n_inputs, n_outputs) = (n_inputs as usize, n_outputs as usize);
        let arrays: &mut [PtArray] = if n_inputs + n_outputs == 0 {
            &mut []
        } else if args.is_null() {
            return Err("the arguments of the call are NULL".into());
        } else {
            // SAFETY: the function's contract.
            unsafe { std::slice::from_raw_parts_mut(args, n_inputs + n_outputs) }
        };
        let (inputs, outputs) = arrays.split_at_mut(n_inputs);
        Ok(Args {
            inputs: Inputs { arrays: inputs },
            outputs: Outputs { arrays: outputs },
        })
    }
}

/// The inputs of a call. (Invariant: every one is an array as `Args::from_raw` asks.)
pub struct Inputs<'a> {
    arrays: &'a [PtArray],
}

impl<'a> Inputs<'a> {
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    /// Input `i`, whatever its elements are.
    #[inline]
    pub fn get(&self, i: usize) -> Result<PtArrayRef<'a>, String> {
        let Some(raw) = self.arrays.get(i) else {
            return Err(no_argument("input", i, self.arrays.len()));
        };
        // SAFETY: the type's invariant.
        match unsafe { PtArrayRef::from_raw(raw) } {
            Ok(array) => Ok(array),
            Err(e) => Err(argument_error("input", i, e)),
        }
    }

    /// Input `i`, an array of `T`.
    #[inline]
    pub fn array<T: Element>(&self, i: usize) -> Result<Array<'a, T>, String> {
        match Array::new(self.get(i)?) {
            Ok(array) => Ok(array),
            Err(e) => Err(argument_error("input", i, e)),
        }
    }
}

// (Errors are rare, and what leads to them is on the way of every kernel call: they
// are made apart.)

#[cold]
#[inline(never)]
fn no_argument(what: &str, i: usize, n: usize) -> String {
    format!("no {what} {i}: the call has {n}")
}

#[cold]
#[inline(never)]
fn argument_error(what: &str, i: usize, e: String) -> String {
    format!("{what} {i}: {e}")
}

fn output_error(o: usize, e: String) -> String {
    argument_error("output", o, e)
}

/// The outputs of a call. (Invariant: every one is an output as `Args::from_raw`
/// asks, until it is produced.)
pub struct Outputs<'a> {
    arrays: &'a mut [PtArray],
}

impl<'a> Outputs<'a> {
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    /// Output `o`, to produce.
    #[inline]
    pub fn slot(&mut self, o: usize) -> Result<OutSlot<'_>, String> {
        let n = self.arrays.len();
        let Some(raw) = self.arrays.get_mut(o) else {
            return Err(no_argument("output", o, n));
        };
        // SAFETY: the type's invariant.
        match unsafe { OutSlot::from_raw(raw) } {
            Ok(slot) => Ok(slot),
            Err(e) => Err(output_error(o, e)),
        }
    }

    /// All outputs, to produce, in order.
    pub fn into_slots(self) -> impl Iterator<Item = Result<OutSlot<'a>, String>> {
        self.arrays.iter_mut().enumerate().map(|(o, raw)| {
            // SAFETY: the type's invariant.
            unsafe { OutSlot::from_raw(raw) }.map_err(|e| output_error(o, e))
        })
    }

    /// Memory for output `o`, an array of `dtype` and `shape`: the provided memory
    /// (whose shape must match), or a new C-contiguous buffer.
    #[inline]
    pub fn untyped(
        &mut self,
        ctx: &mut PtCtx,
        o: usize,
        mode: OutputMode,
        dtype: DType,
        shape: &[i64],
    ) -> Result<PtArrayMut<'_>, String> {
        let slot = self.slot(o)?;
        if slot.dtype() != dtype {
            return Err(output_error(
                o,
                format!(
                    "it is an array of {}, the kernel writes one of {}",
                    slot.dtype().numpy_name(),
                    dtype.numpy_name()
                ),
            ));
        }
        match mode {
            OutputMode::Provided => match slot.provided() {
                Ok(array) if array.shape() == shape => Ok(array),
                Ok(array) => Err(output_error(
                    o,
                    format!("it has shape {:?}, expected {shape:?}", array.shape()),
                )),
                Err(e) => Err(output_error(o, e)),
            },
            OutputMode::Allocated => slot.alloc(ctx, shape).map_err(|e| output_error(o, e)),
            OutputMode::View => Err(output_error(
                o,
                "this kernel writes its outputs, it can't return a view".into(),
            )),
        }
    }

    /// Memory for output `o`, an array of `T` and `shape`.
    pub fn array<T: Element>(
        &mut self,
        ctx: &mut PtCtx,
        o: usize,
        mode: OutputMode,
        shape: &[i64],
    ) -> Result<ArrayMut<'_, T>, String> {
        Ok(ArrayMut {
            raw: self.untyped(ctx, o, mode, T::DTYPE, shape)?,
            _t: PhantomData,
        })
    }

    /// Output `o` is a view of input `i`: its memory, seen with other axes (per output
    /// axis, the input axis it is, or None for a new axis of length 1).
    pub fn view(
        &mut self,
        o: usize,
        inputs: &Inputs<'a>,
        i: usize,
        axes: &[Option<usize>],
    ) -> Result<(), String> {
        let source = inputs.get(i)?;
        // SAFETY: `source` is an input of the call this is an output of (`Args` makes
        // both, with the same `'a`): the caller keeps it alive with the view.
        unsafe { self.slot(o)?.set_view(&source, axes) }.map_err(|e| output_error(o, e))
    }
}

// ------------------------------------------------------------------ typed arrays

/// An array of `T` to read. (Invariant: the dtype of `raw` is `T`'s.)
#[derive(Clone, Copy)]
pub struct Array<'a, T> {
    raw: PtArrayRef<'a>,
    _t: PhantomData<&'a T>,
}

/// An array of `T` to write. (Invariant: the dtype of `raw` is `T`'s.)
pub struct ArrayMut<'a, T> {
    raw: PtArrayMut<'a>,
    _t: PhantomData<&'a mut T>,
}

#[inline]
fn check_dtype<T: Element>(dtype: DType) -> Result<(), String> {
    if dtype != T::DTYPE {
        return Err(other_dtype(T::DTYPE, dtype));
    }
    Ok(())
}

#[cold]
#[inline(never)]
fn other_dtype(expected: DType, dtype: DType) -> String {
    format!(
        "an array of {} is expected, got one of {}",
        expected.numpy_name(),
        dtype.numpy_name()
    )
}

/// (rows, columns, row stride, column stride) of a vector or matrix, strides in
/// elements. A vector is a column, or a row if `as_row`; a scalar is 1 x 1.
/// (Inlined: `itemsize` is a constant where this is used, and dividing by it a shift.)
#[inline(always)]
fn matrix_parts(
    shape: &[i64],
    strides: &[i64],
    itemsize: usize,
    as_row: bool,
) -> (usize, usize, isize, isize) {
    let isz = itemsize as i64;
    match (shape.len(), as_row) {
        (2, _) => (
            shape[0] as usize,
            shape[1] as usize,
            (strides[0] / isz) as isize,
            (strides[1] / isz) as isize,
        ),
        (1, true) => (1, shape[0] as usize, 0, (strides[0] / isz) as isize),
        (1, false) => (shape[0] as usize, 1, (strides[0] / isz) as isize, 0),
        (0, _) => (1, 1, 0, 0),
        (n, _) => panic!("a vector or a matrix is expected, got an array with {n} axes"),
    }
}

impl<'a, T: Element> Array<'a, T> {
    /// `raw` as an array of `T`, if that is what it is.
    #[inline]
    pub fn new(raw: PtArrayRef<'a>) -> Result<Array<'a, T>, String> {
        check_dtype::<T>(raw.dtype())?;
        Ok(Array {
            raw,
            _t: PhantomData,
        })
    }

    pub fn shape(&self) -> &'a [i64] {
        self.raw.shape()
    }

    pub fn ndim(&self) -> usize {
        self.raw.ndim()
    }

    /// The array, whatever its elements are.
    pub fn raw(&self) -> &PtArrayRef<'a> {
        &self.raw
    }

    /// A vector or matrix as a faer matrix. A vector is a column, or a row if `as_row`.
    #[inline]
    pub fn matrix(&self, as_row: bool) -> MatRef<'a, T> {
        let (rows, cols, row_stride, col_stride) =
            matrix_parts(self.raw.shape(), self.raw.strides(), size_of::<T>(), as_row);
        // SAFETY: valid, aligned memory of `T` for this shape and these strides, not
        // written while 'a lasts (the invariants of `PtArrayRef` and of this type).
        unsafe {
            MatRef::from_raw_parts(
                self.raw.as_ptr() as *const T,
                rows,
                cols,
                row_stride,
                col_stride,
            )
        }
    }

    /// The elements of a vector whose elements are adjacent in memory, else None.
    pub fn as_slice(&self) -> Option<&'a [T]> {
        match (self.raw.shape(), self.raw.strides()) {
            (&[n], &[stride]) if n <= 1 || stride == size_of::<T>() as i64 => {
                Some(if n == 0 {
                    &[]
                } else {
                    // SAFETY: n adjacent, aligned, valid `T`s, not written while 'a
                    // lasts (the invariants of `PtArrayRef` and of this type).
                    unsafe { std::slice::from_raw_parts(self.raw.as_ptr() as *const T, n as usize) }
                })
            }
            _ => None,
        }
    }
}

impl<'a, T: Element> ArrayMut<'a, T> {
    /// `raw` as an array of `T`, if that is what it is.
    pub fn new(raw: PtArrayMut<'a>) -> Result<ArrayMut<'a, T>, String> {
        check_dtype::<T>(raw.dtype())?;
        Ok(ArrayMut {
            raw,
            _t: PhantomData,
        })
    }

    pub fn shape(&self) -> &[i64] {
        self.raw.shape()
    }

    pub fn copy_from(&mut self, source: &Array<T>) -> Result<(), String> {
        self.raw.copy_from(&source.raw)
    }

    /// A vector or matrix as a faer matrix. A vector is a column, or a row if `as_row`.
    #[inline]
    pub fn matrix_mut(&mut self, as_row: bool) -> MatMut<'_, T> {
        let (rows, cols, row_stride, col_stride) =
            matrix_parts(self.raw.shape(), self.raw.strides(), size_of::<T>(), as_row);
        // SAFETY: valid, aligned memory of `T` for this shape and these strides that
        // nothing else uses while `self` is borrowed (the invariants of `PtArrayMut`
        // and of this type).
        unsafe {
            MatMut::from_raw_parts_mut(
                self.raw.as_mut_ptr() as *mut T,
                rows,
                cols,
                row_stride,
                col_stride,
            )
        }
    }
}

// ------------------------------------------------------------------ lanes

/// The elements of an array along one axis, the other indices fixed.
pub struct Lane<'a, T> {
    first: *const T,
    len: usize,
    stride: isize,
    _t: PhantomData<&'a T>,
}

pub struct LaneMut<'a, T> {
    first: *mut T,
    len: usize,
    stride: isize,
    _t: PhantomData<&'a mut T>,
}

impl<T: Element> Lane<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Replace the contents of `values` by the lane's elements.
    pub fn read_into(&self, values: &mut Vec<T>) {
        values.clear();
        // SAFETY: the lane's `len` elements, `stride` bytes apart, are valid `T`s.
        values.extend(
            (0..self.len as isize).map(|i| unsafe { *self.first.byte_offset(i * self.stride) }),
        );
    }
}

impl<T: Element> LaneMut<'_, T> {
    /// Set the lane's elements; `values` has one for each.
    pub fn write(&mut self, values: &[T]) {
        assert_eq!(
            values.len(),
            self.len,
            "a lane of {} elements can't take {}",
            self.len,
            values.len()
        );
        for (i, value) in values.iter().enumerate() {
            // SAFETY: the lane's `len` elements, `stride` bytes apart, are ours to write.
            unsafe { *self.first.byte_offset(i as isize * self.stride) = *value };
        }
    }
}

/// Call `f` with every lane of `source` along `axis` and the same lane of `target`,
/// an array of the same shape.
pub fn zip_lanes<T: Element>(
    source: &Array<T>,
    target: &mut ArrayMut<T>,
    axis: usize,
    mut f: impl FnMut(Lane<T>, LaneMut<T>),
) -> Result<(), String> {
    let shape = source.raw.shape();
    if shape != target.raw.shape() {
        return Err(format!(
            "arrays of shape {:?} and {:?} don't have the same lanes",
            shape,
            target.raw.shape()
        ));
    }
    if axis >= shape.len() {
        return Err(format!("axis {axis} of an array with {} axes", shape.len()));
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let (s, t) = (source.raw.strides(), target.raw.strides());
    let (source_data, target_data) = (source.raw.as_ptr(), target.raw.as_mut_ptr());
    let len = shape[axis] as usize;
    let mut index = vec![0i64; shape.len()];
    let (mut source_offset, mut target_offset) = (0isize, 0isize);
    loop {
        // SAFETY: the offsets are those of an index inside both arrays (kept below),
        // so the lanes are elements of the arrays; each target lane is handed out
        // once, and for the duration of `f` only.
        let (lane, lane_mut) = unsafe {
            (
                Lane {
                    first: source_data.offset(source_offset) as *const T,
                    len,
                    stride: s[axis] as isize,
                    _t: PhantomData,
                },
                LaneMut {
                    first: target_data.offset(target_offset) as *mut T,
                    len,
                    stride: t[axis] as isize,
                    _t: PhantomData,
                },
            )
        };
        f(lane, lane_mut);
        // the next index, leaving `axis` at 0
        let mut d = shape.len();
        loop {
            if d == 0 {
                return Ok(());
            }
            d -= 1;
            if d == axis {
                continue;
            }
            index[d] += 1;
            source_offset += s[d] as isize;
            target_offset += t[d] as isize;
            if index[d] < shape[d] {
                break;
            }
            source_offset -= (s[d] * shape[d]) as isize;
            target_offset -= (t[d] * shape[d]) as isize;
            index[d] = 0;
        }
    }
}

// ------------------------------------------------------------------ temporary memory

/// Temporary memory for `n` values of `T`, valid until the task ends.
pub fn scratch<T: Element>(ctx: &mut PtCtx, n: usize) -> &mut [MaybeUninit<T>] {
    ctx.scratch(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::OwnedBuf;

    /// Doubles a vector of f64.
    struct Double(OutputMode);

    impl Kernel for Double {
        fn arity(&self) -> (usize, usize) {
            (1, 1)
        }

        fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
            let x = args.inputs.array::<f64>(0)?;
            let values = x.as_slice().ok_or("a contiguous vector is expected")?;
            let mut out = args.outputs.array::<f64>(ctx, 0, self.0, x.shape())?;
            let mut out = out.matrix_mut(false);
            for (i, value) in values.iter().enumerate() {
                out[(i, 0)] = 2.0 * value;
            }
            Ok(())
        }
    }

    struct Call {
        values: [f64; 3],
        // (room for two axes, for the call that claims two)
        shapes: [[i64; 2]; 2],
        strides: [[i64; 2]; 2],
        result: [f64; 3],
        ctx: PtCtx,
    }

    impl Call {
        fn new() -> Call {
            Call {
                values: [1.0, 2.0, 3.0],
                shapes: [[3, 1], [3, 1]],
                strides: [[8, 8], [8, 8]],
                result: [0.0; 3],
                ctx: PtCtx::new(),
            }
        }

        /// Call `Double(mode)` with the input and one output, changed by `change`.
        fn run(
            &mut self,
            mode: OutputMode,
            n: (i32, i32),
            change: fn(&mut [PtArray; 2]),
        ) -> Result<PtArray, String> {
            let [in_shape, out_shape] = &mut self.shapes;
            let [in_strides, out_strides] = &mut self.strides;
            let array = |data: *mut f64, shape: &mut [i64; 2], strides: &mut [i64; 2]| PtArray {
                data: data as *mut u8,
                shape: shape.as_mut_ptr(),
                strides: strides.as_mut_ptr(),
                owner: std::ptr::null_mut(),
                ndim: 1,
                dtype: DType::Float64.code(),
            };
            let provided = match mode {
                OutputMode::Provided => self.result.as_mut_ptr(),
                _ => std::ptr::null_mut(),
            };
            let mut args = [
                array(self.values.as_mut_ptr(), in_shape, in_strides),
                array(provided, out_shape, out_strides),
            ];
            change(&mut args);
            let kernel = Double(mode);
            // SAFETY: a live context, the kernel as its config, and a valid input and
            // output of f64 that don't overlap (`change` only makes changes that the
            // kernel's checks reject before it uses any memory).
            let code = unsafe {
                entry::<Double>(
                    &mut self.ctx,
                    args.as_mut_ptr(),
                    n.0,
                    n.1,
                    &kernel as *const _ as *const u8,
                )
            };
            match code {
                0 => Ok(args[1]),
                _ => Err(self.ctx.error.take().unwrap().1),
            }
        }
    }

    #[test]
    fn a_kernel_writes_provided_and_allocated_outputs() {
        let mut call = Call::new();
        call.run(OutputMode::Provided, (1, 1), |_| ()).unwrap();
        assert_eq!(call.result, [2.0, 4.0, 6.0]);

        let out = call.run(OutputMode::Allocated, (1, 1), |_| ()).unwrap();
        // SAFETY: the kernel allocated three f64s and handed over its reference.
        let (values, buf) = unsafe {
            (
                std::slice::from_raw_parts(out.data as *const f64, 3),
                OwnedBuf::from_raw(out.owner),
            )
        };
        assert_eq!(values, [2.0, 4.0, 6.0]);
        assert!(buf.is_unique());
    }

    #[test]
    fn a_kernel_called_with_other_arguments_than_it_asks_for_fails() {
        let mut call = Call::new();
        let mut fails = |mode, n, change, message: &str| {
            let error = call.run(mode, n, change).err().unwrap();
            assert!(error.contains(message), "{error}");
        };
        let p = OutputMode::Provided;
        fails(p, (2, 0), |_| (), "takes 1 inputs and 1 outputs");
        fails(p, (-1, 1), |_| (), "-1 inputs");
        fails(
            p,
            (1, 1),
            |a| a[0].dtype = DType::Float32.code(),
            "input 0: an array of float64",
        );
        fails(
            p,
            (1, 1),
            |a| a[0].dtype = 77,
            "input 0: unknown dtype code 77",
        );
        fails(
            p,
            (1, 1),
            |a| a[1].dtype = DType::Int64.code(),
            "output 0: it is an array of int64",
        );
        fails(
            p,
            (1, 1),
            |a| a[1].ndim = 2,
            "output 0: it has shape [3, 1]",
        );
        fails(
            p,
            (1, 1),
            |a| a[1].data = std::ptr::null_mut(),
            "provides no memory",
        );
        fails(
            OutputMode::Allocated,
            (1, 1),
            |a| a[1].data = a[0].data,
            "provides memory",
        );
        fails(OutputMode::View, (1, 1), |_| (), "can't return a view");
        assert_eq!(call.result, [0.0; 3]);
    }
}
