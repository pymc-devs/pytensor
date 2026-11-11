//! Writing Rust kernels.
//!
//! A kernel is a type that holds its parameters and implements `Kernel`: a safe
//! function from the inputs to the outputs. `entry::<K>` is its function with the
//! kernel ABI (`abi::KernelFn`), and a `K` is its config:
//!
//! ```ignore
//! pub struct Negate<T> { mode: OutputMode, _t: PhantomData<T> }
//!
//! unsafe impl<T: Element + Neg<Output = T>> Kernel for Negate<T> {
//!     fn arity(&self) -> (usize, usize) { (1, 1) }
//!     fn run(&self, ctx: &mut PtCtx, args: Args) -> Result<(), String> {
//!         let x = args.inputs.array::<T>(0);
//!         let mut out = args.outputs.array::<T>(ctx, 0, self.mode, x.shape())?;
//!         zip_lanes(&x, &mut out, 0, |x, mut out| { ... })
//!     }
//! }
//! ```
//!
//! All the raw-pointer work of a kernel is here: `Args` turns the `PtArray`s of a call
//! into views (`Array`, `ArrayMut`: typed; `Bytes`, `BytesMut`: any element type) whose
//! methods are safe. What makes them safe is the contract of the call, which `entry`
//! takes on (the arrays are valid and are what the kernel was planned for), and one
//! promise of the kernel: it asks for the element types it was planned with (that is
//! why `Kernel` is an unsafe trait, and the only `unsafe` a kernel needs).

use crate::abi::{OutputMode, PtArray, bad_size, checked_nbytes};
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
///
/// # Safety
/// `run` asks `args` only for arrays of the element types the kernel is planned with
/// (whoever makes the kernel for a node, `interp/py.rs`, picks the type from the
/// node's dtypes), and for no more inputs and outputs than `arity` says.
pub unsafe trait Kernel: Send + Sync + 'static {
    /// (number of inputs, number of outputs)
    fn arity(&self) -> (usize, usize);

    fn run(&self, ctx: &mut PtCtx, args: Args) -> Result<(), String>;
}

/// The kernel `K` with the kernel ABI (`abi::KernelFn`).
///
/// # Safety
/// The kernel contract: `ctx` is the live context of the call, `config` points to a
/// `K`, and `args` holds the arrays `K` was planned for: valid inputs of its element
/// types, then outputs as their modes say (provided: valid writable memory that
/// overlaps no input and no other output; otherwise writable shape and strides
/// storage of the output's ndim). Strides are multiples of the itemsize and data is
/// aligned (the C API checks both for what comes from outside).
pub unsafe extern "C-unwind" fn entry<K: Kernel>(
    ctx: *mut PtCtx,
    args: *mut PtArray,
    config: *const u8,
) -> i32 {
    // SAFETY: the function's contract.
    let (kernel, c) = unsafe { (&*(config as *const K), &mut *ctx) };
    let (n_in, n_out) = kernel.arity();
    // SAFETY: `args` holds n_in + n_out arrays (the function's contract), and nothing
    // else uses them during the call.
    let arrays = unsafe { std::slice::from_raw_parts_mut(args, n_in + n_out) };
    let (inputs, outputs) = arrays.split_at_mut(n_in);
    match kernel.run(
        c,
        Args {
            inputs: Inputs { arrays: inputs },
            outputs: Outputs { arrays: outputs },
        },
    ) {
        Ok(()) => 0,
        Err(message) => {
            c.error = Some((1, message));
            1
        }
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
    pub fn bytes(&self, i: usize, itemsize: usize) -> Bytes<'a> {
        let a = &self.arrays[i];
        // SAFETY: inputs are valid arrays (`entry`'s contract): shape and strides have
        // `ndim` entries, which nothing writes during the call.
        let (shape, strides) = unsafe { (a.shape(), a.strides()) };
        Bytes {
            data: a.data,
            shape,
            strides,
            itemsize,
        }
    }

    /// Input `i`, an array of `T` (the kernel's promise: it is one).
    pub fn array<T: Element>(&self, i: usize) -> Array<'a, T> {
        Array {
            bytes: self.bytes(i, size_of::<T>()),
            _t: PhantomData,
        }
    }
}

pub struct Outputs<'a> {
    arrays: &'a mut [PtArray],
}

impl Outputs<'_> {
    /// Memory for output `o` of `shape`: the provided memory (whose shape must match),
    /// or a new C-contiguous buffer.
    pub fn bytes(
        &mut self,
        ctx: &mut PtCtx,
        o: usize,
        mode: OutputMode,
        itemsize: usize,
        shape: &[i64],
    ) -> Result<BytesMut<'_>, String> {
        let a = &mut self.arrays[o];
        match mode {
            OutputMode::Provided => {
                // SAFETY: provided outputs are valid arrays (`entry`'s contract).
                let (have, strides) = unsafe { (a.shape(), a.strides()) };
                if have != shape {
                    return Err(format!("output {o} has shape {have:?}, expected {shape:?}"));
                }
                Ok(BytesMut {
                    data: a.data,
                    shape: have,
                    strides,
                    itemsize,
                })
            }
            OutputMode::Allocated => {
                if a.ndim as usize != shape.len() {
                    return Err(format!(
                        "output {o} has {} axes, expected {}",
                        a.ndim,
                        shape.len()
                    ));
                }
                let n = checked_nbytes(shape, itemsize).ok_or_else(|| bad_size("output", shape))?;
                // SAFETY: ctx is the live context.
                let (data, owner) = unsafe { PtCtx::alloc(ctx, n) };
                if data.is_null() {
                    return Err(format!("could not allocate {n} bytes"));
                }
                // SAFETY: the output has shape storage of its ndim (`entry`'s contract;
                // ndim checked above), ours for the call.
                unsafe { a.shape_mut() }.copy_from_slice(shape);
                // SAFETY: as above, for the strides.
                let strides = unsafe { a.strides_mut() };
                let mut acc = itemsize as i64;
                for d in (0..shape.len()).rev() {
                    strides[d] = acc;
                    acc *= shape[d].max(1);
                }
                a.data = data;
                a.owner = owner;
                // SAFETY: as above; only read from here on.
                let (shape, strides) = unsafe { (a.shape(), a.strides()) };
                Ok(BytesMut {
                    data,
                    shape,
                    strides,
                    itemsize,
                })
            }
            OutputMode::View => Err(format!(
                "output {o}: this kernel writes its outputs, it can't return a view"
            )),
        }
    }

    /// Memory for output `o`, an array of `T` (the kernel's promise: it is one).
    pub fn array<T: Element>(
        &mut self,
        ctx: &mut PtCtx,
        o: usize,
        mode: OutputMode,
        shape: &[i64],
    ) -> Result<ArrayMut<'_, T>, String> {
        Ok(ArrayMut {
            bytes: self.bytes(ctx, o, mode, size_of::<T>(), shape)?,
            _t: PhantomData,
        })
    }

    /// Output `o` is a view of input `i`: its memory, seen with `shape` and `strides`
    /// (per output axis, the input axis it is, or None for a new axis of length 1).
    pub fn view(
        &mut self,
        o: usize,
        inputs: &Inputs,
        i: usize,
        axes: &[Option<usize>],
    ) -> Result<(), String> {
        let (out, source) = (&mut self.arrays[o], &inputs.arrays[i]);
        if out.ndim as usize != axes.len() {
            return Err(format!(
                "output {o} has {} axes, expected {}",
                out.ndim,
                axes.len()
            ));
        }
        // SAFETY: a valid input (`entry`'s contract).
        let (shape, strides) = unsafe { (source.shape(), source.strides()) };
        // SAFETY: the output has shape storage of its ndim (`entry`'s contract; ndim
        // checked above), ours for the call.
        for (length, axis) in unsafe { out.shape_mut() }.iter_mut().zip(axes) {
            *length = axis.map_or(1, |axis| shape[axis]);
        }
        // SAFETY: as above, for the strides.
        for (stride, axis) in unsafe { out.strides_mut() }.iter_mut().zip(axes) {
            *stride = axis.map_or(0, |axis| strides[axis]);
        }
        out.data = source.data;
        out.owner = source.owner;
        Ok(())
    }
}

// ------------------------------------------------------------------ views

/// An array to read, of elements of some size: valid memory for its shape and
/// strides (in bytes), not written while this lives.
#[derive(Clone, Copy)]
pub struct Bytes<'a> {
    data: *const u8,
    shape: &'a [i64],
    strides: &'a [i64],
    itemsize: usize,
}

/// An array to write; nothing else reads or writes its memory while this lives.
pub struct BytesMut<'a> {
    data: *mut u8,
    shape: &'a [i64],
    strides: &'a [i64],
    itemsize: usize,
}

impl<'a> Bytes<'a> {
    pub fn shape(&self) -> &'a [i64] {
        self.shape
    }

    pub fn strides(&self) -> &'a [i64] {
        self.strides
    }
}

impl BytesMut<'_> {
    pub fn shape(&self) -> &[i64] {
        self.shape
    }

    /// Copy the elements of `source`, an array of the same shape and element size.
    pub fn copy_from(&mut self, source: &Bytes) -> Result<(), String> {
        if source.shape != self.shape || source.itemsize != self.itemsize {
            return Err(format!(
                "can't copy an array of shape {:?} into one of shape {:?}",
                source.shape, self.shape
            ));
        }
        // SAFETY: both are valid memory of this shape with their strides, and memory
        // to write overlaps nothing else (the invariants of the two types).
        unsafe {
            crate::interp::program::copy_strided(
                source.data,
                self.shape,
                source.strides,
                self.data,
                self.strides,
                self.itemsize,
            )
        };
        Ok(())
    }
}

/// An array of `T` to read.
#[derive(Clone, Copy)]
pub struct Array<'a, T> {
    bytes: Bytes<'a>,
    _t: PhantomData<&'a T>,
}

/// An array of `T` to write.
pub struct ArrayMut<'a, T> {
    bytes: BytesMut<'a>,
    _t: PhantomData<&'a mut T>,
}

/// (rows, columns, row stride, column stride) of a vector or matrix, strides in
/// elements. A vector is a column, or a row if `as_row`; a scalar is 1 x 1.
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
    pub fn shape(&self) -> &'a [i64] {
        self.bytes.shape
    }

    pub fn ndim(&self) -> usize {
        self.bytes.shape.len()
    }

    pub fn bytes(&self) -> &Bytes<'a> {
        &self.bytes
    }

    /// A vector or matrix as a faer matrix. A vector is a column, or a row if `as_row`.
    pub fn matrix(&self, as_row: bool) -> MatRef<'a, T> {
        let (rows, cols, row_stride, col_stride) =
            matrix_parts(self.bytes.shape, self.bytes.strides, size_of::<T>(), as_row);
        // SAFETY: valid, aligned memory of `T` for this shape and these strides, not
        // written while 'a lasts (the type's invariant).
        unsafe {
            MatRef::from_raw_parts(
                self.bytes.data as *const T,
                rows,
                cols,
                row_stride,
                col_stride,
            )
        }
    }

    /// The elements of a vector whose elements are adjacent in memory, else None.
    pub fn as_slice(&self) -> Option<&'a [T]> {
        match (self.bytes.shape, self.bytes.strides) {
            (&[n], &[stride]) if n <= 1 || stride == size_of::<T>() as i64 => {
                // SAFETY: n adjacent, aligned, valid `T`s, not written while 'a lasts.
                Some(if n == 0 {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts(self.bytes.data as *const T, n as usize) }
                })
            }
            _ => None,
        }
    }
}

impl<T: Element> ArrayMut<'_, T> {
    pub fn shape(&self) -> &[i64] {
        self.bytes.shape
    }

    pub fn copy_from(&mut self, source: &Array<T>) -> Result<(), String> {
        self.bytes.copy_from(&source.bytes)
    }

    /// A vector or matrix as a faer matrix. A vector is a column, or a row if `as_row`.
    pub fn matrix_mut(&mut self, as_row: bool) -> MatMut<'_, T> {
        let (rows, cols, row_stride, col_stride) =
            matrix_parts(self.bytes.shape, self.bytes.strides, size_of::<T>(), as_row);
        // SAFETY: valid, aligned memory of `T` for this shape and these strides that
        // nothing else uses while `self` is borrowed (the type's invariant).
        unsafe {
            MatMut::from_raw_parts_mut(
                self.bytes.data as *mut T,
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
    let shape = source.bytes.shape;
    if shape != target.bytes.shape {
        return Err(format!(
            "arrays of shape {:?} and {:?} don't have the same lanes",
            shape, target.bytes.shape
        ));
    }
    if axis >= shape.len() {
        return Err(format!("axis {axis} of an array with {} axes", shape.len()));
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let (s, t) = (source.bytes.strides, target.bytes.strides);
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
                    first: source.bytes.data.offset(source_offset) as *const T,
                    len,
                    stride: s[axis] as isize,
                    _t: PhantomData,
                },
                LaneMut {
                    first: target.bytes.data.offset(target_offset) as *mut T,
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
    let bytes = n
        .checked_mul(size_of::<T>())
        .expect("scratch size overflows");
    // SAFETY: `ctx` is a live context. Scratch memory is aligned for every element
    // type, at least `bytes` long, distinct from everything else, and stays valid
    // until the task ends, which outlives the borrow of `ctx`.
    unsafe { std::slice::from_raw_parts_mut(PtCtx::scratch(ctx, bytes) as *mut MaybeUninit<T>, n) }
}
