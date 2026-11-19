//! Calling Rust kernels as generated code does, on arrays made by hand (tests only).
//!
//! Inputs live in memory of the test and may have any strides (`In`); outputs are
//! provided by the test, between guard bytes, or allocated by the kernel (`Out`).
//! `call` runs a kernel through its ABI entry and copies the outputs out (`Array`), and
//! checks what the ABI promises the caller: the inputs were not written, nothing next to
//! a provided output was written, and every buffer the kernel handed over is released
//! (so that miri sees no leaks). The tests also run under miri: keep the arrays small.

use crate::abi::{OutputMode, OwnedBuf, PtArray};
use crate::dtype::DType;
use crate::kernel::{Element, Kernel, entry};
use crate::runtime::PtCtx;

/// Bytes before and after the memory of a provided output, which must stay untouched.
const GUARD: usize = 64;
/// What memory is filled with before a call: NaN for floats, -1 for integers.
const FILL: u8 = 0xFF;

/// 8-byte aligned memory (enough for every element type) of at least `bytes` bytes,
/// filled with `FILL`.
fn memory(bytes: usize) -> Vec<u64> {
    vec![u64::from_ne_bytes([FILL; 8]); bytes.div_ceil(8).max(1)]
}

fn c_strides(shape: &[usize], itemsize: usize) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut stride = itemsize as i64;
    for (s, &n) in strides.iter_mut().zip(shape).rev() {
        *s = stride;
        stride *= n.max(1) as i64;
    }
    strides
}

fn to_i64(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|&n| n as i64).collect()
}

/// An input: values in memory of its own, seen with a shape and strides.
#[derive(Clone)]
pub struct In {
    memory: Vec<u64>,
    /// byte offset of the first element in `memory`
    offset: i64,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype: DType,
}

impl In {
    /// `values` (in C order) as a C-contiguous array.
    pub fn new<T: Element>(shape: &[usize], values: &[T]) -> In {
        In::spread(shape, values, 1)
    }

    /// `values` (in C order) with `gap - 1` unused elements after each: every stride
    /// is `gap` times the C-contiguous one.
    pub fn spread<T: Element>(shape: &[usize], values: &[T], gap: usize) -> In {
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
            "values for shape {shape:?}"
        );
        let mut memory = memory(values.len() * gap * size_of::<T>());
        let base = memory.as_mut_ptr() as *mut T;
        for (k, &value) in values.iter().enumerate() {
            // SAFETY: element `k * gap` of memory for `len * gap` elements of `T`,
            // aligned for `T` (8-byte aligned, and no element type is larger).
            unsafe { base.add(k * gap).write(value) };
        }
        let strides = c_strides(shape, size_of::<T>())
            .into_iter()
            .map(|s| s * gap as i64)
            .collect();
        In {
            memory,
            offset: 0,
            shape: to_i64(shape),
            strides,
            dtype: T::DTYPE,
        }
    }

    /// The same elements with the axes in reverse order: the transpose.
    pub fn t(mut self) -> In {
        self.shape.reverse();
        self.strides.reverse();
        self
    }

    /// The same memory, read backwards along `axis` (a negative stride).
    pub fn flip(mut self, axis: usize) -> In {
        if self.shape[axis] > 0 {
            self.offset += (self.shape[axis] - 1) * self.strides[axis];
        }
        self.strides[axis] = -self.strides[axis];
        self
    }
}

/// An output: memory the test provides (C or Fortran order), or one the kernel
/// allocates.
#[derive(Clone)]
pub struct Out {
    dtype: DType,
    /// provided: `GUARD` bytes, the array, `GUARD` bytes; None: allocated
    memory: Option<Vec<u64>>,
    /// the header: for an allocated output, storage the kernel fills
    shape: Vec<i64>,
    strides: Vec<i64>,
}

impl Out {
    /// An output of `shape` as `mode` says: provided (C order) or allocated.
    pub fn new<T: Element>(mode: OutputMode, shape: &[usize]) -> Out {
        match mode {
            OutputMode::Provided => Out::provided::<T>(shape),
            OutputMode::Allocated => Out::allocated::<T>(shape.len()),
            OutputMode::View => panic!("a view is not an output the test makes"),
        }
    }

    /// Memory the test provides, C-contiguous.
    pub fn provided<T: Element>(shape: &[usize]) -> Out {
        Out::provided_with(T::DTYPE, shape, c_strides(shape, size_of::<T>()))
    }

    /// Memory the test provides, Fortran-contiguous.
    pub fn provided_fortran<T: Element>(shape: &[usize]) -> Out {
        let reversed: Vec<usize> = shape.iter().rev().copied().collect();
        let mut strides = c_strides(&reversed, size_of::<T>());
        strides.reverse();
        Out::provided_with(T::DTYPE, shape, strides)
    }

    fn provided_with(dtype: DType, shape: &[usize], strides: Vec<i64>) -> Out {
        let bytes = shape.iter().product::<usize>() * dtype.itemsize();
        Out {
            dtype,
            memory: Some(memory(2 * GUARD + bytes)),
            shape: to_i64(shape),
            strides,
        }
    }

    /// An output with `ndim` axes that the kernel allocates.
    pub fn allocated<T: Element>(ndim: usize) -> Out {
        Out {
            dtype: T::DTYPE,
            memory: None,
            shape: vec![0; ndim],
            strides: vec![0; ndim],
        }
    }
}

/// An output of a call, copied out in C order.
#[derive(Clone, Debug)]
pub struct Array {
    pub dtype: DType,
    pub shape: Vec<i64>,
    bytes: Vec<u8>,
}

impl Array {
    /// The elements, in C order.
    pub fn values<T: Element>(&self) -> Vec<T> {
        assert_eq!(self.dtype, T::DTYPE, "the output's dtype");
        self.bytes
            .chunks_exact(size_of::<T>())
            // SAFETY: `size_of::<T>()` bytes of an element of `T`'s dtype, and every
            // bit pattern of an `Element` is a value.
            .map(|bytes| unsafe { (bytes.as_ptr() as *const T).read_unaligned() })
            .collect()
    }
}

/// The elements of an array as bytes, in C order.
///
/// # Safety
/// `data`, `shape` and `strides` (in bytes) describe readable elements of `itemsize`
/// bytes.
unsafe fn gather(data: *const u8, shape: &[i64], strides: &[i64], itemsize: usize) -> Vec<u8> {
    let n: i64 = shape.iter().product();
    let mut bytes = Vec::with_capacity(n.max(0) as usize * itemsize);
    if n <= 0 {
        return bytes;
    }
    let mut index = vec![0i64; shape.len()];
    loop {
        let offset: i64 = index.iter().zip(strides).map(|(i, s)| i * s).sum();
        // SAFETY: an element of the array (the function's contract).
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(data.wrapping_offset(offset as isize), itemsize)
        });
        let mut d = shape.len();
        loop {
            if d == 0 {
                return bytes;
            }
            d -= 1;
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// Call `kernel` (through its ABI entry) on `inputs` and `outputs`, in a new context:
/// the outputs, or the kernel's error message.
pub fn call<K: Kernel>(kernel: &K, inputs: &[In], outputs: Vec<Out>) -> Result<Vec<Array>, String> {
    call_in(&mut PtCtx::new(), kernel, inputs, outputs)
}

/// `call` in the context `ctx`, with the scratch mark generated code puts around a
/// kernel call.
pub fn call_in<K: Kernel>(
    ctx: &mut PtCtx,
    kernel: &K,
    inputs: &[In],
    mut outputs: Vec<Out>,
) -> Result<Vec<Array>, String> {
    let before: Vec<Vec<u64>> = inputs.iter().map(|i| i.memory.clone()).collect();
    // headers of their own for the inputs: the kernel must not write those either, but
    // they are not ours to lend mutably
    let mut headers: Vec<(Vec<i64>, Vec<i64>)> = inputs
        .iter()
        .map(|i| (i.shape.clone(), i.strides.clone()))
        .collect();
    let mut args = Vec::with_capacity(inputs.len() + outputs.len());
    for (input, (shape, strides)) in inputs.iter().zip(&mut headers) {
        args.push(PtArray {
            data: (input.memory.as_ptr() as *mut u8).wrapping_offset(input.offset as isize),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            owner: std::ptr::null_mut(),
            ndim: shape.len() as i32,
            dtype: input.dtype.code(),
        });
    }
    for out in &mut outputs {
        let data = match &mut out.memory {
            Some(memory) => (memory.as_mut_ptr() as *mut u8).wrapping_add(GUARD),
            None => std::ptr::null_mut(),
        };
        args.push(PtArray {
            data,
            shape: out.shape.as_mut_ptr(),
            strides: out.strides.as_mut_ptr(),
            owner: std::ptr::null_mut(),
            ndim: out.shape.len() as i32,
            dtype: out.dtype.code(),
        });
    }
    let mark = ctx.scratch_mark();
    // SAFETY: a live context used by nobody else, `kernel` as the config of its own
    // entry, and `inputs.len()` inputs, then `outputs.len()` outputs. Every input is an
    // array of its dtype in memory of its own (aligned, its strides inside it), which
    // nothing writes during the call. Every output has header storage of its number of
    // axes, and either provided memory of its shape and strides, inside its own
    // allocation and overlapping nothing else, or none (null) for the kernel to
    // produce.
    let code = unsafe {
        entry::<K>(
            ctx,
            args.as_mut_ptr(),
            inputs.len() as i32,
            outputs.len() as i32,
            kernel as *const K as *const u8,
        )
    };
    ctx.scratch_release(mark);
    let error = (code != 0).then(|| {
        ctx.error.take().map_or_else(
            || format!("code {code} and no message"),
            |(_, message)| message,
        )
    });
    for (k, (input, before)) in inputs.iter().zip(&before).enumerate() {
        assert!(input.memory == *before, "the kernel wrote input {k}");
    }
    let mut results = Vec::with_capacity(outputs.len());
    for (k, (out, raw)) in outputs.iter().zip(&args[inputs.len()..]).enumerate() {
        // a buffer the kernel allocated is ours now, also when the call failed after
        let buf = (!raw.owner.is_null()).then(|| {
            // SAFETY: the kernel handed over one reference to this buffer.
            unsafe { OwnedBuf::from_raw(raw.owner) }
        });
        if let Some(memory) = &out.memory {
            let bytes = out.shape.iter().product::<i64>() as usize * out.dtype.itemsize();
            // SAFETY: plain bytes of `memory`, which the kernel no longer uses.
            let all = unsafe {
                std::slice::from_raw_parts(memory.as_ptr() as *const u8, memory.len() * 8)
            };
            assert!(
                all[..GUARD]
                    .iter()
                    .chain(&all[GUARD + bytes..])
                    .all(|&b| b == FILL),
                "the kernel wrote next to the memory of output {k}"
            );
        }
        if error.is_some() {
            continue;
        }
        // SAFETY: after a call that succeeded, the output is an array of its dtype with
        // this header: the provided memory, or the buffer `buf` holds.
        let bytes = unsafe { gather(raw.data, &out.shape, &out.strides, out.dtype.itemsize()) };
        drop(buf);
        results.push(Array {
            dtype: out.dtype,
            shape: out.shape.clone(),
            bytes,
        });
    }
    match error {
        Some(message) => Err(message),
        None => Ok(results),
    }
}
