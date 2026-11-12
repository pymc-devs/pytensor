//! Arrays of the kernel ABI, as Rust sees them.
//!
//! A `PtArray` (`crate::abi`) is five pointers and two numbers that anyone can write.
//! The types here are what it means once somebody has vouched for it:
//!
//! * `PtArrayRef<'a>`: an array to read,
//! * `PtArrayMut<'a>`: an array to write,
//! * `OutSlot<'a>`: an output of a kernel call, before it is produced,
//! * `OwnedArray`: an array in a buffer we hold a reference to.
//!
//! Each has an invariant (in its docs), and making one from raw parts is `unsafe`: the
//! caller of the constructor promises the invariant, or, for the `checked`
//! constructors, only the part of it that can't be looked at (that the memory is
//! there). Their methods are safe. So the places that take something on trust are the
//! constructor calls, and each says why it may.
//!
//! Elements are plain data: every dtype is a type all of whose bit patterns are valid,
//! except bool, which only kernels that were made for bools write. Memory to write may
//! hold anything, also nothing yet (a fresh output): kernels write before they read.

use crate::abi::{
    MAX_NDIM, OwnedBuf, PtArray, PtBuffer, bad_size, c_strides, checked_nbytes, is_c_contiguous,
    validate_array,
};
use crate::dtype::DType;
use crate::runtime::PtCtx;
use std::marker::PhantomData;

/// The shape, strides and dtype in the header of `a`.
///
/// # Safety
/// If `a.ndim` is in `1..=MAX_NDIM`, `a.shape` and `a.strides` are null or point to
/// `a.ndim` i64s each, readable and not written for `'a`.
#[inline]
unsafe fn header<'a>(a: &PtArray) -> Result<(&'a [i64], &'a [i64], DType), String> {
    let ndim = a.ndim as usize;
    let (Some(dtype), true) = (
        DType::from_code(a.dtype),
        // (a negative ndim is a huge one)
        ndim <= MAX_NDIM && (ndim == 0 || !(a.shape.is_null() || a.strides.is_null())),
    ) else {
        return Err(bad_header(a));
    };
    if ndim == 0 {
        return Ok((&[], &[], dtype));
    }
    // SAFETY: non-null, and `ndim` readable i64s by the function's contract.
    let (shape, strides) = unsafe {
        (
            std::slice::from_raw_parts(a.shape, ndim),
            std::slice::from_raw_parts(a.strides, ndim),
        )
    };
    Ok((shape, strides, dtype))
}

/// What is wrong with a header `header` rejects. (Apart: errors are rare, and the
/// checks are on the way of every kernel call.)
#[cold]
#[inline(never)]
fn bad_header(a: &PtArray) -> String {
    if DType::from_code(a.dtype).is_none() {
        format!("unknown dtype code {}", a.dtype)
    } else if a.ndim < 0 || a.ndim as usize > MAX_NDIM {
        format!("an array with {} axes", a.ndim)
    } else {
        "shape and strides must not be NULL".into()
    }
}

// ------------------------------------------------------------------ arrays to read

/// An array to read.
///
/// Invariant: `shape` and `strides` (in bytes) have one entry per axis and no length is
/// negative. Unless the array is empty, `data` is aligned to the itemsize of `dtype`,
/// the strides of axes longer than 1 are multiples of it, and every element is
/// readable memory holding a value of `dtype`, which nothing writes during `'a`.
/// `owner` is null or the buffer the memory is in, alive during `'a`.
#[derive(Clone, Copy)]
pub struct PtArrayRef<'a> {
    data: *const u8,
    shape: &'a [i64],
    strides: &'a [i64],
    dtype: DType,
    owner: *mut PtBuffer,
}

impl<'a> PtArrayRef<'a> {
    /// # Safety
    /// The invariant of the type: see there. (`shape` and `strides` have the same
    /// length.)
    pub unsafe fn new(
        data: *const u8,
        shape: &'a [i64],
        strides: &'a [i64],
        dtype: DType,
    ) -> PtArrayRef<'a> {
        debug_assert_eq!(shape.len(), strides.len());
        PtArrayRef {
            data,
            shape,
            strides,
            dtype,
            owner: std::ptr::null_mut(),
        }
    }

    /// An array somebody else describes: everything that can be looked at is checked
    /// (lengths, alignment, strides, that the address range makes sense).
    ///
    /// # Safety
    /// If the checks pass, the memory the arguments describe is readable and holds
    /// values of `dtype`, and nothing writes it during `'a`.
    pub unsafe fn checked(
        data: *const u8,
        shape: &'a [i64],
        strides: &'a [i64],
        dtype: DType,
    ) -> Result<PtArrayRef<'a>, String> {
        if shape.len() != strides.len() || shape.len() > MAX_NDIM {
            return Err(format!(
                "an array with {} lengths and {} strides",
                shape.len(),
                strides.len()
            ));
        }
        validate_array(data, shape, strides, dtype.itemsize())?;
        // SAFETY: just checked, and the function's contract for the memory.
        Ok(unsafe { PtArrayRef::new(data, shape, strides, dtype) })
    }

    /// The array a `PtArray` of the ABI describes. Only its header is checked (the
    /// number of axes, the dtype code).
    ///
    /// # Safety
    /// `a.shape` and `a.strides` point to `a.ndim` i64s each that nothing writes during
    /// `'a`, and with them the invariant of the type holds. `a.owner` is null or a live
    /// buffer during `'a`.
    #[inline]
    pub unsafe fn from_raw(a: &PtArray) -> Result<PtArrayRef<'a>, String> {
        // SAFETY: the function's contract.
        let (shape, strides, dtype) = unsafe { header(a)? };
        Ok(PtArrayRef {
            data: a.data,
            shape,
            strides,
            dtype,
            owner: a.owner,
        })
    }

    /// As `from_raw`, for a `PtArray` somebody else filled in: checked as in `checked`.
    ///
    /// # Safety
    /// `a.shape` and `a.strides` are null or point to `a.ndim` i64s each that nothing
    /// writes during `'a`. If the checks pass, the memory described is readable and
    /// holds values of the dtype, and nothing writes it during `'a`. `a.owner` is null
    /// or a live buffer during `'a`.
    pub unsafe fn from_raw_checked(a: &PtArray) -> Result<PtArrayRef<'a>, String> {
        // SAFETY: the function's contract.
        let (shape, strides, dtype) = unsafe { header(a)? };
        validate_array(a.data, shape, strides, dtype.itemsize())?;
        Ok(PtArrayRef {
            data: a.data,
            shape,
            strides,
            dtype,
            owner: a.owner,
        })
    }

    pub fn shape(&self) -> &'a [i64] {
        self.shape
    }

    /// Strides in bytes.
    pub fn strides(&self) -> &'a [i64] {
        self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn itemsize(&self) -> usize {
        self.dtype.itemsize()
    }

    /// The first element (dangling or null if the array is empty).
    pub fn as_ptr(&self) -> *const u8 {
        self.data
    }

    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }

    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(self.shape, self.strides, self.itemsize())
    }

    /// The byte range `[lo, hi)` the elements lie in (empty for an empty array).
    pub fn extent(&self) -> (usize, usize) {
        extent(self.data, self.shape, self.strides, self.itemsize())
    }

    /// Index `i` along the first axis: an array with one axis less. None if there is
    /// no axis or `i` is out of range.
    pub fn row(&self, i: i64) -> Option<PtArrayRef<'a>> {
        let (&n, &stride) = (self.shape.first()?, self.strides.first()?);
        if i < 0 || i >= n {
            return None;
        }
        Some(PtArrayRef {
            // (inside the array: 0 <= i < n)
            data: self.data.wrapping_offset((i * stride) as isize),
            shape: &self.shape[1..],
            strides: &self.strides[1..],
            dtype: self.dtype,
            owner: self.owner,
        })
    }

    /// The value of an array with exactly one element, as an integer (floats are
    /// truncated, non-finite ones are 0; a bool is 0 or 1).
    pub fn item_as_i64(&self) -> Result<i64, String> {
        if self.shape.iter().any(|&n| n != 1) {
            return Err(format!(
                "one value is expected, got an array of shape {:?}",
                self.shape
            ));
        }
        // SAFETY: the array has one element, a valid aligned value of `dtype` (the
        // type's invariant).
        Ok(unsafe { crate::interp::consts::read_scalar(self.data, self.dtype).1 })
    }
}

fn extent(data: *const u8, shape: &[i64], strides: &[i64], itemsize: usize) -> (usize, usize) {
    // (can't fail for arrays that keep their type's invariant)
    validate_array(data, shape, strides, itemsize).unwrap_or((data as usize, data as usize))
}

// ------------------------------------------------------------------ arrays to write

/// An array to write.
///
/// Invariant: as for `PtArrayRef`, with memory that is writable and that nothing else
/// reads or writes during `'a`. The memory need not hold values yet.
pub struct PtArrayMut<'a> {
    data: *mut u8,
    shape: &'a [i64],
    strides: &'a [i64],
    dtype: DType,
    _memory: PhantomData<&'a mut [u8]>,
}

impl<'a> PtArrayMut<'a> {
    /// # Safety
    /// The invariant of the type: see there. (`shape` and `strides` have the same
    /// length.)
    pub unsafe fn new(
        data: *mut u8,
        shape: &'a [i64],
        strides: &'a [i64],
        dtype: DType,
    ) -> PtArrayMut<'a> {
        debug_assert_eq!(shape.len(), strides.len());
        PtArrayMut {
            data,
            shape,
            strides,
            dtype,
            _memory: PhantomData,
        }
    }

    /// As `PtArrayRef::from_raw_checked`, for memory to write.
    ///
    /// # Safety
    /// `a.shape` and `a.strides` are null or point to `a.ndim` i64s each that nothing
    /// writes during `'a`. If the checks pass, the memory described is writable, and
    /// nothing else reads or writes it during `'a`.
    pub unsafe fn from_raw_checked(a: &PtArray) -> Result<PtArrayMut<'a>, String> {
        // SAFETY: the function's contract.
        let (shape, strides, dtype) = unsafe { header(a)? };
        validate_array(a.data, shape, strides, dtype.itemsize())?;
        // SAFETY: just checked, and the function's contract for the memory.
        Ok(unsafe { PtArrayMut::new(a.data, shape, strides, dtype) })
    }

    pub fn shape(&self) -> &'a [i64] {
        self.shape
    }

    /// Strides in bytes.
    pub fn strides(&self) -> &'a [i64] {
        self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn itemsize(&self) -> usize {
        self.dtype.itemsize()
    }

    /// The first element (dangling or null if the array is empty).
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data
    }

    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(self.shape, self.strides, self.itemsize())
    }

    /// The byte range `[lo, hi)` the elements lie in (empty for an empty array).
    pub fn extent(&self) -> (usize, usize) {
        extent(self.data, self.shape, self.strides, self.itemsize())
    }

    /// The same array, to read.
    pub fn as_ref(&self) -> PtArrayRef<'_> {
        // SAFETY: our invariant is the one of `PtArrayRef`, and while the result
        // borrows `self` nothing writes through it.
        unsafe { PtArrayRef::new(self.data, self.shape, self.strides, self.dtype) }
    }

    /// The same array, for a shorter time.
    pub fn reborrow(&mut self) -> PtArrayMut<'_> {
        PtArrayMut { ..*self }
    }

    /// Index `i` along the first axis: an array with one axis less. None if there is
    /// no axis or `i` is out of range.
    pub fn row(self, i: i64) -> Option<PtArrayMut<'a>> {
        let (&n, &stride) = (self.shape.first()?, self.strides.first()?);
        if i < 0 || i >= n {
            return None;
        }
        Some(PtArrayMut {
            // (inside the array: 0 <= i < n)
            data: self.data.wrapping_offset((i * stride) as isize),
            shape: &self.shape[1..],
            strides: &self.strides[1..],
            dtype: self.dtype,
            _memory: PhantomData,
        })
    }

    /// Copy the elements of `source`, an array of the same shape and dtype.
    pub fn copy_from(&mut self, source: &PtArrayRef) -> Result<(), String> {
        if source.dtype != self.dtype {
            return Err(format!(
                "can't copy an array of {} into one of {}",
                source.dtype.numpy_name(),
                self.dtype.numpy_name()
            ));
        }
        if source.shape != self.shape {
            return Err(format!(
                "can't copy an array of shape {:?} into one of shape {:?}",
                source.shape, self.shape
            ));
        }
        // SAFETY: both are valid memory of this shape with their strides, and memory
        // to write overlaps nothing else (the invariants of the two types).
        unsafe {
            copy_strided(
                source.data,
                self.shape,
                source.strides,
                self.data,
                self.strides,
                self.itemsize(),
            )
        };
        Ok(())
    }

    /// Set every byte of every element to zero.
    pub fn zero(&mut self) {
        let itemsize = self.itemsize();
        if self.is_c_contiguous() {
            let n = checked_nbytes(self.shape, itemsize).unwrap_or(0);
            if n > 0 {
                // SAFETY: `n` adjacent bytes, ours to write (the type's invariant).
                unsafe { std::ptr::write_bytes(self.data, 0, n) };
            }
            return;
        }
        let data = self.data;
        for_each_offset(self.shape, self.strides, |offset| {
            // SAFETY: the offset of an element, ours to write (the type's invariant).
            unsafe { std::ptr::write_bytes(data.offset(offset), 0, itemsize) };
        });
    }
}

/// Call `f` with the byte offset of every element of an array, in C order.
fn for_each_offset(shape: &[i64], strides: &[i64], mut f: impl FnMut(isize)) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0i64; shape.len()];
    let mut offset = 0isize;
    loop {
        f(offset);
        let mut axis = shape.len();
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            index[axis] += 1;
            offset += strides[axis] as isize;
            if index[axis] < shape[axis] {
                break;
            }
            offset -= (strides[axis] * shape[axis]) as isize;
            index[axis] = 0;
        }
    }
}

/// Copy elements between two arrays of the same shape.
///
/// # Safety
/// Both describe valid memory of `shape` with their strides; they don't overlap.
unsafe fn copy_strided(
    src: *const u8,
    shape: &[i64],
    src_strides: &[i64],
    dst: *mut u8,
    dst_strides: &[i64],
    itemsize: usize,
) {
    if shape.contains(&0) {
        return;
    }
    if is_c_contiguous(shape, src_strides, itemsize)
        && is_c_contiguous(shape, dst_strides, itemsize)
    {
        let n = shape.iter().product::<i64>() as usize * itemsize;
        // SAFETY: both are `n` adjacent bytes (the function's contract).
        unsafe { std::ptr::copy_nonoverlapping(src, dst, n) };
        return;
    }
    let nd = shape.len();
    let mut idx = vec![0i64; nd];
    loop {
        let (mut so, mut d) = (0isize, 0isize);
        for a in 0..nd {
            so += (idx[a] * src_strides[a]) as isize;
            d += (idx[a] * dst_strides[a]) as isize;
        }
        // SAFETY: the offsets stay inside both arrays (index < shape on every axis).
        unsafe { std::ptr::copy_nonoverlapping(src.offset(so), dst.offset(d), itemsize) };
        let mut a = nd;
        loop {
            if a == 0 {
                return;
            }
            a -= 1;
            idx[a] += 1;
            if idx[a] < shape[a] {
                break;
            }
            idx[a] = 0;
        }
    }
}

// ------------------------------------------------------------------ outputs

/// An output of a kernel call that is still to be produced: the `PtArray` the caller
/// passed for it.
///
/// Invariant: the header's `shape` and `strides` point to storage of its `ndim` that is
/// ours alone to read and write during `'a`, and its dtype code is valid. If `data` is
/// not null the caller provides the memory: with the header's shape and strides it is
/// an array to write (the invariant of `PtArrayMut`). If `data` is null the kernel
/// produces the output, and what it puts there is the caller's from then on: a buffer
/// reference, or a view of memory the caller keeps alive.
pub struct OutSlot<'a> {
    raw: &'a mut PtArray,
    dtype: DType,
}

impl<'a> OutSlot<'a> {
    /// # Safety
    /// The invariant of the type: see there. Only the header is checked (the number
    /// of axes, the dtype code, non-null shape and strides).
    #[inline]
    pub unsafe fn from_raw(raw: &'a mut PtArray) -> Result<OutSlot<'a>, String> {
        // SAFETY: the function's contract (the storage is ours during `'a`; the
        // slices made here are dropped at once).
        let (_, _, dtype) = unsafe { header(raw)? };
        Ok(OutSlot { raw, dtype })
    }

    pub fn ndim(&self) -> usize {
        self.raw.ndim as usize
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Whether the caller provides the memory (otherwise the kernel produces the
    /// output).
    pub fn is_provided(&self) -> bool {
        !self.raw.data.is_null()
    }

    /// The header's shape and strides storage.
    #[inline]
    fn storage(&mut self) -> (&mut [i64], &mut [i64]) {
        let ndim = self.ndim();
        if ndim == 0 {
            return (&mut [], &mut []);
        }
        // SAFETY: two distinct storages of `ndim` i64s that are ours alone while the
        // slot is borrowed (the type's invariant; `from_raw` checked they aren't
        // null).
        unsafe {
            (
                std::slice::from_raw_parts_mut(self.raw.shape, ndim),
                std::slice::from_raw_parts_mut(self.raw.strides, ndim),
            )
        }
    }

    /// The header's shape and strides, for as long as the slot was good for.
    #[inline]
    fn into_storage(self) -> (&'a [i64], &'a [i64]) {
        let ndim = self.ndim();
        if ndim == 0 {
            return (&[], &[]);
        }
        // SAFETY: as in `storage`; the slot is gone, so nothing writes them any more
        // during `'a`.
        unsafe {
            (
                std::slice::from_raw_parts(self.raw.shape, ndim),
                std::slice::from_raw_parts(self.raw.strides, ndim),
            )
        }
    }

    #[inline]
    fn check(&self, what: &str, dtype: DType, ndim: usize) -> Result<(), String> {
        if dtype != self.dtype {
            return Err(format!(
                "the output is an array of {}, {what} one of {}",
                self.dtype.numpy_name(),
                dtype.numpy_name()
            ));
        }
        if ndim != self.ndim() {
            return Err(format!(
                "the output has {} axes, {what} {ndim}",
                self.ndim()
            ));
        }
        Ok(())
    }

    #[inline]
    fn not_provided(&self) -> Result<(), String> {
        if self.is_provided() {
            return Err(
                "the caller provides memory for the output, but the kernel was made to \
                 return it"
                    .into(),
            );
        }
        Ok(())
    }

    /// The memory the caller provides.
    #[inline]
    pub fn provided(self) -> Result<PtArrayMut<'a>, String> {
        let (data, dtype) = (self.raw.data, self.dtype);
        let (shape, strides) = self.into_storage();
        if data.is_null() && !shape.contains(&0) {
            return Err(
                "the caller provides no memory for the output, but the kernel was made \
                 to write into it"
                    .into(),
            );
        }
        // SAFETY: the type's invariant for provided memory.
        Ok(unsafe { PtArrayMut::new(data, shape, strides, dtype) })
    }

    /// Produce the output in a new C-contiguous buffer of `shape` (not initialized),
    /// which is returned to write it.
    pub fn alloc(mut self, ctx: &mut PtCtx, shape: &[i64]) -> Result<PtArrayMut<'a>, String> {
        self.not_provided()?;
        self.check("the kernel makes", self.dtype, shape.len())?;
        let itemsize = self.dtype.itemsize();
        let n = checked_nbytes(shape, itemsize).ok_or_else(|| bad_size("output", shape))?;
        let buf = ctx
            .alloc(n)
            .ok_or_else(|| format!("could not allocate {n} bytes"))?;
        let (out_shape, out_strides) = self.storage();
        out_shape.copy_from_slice(shape);
        let mut stride = itemsize as i64;
        for (out_stride, &n) in out_strides.iter_mut().zip(shape).rev() {
            *out_stride = stride;
            stride = stride.saturating_mul(n.max(1));
        }
        let (data, dtype) = (buf.data(), self.dtype);
        self.raw.data = data;
        self.raw.owner = buf.into_raw();
        let (shape, strides) = self.into_storage();
        // SAFETY: a new buffer of this C-contiguous shape, 64-byte aligned. Its one
        // reference is in the output's header, which nobody touches during `'a` (the
        // slot is gone), so the memory stays, and it is ours alone.
        Ok(unsafe { PtArrayMut::new(data, shape, strides, dtype) })
    }

    /// The output is `array`: its buffer reference goes to the caller.
    pub fn set(mut self, array: OwnedArray) -> Result<(), String> {
        self.not_provided()?;
        self.check("the kernel returns", array.dtype, array.shape.len())?;
        let (shape, strides) = self.storage();
        shape.copy_from_slice(&array.shape);
        strides.copy_from_slice(&array.strides);
        self.raw.data = array.data;
        self.raw.owner = array.buf.into_raw();
        Ok(())
    }

    /// The output is a view of `source`: its memory, seen with other axes (per output
    /// axis, the axis of `source` it is, or None for a new axis of length 1).
    ///
    /// # Safety
    /// The caller of the kernel keeps the memory of `source` alive for as long as it
    /// uses this output: `source` is an input of the same call.
    pub unsafe fn set_view(
        mut self,
        source: &PtArrayRef,
        axes: &[Option<usize>],
    ) -> Result<(), String> {
        self.not_provided()?;
        self.check("the kernel returns a view of", source.dtype, axes.len())?;
        if let Some(axis) = axes.iter().flatten().find(|&&axis| axis >= source.ndim()) {
            return Err(format!(
                "axis {axis} of an array with {} axes",
                source.ndim()
            ));
        }
        // an axis the view drops must have length 1 (the type may not say so)
        if let Some(a) =
            (0..source.ndim()).find(|&a| source.shape[a] != 1 && !axes.contains(&Some(a)))
        {
            return Err(format!(
                "cannot drop axis {a} of length {} (not 1)",
                source.shape[a]
            ));
        }
        let (shape, strides) = self.storage();
        for (k, axis) in axes.iter().enumerate() {
            shape[k] = axis.map_or(1, |axis| source.shape[axis]);
            strides[k] = axis.map_or(0, |axis| source.strides[axis]);
        }
        self.raw.data = source.data as *mut u8;
        self.raw.owner = source.owner;
        Ok(())
    }
}

// ------------------------------------------------------------------ owned arrays

/// An array in a buffer we hold a reference to.
///
/// Invariant: `data`, `shape` and `strides` describe an array of `dtype` (as in
/// `PtArrayRef`) whose elements all lie in `buf`'s memory.
pub struct OwnedArray {
    buf: OwnedBuf,
    data: *mut u8,
    shape: Vec<i64>,
    strides: Vec<i64>,
    dtype: DType,
}

// SAFETY: a buffer reference (`OwnedBuf` is Send) and a pointer into that buffer.
unsafe impl Send for OwnedArray {}

impl OwnedArray {
    /// A new C-contiguous array of zeros.
    pub fn zeros(dtype: DType, shape: Vec<i64>) -> Result<OwnedArray, String> {
        let n =
            checked_nbytes(&shape, dtype.itemsize()).ok_or_else(|| bad_size("array", &shape))?;
        let buf = OwnedBuf::zeroed(n).ok_or_else(|| format!("could not allocate {n} bytes"))?;
        let strides = c_strides(&shape, dtype.itemsize() as i64);
        Ok(OwnedArray {
            data: buf.data(),
            buf,
            shape,
            strides,
            dtype,
        })
    }

    /// A new C-contiguous array holding the elements of `source`.
    pub fn copy_of(source: &PtArrayRef) -> Result<OwnedArray, String> {
        let shape = source.shape().to_vec();
        let n =
            checked_nbytes(&shape, source.itemsize()).ok_or_else(|| bad_size("array", &shape))?;
        let buf = OwnedBuf::new(n).ok_or_else(|| format!("could not allocate {n} bytes"))?;
        let strides = c_strides(&shape, source.itemsize() as i64);
        let mut array = OwnedArray {
            data: buf.data(),
            buf,
            shape,
            strides,
            dtype: source.dtype(),
        };
        array.as_mut()?.copy_from(source)?;
        Ok(array)
    }

    /// The array `data`, `shape`, `strides` in `buf`. Checked: that it is a valid
    /// array all of whose elements lie in the buffer.
    pub fn in_buffer(
        buf: OwnedBuf,
        data: *mut u8,
        shape: Vec<i64>,
        strides: Vec<i64>,
        dtype: DType,
    ) -> Result<OwnedArray, String> {
        if shape.len() != strides.len() {
            return Err("one stride per axis is expected".into());
        }
        let (lo, hi) = validate_array(data, &shape, &strides, dtype.itemsize())?;
        let (start, end) = buf.range();
        if lo < hi && (lo < start || hi > end) {
            return Err("the array isn't inside its buffer".into());
        }
        Ok(OwnedArray {
            buf,
            data,
            shape,
            strides,
            dtype,
        })
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The length of the first axis (None without axes).
    pub fn rows(&self) -> Option<i64> {
        self.shape.first().copied()
    }

    /// Keep only the first `rows` indices of the first axis (the same memory).
    pub fn truncate(&mut self, rows: i64) {
        if let Some(n) = self.shape.first_mut() {
            *n = rows.clamp(0, *n);
        }
    }

    /// The array, to read.
    ///
    /// Others may hold the buffer too. They only read it: writing takes `as_mut`,
    /// which only the buffer's one holder gets, or the raw parts (`into_parts`),
    /// whose users answer for what they do.
    pub fn as_ref(&self) -> PtArrayRef<'_> {
        // SAFETY: the type's invariant; the buffer lives as long as `self`, and for
        // writers see above.
        unsafe { PtArrayRef::new(self.data, &self.shape, &self.strides, self.dtype) }
    }

    /// The array, to write: only if nobody else holds its buffer.
    pub fn as_mut(&mut self) -> Result<PtArrayMut<'_>, String> {
        if !self.buf.is_unique() {
            return Err("can't write an array whose buffer is shared".into());
        }
        // SAFETY: the type's invariant; we hold the buffer's only reference, and
        // `self` is borrowed for as long as the result lives, so nothing else reads
        // or writes the memory.
        Ok(unsafe { PtArrayMut::new(self.data, &self.shape, &self.strides, self.dtype) })
    }

    /// (the buffer reference, data, shape, strides), for handing the array on (the C
    /// API, numpy).
    pub fn into_parts(self) -> (OwnedBuf, *mut u8, Vec<i64>, Vec<i64>) {
        (self.buf, self.data, self.shape, self.strides)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(data: &mut [f64], shape: &mut [i64], strides: &mut [i64]) -> PtArray {
        PtArray {
            data: data.as_mut_ptr() as *mut u8,
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            owner: std::ptr::null_mut(),
            ndim: shape.len() as i32,
            dtype: DType::Float64.code(),
        }
    }

    #[test]
    fn headers_are_checked() {
        let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (mut shape, mut strides) = ([2i64, 3], [24i64, 8]);
        let good = raw(&mut values, &mut shape, &mut strides);
        // SAFETY: `good` describes `values`, which nothing writes.
        let a = unsafe { PtArrayRef::from_raw_checked(&good) }.unwrap();
        assert_eq!((a.shape(), a.dtype()), (&[2i64, 3][..], DType::Float64));
        let (lo, hi) = a.extent();
        assert_eq!(hi - lo, 48);
        let row = a.row(1).unwrap();
        assert_eq!(row.shape(), &[3]);
        assert_eq!(row.as_ptr() as usize - lo, 24);
        assert!(a.row(2).is_none());

        let bad = |change: fn(&mut PtArray)| {
            let mut a = good;
            change(&mut a);
            // SAFETY: as `good`, changed so that a check fails before any memory is
            // used.
            unsafe { PtArrayRef::from_raw_checked(&a) }.err().unwrap()
        };
        assert!(bad(|a| a.dtype = 99).contains("dtype code"));
        assert!(bad(|a| a.ndim = -1).contains("axes"));
        assert!(bad(|a| a.ndim = 1000).contains("axes"));
        assert!(bad(|a| a.shape = std::ptr::null_mut()).contains("NULL"));
        assert!(bad(|a| a.data = std::ptr::null_mut()).contains("null data"));
        assert!(bad(|a| a.data = a.data.wrapping_add(4)).contains("aligned"));
    }

    #[test]
    fn copies_between_layouts() {
        // 2 x 3, read column by column into a C-contiguous array
        let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (mut shape, mut strides) = ([2i64, 3], [8i64, 16]);
        let source = raw(&mut values, &mut shape, &mut strides);
        // SAFETY: describes `values`, which nothing writes.
        let source = unsafe { PtArrayRef::from_raw_checked(&source) }.unwrap();
        let mut copy = OwnedArray::copy_of(&source).unwrap();
        assert!(copy.as_ref().is_c_contiguous());
        let read = |a: &PtArrayRef| -> Vec<i64> {
            (0..2)
                .flat_map(|i| (0..3).map(move |j| (i, j)))
                .map(|(i, j)| a.row(i).unwrap().row(j).unwrap().item_as_i64().unwrap())
                .collect()
        };
        assert_eq!(read(&copy.as_ref()), [1, 3, 5, 2, 4, 6]);
        copy.as_mut().unwrap().row(0).unwrap().zero();
        assert_eq!(read(&copy.as_ref()), [0, 0, 0, 2, 4, 6]);

        // of another shape or dtype: an error
        let mut other = OwnedArray::zeros(DType::Float64, vec![3, 2]).unwrap();
        assert!(other.as_mut().unwrap().copy_from(&source).is_err());
        let mut other = OwnedArray::zeros(DType::Int64, vec![2, 3]).unwrap();
        assert!(other.as_mut().unwrap().copy_from(&source).is_err());
    }

    #[test]
    fn slots_are_produced_once_and_as_the_caller_asked() {
        let mut ctx = PtCtx::new();
        let (mut shape, mut strides) = ([2i64, 3], [0i64; 2]);
        let mut header = PtArray {
            data: std::ptr::null_mut(),
            shape: shape.as_mut_ptr(),
            strides: strides.as_mut_ptr(),
            owner: std::ptr::null_mut(),
            ndim: 2,
            dtype: DType::Float32.code(),
        };
        fn slot(header: &mut PtArray) -> OutSlot<'_> {
            // SAFETY: the header of this test: shape and strides storage of two axes,
            // ours; and no memory, or what a slot made of it allocated before.
            unsafe { OutSlot::from_raw(header) }.unwrap()
        }
        assert!(slot(&mut header).provided().is_err());
        assert!(slot(&mut header).alloc(&mut ctx, &[2, 3, 4]).is_err());
        let wrong = OwnedArray::zeros(DType::Float64, vec![2, 3]).unwrap();
        let error = slot(&mut header).set(wrong).unwrap_err();
        assert!(error.contains("float32"), "{error}");
        assert!(header.data.is_null() && header.owner.is_null());

        let mut array = slot(&mut header).alloc(&mut ctx, &[2, 3]).unwrap();
        array.zero();
        assert_eq!(
            (array.shape(), array.strides()),
            (&[2i64, 3][..], &[12i64, 4][..])
        );
        assert!(!header.owner.is_null());
        // now the memory is there: the slot can't be produced again
        assert!(slot(&mut header).alloc(&mut ctx, &[2, 3]).is_err());
        // SAFETY: the reference `alloc` put in the header.
        drop(unsafe { OwnedBuf::from_raw(header.owner) });
    }

    #[test]
    fn shared_buffers_are_read_only() {
        let mut a = OwnedArray::zeros(DType::Int32, vec![4]).unwrap();
        assert!(a.as_mut().is_ok());
        let (buf, data, shape, strides) = a.into_parts();
        let mut one = OwnedArray::in_buffer(
            buf.clone(),
            data,
            shape.clone(),
            strides.clone(),
            DType::Int32,
        )
        .unwrap();
        assert!(one.as_mut().is_err());
        // an array that doesn't fit in the buffer
        let size = buf.size() as i64;
        assert!(OwnedArray::in_buffer(buf, data, vec![size], strides, DType::Int32).is_err());
        assert!(one.as_mut().is_ok());
    }
}
