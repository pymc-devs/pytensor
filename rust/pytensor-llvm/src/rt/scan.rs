//! Scan as a kernel: a Rust loop around the step function, which is a nested program.
//!
//! Buffer conventions follow PyTensor's Scan (and its numba implementation): every
//! traced output (mit-mot, mit-sot, sit-sot) has a buffer whose first rows hold the
//! initial taps; the step reads tap `k` at row `(i + lookback + k) % len` and writes at
//! `(i + lookback + out_tap) % len`. After the loop, buffers shorter than the full trace
//! are rotated so the last value is at the end, while-loops are truncated to the steps
//! taken, and unused rows of regular loops are zeroed. nit-sot buffers are allocated once
//! the first step's output shape is known.
//!
//! Step outputs with static shapes are written into memory this kernel provides; the
//! others come back as buffers of the step program. Either way they are copied into
//! their rows, so a circular buffer whose output row coincides with an input tap is
//! still correct.

use super::program::{copy_strided, Program};
use crate::abi::{OwnedBuf, PtArray, PtBuffer, PtCtx};
use crate::dtype::DType;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ScanInfo {
    pub n_seqs: usize,
    pub mit_mot_in: Vec<Vec<i64>>,
    pub mit_mot_out: Vec<Vec<i64>>,
    pub mit_sot_in: Vec<Vec<i64>>,
    pub sit_sot_in: Vec<Vec<i64>>,
    pub n_untraced: usize,
    pub n_nit_sot: usize,
    pub n_non_seqs: usize,
    pub as_while: bool,
}

pub struct ScanEnv {
    pub info: ScanInfo,
    pub step: Arc<Program>,
    pub outer_in_dtypes: Vec<DType>,
    pub inner_out_dtypes: Vec<DType>,
    /// the Python step program: keeps its compiled code and kernels alive
    pub keepalive: pyo3::Py<pyo3::PyAny>,
}

/// A C-contiguous buffer owned by the driver (until handed out as an output).
struct Buf {
    owner: *mut PtBuffer,
    data: *mut u8,
    shape: Vec<i64>,
    strides: Vec<i64>,
    itemsize: i64,
}

fn c_strides(shape: &[i64], itemsize: i64) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    let mut acc = itemsize;
    for d in (0..shape.len()).rev() {
        strides[d] = acc;
        acc *= shape[d].max(1);
    }
    strides
}

impl Buf {
    /// The buffer reference is pushed onto `owned` (released on errors).
    fn new(shape: Vec<i64>, itemsize: i64, owned: &mut Vec<OwnedBuf>) -> Result<Buf, String> {
        let nbytes = crate::abi::checked_nbytes(&shape, itemsize as usize).ok_or_else(|| crate::abi::bad_size("scan buffer", &shape))?;
        let owner = PtBuffer::new(nbytes, 64);
        // SAFETY: a buffer reference we just created and now own.
        owned.push(unsafe { OwnedBuf::from_raw(owner) });
        let strides = c_strides(&shape, itemsize);
        // SAFETY: `owner` was just allocated.
        Ok(Buf { owner, data: unsafe { (*owner).data }, shape, strides, itemsize })
    }

    fn len(&self) -> i64 {
        self.shape[0]
    }

    fn row_bytes(&self) -> i64 {
        self.shape[1..].iter().product::<i64>() * self.itemsize
    }

    fn row(&self, r: i64) -> *mut u8 {
        // SAFETY: 0 <= r < len (callers index rows modulo the length): inside the buffer.
        unsafe { self.data.offset((r * self.row_bytes()) as isize) }
    }

    fn view(&mut self, r: Option<i64>) -> PtArray {
        match r {
            None => PtArray { data: self.data, shape: self.shape.as_mut_ptr(), strides: self.strides.as_mut_ptr(), owner: std::ptr::null_mut(), ndim: self.shape.len() as i64 },
            Some(r) => PtArray {
                data: self.row(r),
                shape: self.shape[1..].as_mut_ptr(),
                strides: self.strides[1..].as_mut_ptr(),
                owner: std::ptr::null_mut(),
                ndim: self.shape.len() as i64 - 1,
            },
        }
    }

    /// Copy a valid array of this buffer's row shape into row `r`.
    fn write_row(&self, r: i64, src: &PtArray) -> Result<(), String> {
        // SAFETY: `src` is a valid array (a step output or input).
        let (sh, st) = unsafe { (src.shape(), src.strides()) };
        if sh != &self.shape[1..] {
            return Err(format!("Scan: step output has shape {sh:?}, expected {:?}", &self.shape[1..]));
        }
        // SAFETY: row r has this shape (C-contiguous) and doesn't overlap `src` (step
        // outputs are fresh or provided temporaries).
        unsafe { copy_strided(src.data, sh, st, self.row(r), &self.strides[1..], self.itemsize as usize) };
        Ok(())
    }
}

/// # Safety
/// `a` is a valid 0-d array of `dt`.
unsafe fn read_int(a: &PtArray, dt: DType) -> i64 {
    // SAFETY: the function's contract.
    unsafe { super::consts::read_scalar(a.data, dt).1 }
}

struct Traced {
    buf: Buf,
    lookback: i64,
    in_taps: Vec<i64>,
    out_taps: Vec<i64>,
    post_process: bool,
}

/// args = outer inputs ++ outer outputs (all allocated by this kernel).
///
/// # Safety
/// The kernel ABI; `env` is a `ScanEnv` matching the arguments.
pub unsafe extern "C-unwind" fn k_scan(ctx: *mut PtCtx, args: *mut PtArray, env: *const u8, _ws: *mut u8) -> i32 {
    // SAFETY: the function's contract.
    let (d, c) = unsafe { (&*(env as *const ScanEnv), &mut *ctx) };
    let mut owned: Vec<OwnedBuf> = Vec::new();
    // SAFETY: the function's contract.
    match unsafe { run(c, args, d, &mut owned) } {
        Ok(()) => {
            // the references now live in the outputs' owner fields
            owned.into_iter().for_each(|b| {
                b.into_raw();
            });
            0
        }
        // SAFETY: ctx is the live context.
        Err(m) => unsafe { PtCtx::fail(ctx, 1, &m) },
    }
}

/// # Safety
/// As for `k_scan`.
unsafe fn run(ctx: &mut PtCtx, args: *mut PtArray, d: &ScanEnv, owned: &mut Vec<OwnedBuf>) -> Result<(), String> {
    let info = &d.info;
    // SAFETY: args holds every outer input and output (the contract).
    let arg = |i: usize| unsafe { *args.add(i) };
    let (n_mm, n_ms, n_ss) = (info.mit_mot_in.len(), info.mit_sot_in.len(), info.sit_sot_in.len());
    let o_seq = 1;
    let o_mm = o_seq + info.n_seqs;
    let o_ms = o_mm + n_mm;
    let o_ss = o_ms + n_ms;
    let o_ut = o_ss + n_ss;
    let o_nit = o_ut + info.n_untraced;
    let o_ns = o_nit + info.n_nit_sot;
    let n_outer_in = o_ns + info.n_non_seqs;
    // SAFETY: n_steps is a valid 0-d integer array.
    let n_steps = unsafe { read_int(&arg(0), d.outer_in_dtypes[0]) }.max(0);
    let step = &*d.step;

    // ---- buffers: copies of the outer buffers (the loop writes into them)
    let copy_of = |a: PtArray, dt: DType, owned: &mut Vec<OwnedBuf>| -> Result<Buf, String> {
        // SAFETY: outer inputs are valid arrays.
        let (sh, st) = unsafe { (a.shape().to_vec(), a.strides().to_vec()) };
        let b = Buf::new(sh.clone(), dt.itemsize() as i64, owned)?;
        // SAFETY: a valid array and a fresh buffer of its shape.
        unsafe { copy_strided(a.data, &sh, &st, b.data, &b.strides, b.itemsize as usize) };
        Ok(b)
    };
    let mut traced: Vec<Traced> = Vec::new();
    let groups = [(o_mm, &info.mit_mot_in, true), (o_ms, &info.mit_sot_in, false), (o_ss, &info.sit_sot_in, false)];
    for (base, taps, is_mm) in groups {
        for (j, in_taps) in taps.iter().enumerate() {
            let a = arg(base + j);
            if a.ndim < 1 {
                return Err("Scan: state buffers need at least one dimension".into());
            }
            let buf = copy_of(a, d.outer_in_dtypes[base + j], owned)?;
            let lookback = -in_taps.iter().copied().min().unwrap_or(0).min(0);
            let out_taps = if is_mm { info.mit_mot_out[j].clone() } else { vec![0] };
            traced.push(Traced { buf, lookback, in_taps: in_taps.clone(), out_taps, post_process: !is_mm });
        }
    }
    let mut untraced: Vec<Buf> =
        (0..info.n_untraced).map(|j| copy_of(arg(o_ut + j), d.outer_in_dtypes[o_ut + j], owned)).collect::<Result<_, _>>()?;
    let n_mm_outs: usize = info.mit_mot_out.iter().map(|t| t.len()).sum();
    let first_nit_out = n_mm_outs + n_ms + n_ss;
    let mut nit: Vec<Option<Buf>> = Vec::new();
    let mut nit_len: Vec<i64> = Vec::new();
    for j in 0..info.n_nit_sot {
        // SAFETY: nit-sot lengths are valid 0-d integer arrays.
        let len = unsafe { read_int(&arg(o_nit + j), d.outer_in_dtypes[o_nit + j]) }.max(0);
        nit_len.push(len);
        nit.push(match step.output_static_shape(first_nit_out + j) {
            Some(s) => {
                let mut shape = vec![len];
                shape.extend(s);
                Some(Buf::new(shape, d.inner_out_dtypes[first_nit_out + j].itemsize() as i64, owned)?)
            }
            None => None,
        });
    }

    // ---- step outputs with static shapes go into temporaries we provide
    let mut temps: Vec<Option<Buf>> = Vec::new();
    let mut temp_owned = Vec::new();
    for o in 0..step.n_outputs() {
        temps.push(match step.output_static_shape(o) {
            Some(s) => Some(Buf::new(s, d.inner_out_dtypes[o].itemsize() as i64, &mut temp_owned)?),
            None => None,
        });
    }
    let provided: Vec<Option<*mut u8>> = temps.iter().map(|t| t.as_ref().map(|b| b.data)).collect();

    let mut st = step.take_state();
    let mut step_in: Vec<PtArray> = Vec::new();
    let mut i: i64 = 0;
    let mut stop = false;
    let mut result: Result<(), String> = Ok(());
    while i < n_steps && !stop {
        step_in.clear();
        for k in 0..info.n_seqs {
            let a = arg(o_seq + k);
            // SAFETY: sequences are valid arrays.
            let (sh, sts) = unsafe { (a.shape(), a.strides()) };
            if sh.is_empty() || sh[0] <= i {
                result = Err(format!("Scan: sequence {k} is shorter than n_steps"));
                break;
            }
            // SAFETY: row i of a sequence (i < its length) and its trailing axes.
            step_in.push(unsafe {
                PtArray { data: a.data.offset((i * sts[0]) as isize), shape: a.shape.add(1), strides: a.strides.add(1), owner: std::ptr::null_mut(), ndim: a.ndim - 1 }
            });
        }
        if result.is_err() {
            break;
        }
        for t in traced.iter_mut() {
            let len = t.buf.len().max(1);
            for k in 0..t.in_taps.len() {
                let r = (i + t.lookback + t.in_taps[k]).rem_euclid(len);
                step_in.push(t.buf.view(Some(r)));
            }
        }
        for u in untraced.iter_mut() {
            step_in.push(u.view(None));
        }
        for k in 0..info.n_non_seqs {
            step_in.push(arg(o_ns + k));
        }

        // run the step
        let r = (|| -> Result<(), String> {
            for (k, a) in step_in.iter().enumerate() {
                // SAFETY: valid arrays (views into live buffers or outer inputs).
                let (sh, sts) = unsafe { (a.shape().to_vec(), a.strides().to_vec()) };
                step.set_input(&mut st, k, a.data, &sh, &sts)?;
            }
            step.execute(&mut st, ctx, &provided)
        })();
        if let Err(m) = r {
            result = Err(format!("in Scan step: {m}"));
            step.finish(&mut st);
            break;
        }
        let outs: Vec<PtArray> = (0..step.n_outputs()).map(|o| step.output_view(&st, o)).collect();

        // write the step outputs into their rows
        let mut q = 0;
        let mut r: Result<(), String> = Ok(());
        for t in traced.iter_mut() {
            let len = t.buf.len().max(1);
            for k in 0..t.out_taps.len() {
                let row = (i + t.lookback + t.out_taps[k]).rem_euclid(len);
                r = r.and(t.buf.write_row(row, &outs[q]));
                q += 1;
            }
        }
        for j in 0..info.n_nit_sot {
            if nit[j].is_none() {
                // SAFETY: a step output.
                let mut shape = vec![nit_len[j]];
                shape.extend(unsafe { outs[q].shape() });
                nit[j] = Some(Buf::new(shape, d.inner_out_dtypes[first_nit_out + j].itemsize() as i64, owned)?);
            }
            let b = nit[j].as_ref().unwrap();
            if b.len() > 0 {
                r = r.and(b.write_row(i.rem_euclid(b.len()), &outs[q]));
            }
            q += 1;
        }
        for u in untraced.iter_mut() {
            // an untraced state becomes the step output (its shape may change)
            // SAFETY: a step output.
            let (sh, sts) = unsafe { (outs[q].shape().to_vec(), outs[q].strides().to_vec()) };
            if sh != u.shape {
                *u = Buf::new(sh.clone(), u.itemsize, owned)?;
            }
            // SAFETY: a valid step output and a buffer of its shape.
            unsafe { copy_strided(outs[q].data, &sh, &sts, u.data, &u.strides, u.itemsize as usize) };
            q += 1;
        }
        if info.as_while {
            // SAFETY: the condition is a valid 0-d bool step output.
            stop = unsafe { *outs[q].data } != 0;
        }
        step.finish(&mut st);
        if let Err(m) = r {
            result = Err(m);
            break;
        }
        i += 1;
    }
    step.put_state(st);
    drop(temp_owned);
    result?;

    // ---- post-processing: rotate circular buffers, truncate (while) or zero the tail
    let finish = |b: &mut Buf, max_offset: i64, owned: &mut Vec<OwnedBuf>| -> Result<(), String> {
        let len = b.len();
        let used = i + max_offset;
        if 1 < len && len < used {
            let shift = used.rem_euclid(len);
            if shift > 0 {
                let nb = Buf::new(b.shape.clone(), b.itemsize, owned)?;
                let rb = b.row_bytes();
                // SAFETY: both buffers have `len` rows of `rb` bytes; they don't overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(b.row(shift), nb.data, ((len - shift) * rb) as usize);
                    std::ptr::copy_nonoverlapping(b.data, nb.row(len - shift), (shift * rb) as usize);
                }
                *b = nb;
            }
        } else if len > used {
            if info.as_while {
                b.shape[0] = used; // a shorter view of the same memory
            } else {
                // SAFETY: rows used..len are inside the buffer.
                unsafe { std::ptr::write_bytes(b.row(used), 0, ((len - used) * b.row_bytes()) as usize) };
            }
        }
        Ok(())
    };
    for t in traced.iter_mut() {
        if t.post_process {
            let max_offset = t.lookback + t.out_taps.iter().copied().max().unwrap_or(0);
            finish(&mut t.buf, max_offset, owned)?;
        }
    }
    for (j, b) in nit.iter_mut().enumerate() {
        if b.is_none() {
            // no step ran and the shape isn't static: an empty array
            let mut shape = vec![nit_len[j]];
            shape.extend(std::iter::repeat_n(0, step.output_ndim(first_nit_out + j)));
            *b = Some(Buf::new(shape, d.inner_out_dtypes[first_nit_out + j].itemsize() as i64, owned)?);
        }
        finish(b.as_mut().unwrap(), 0, owned)?;
    }

    // ---- hand the buffers out as the outer outputs: mit-mot, mit-sot, sit-sot, nit-sot, untraced
    // Buffers replaced during the loop are still in `owned`: hand out exactly the final
    // ones and keep the rest there to be released.
    let mut k = n_outer_in;
    let mut handed: Vec<*mut PtBuffer> = Vec::new();
    let mut emit = |b: &Buf| {
        // SAFETY: output k exists and has shape/strides storage of its ndim.
        let o = unsafe { &mut *args.add(k) };
        o.data = b.data;
        o.owner = b.owner;
        // SAFETY: as above.
        unsafe {
            o.shape_mut().copy_from_slice(&b.shape);
            o.strides_mut().copy_from_slice(&b.strides);
        }
        handed.push(b.owner);
        k += 1;
    };
    for t in &traced {
        emit(&t.buf);
    }
    for b in &nit {
        emit(b.as_ref().unwrap());
    }
    for u in &untraced {
        emit(u);
    }
    // release buffers that aren't outputs; the rest are handed over by the caller
    owned.retain(|b| handed.contains(&b.as_ptr()));
    Ok(())
}
