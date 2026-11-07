//! Native Scan: a Rust driver loop around a compiled step function.
//!
//! The step function (Scan's inner graph) runs once per step through one of two
//! executors:
//!
//! * fast path: the whole step graph compiled as a single segment (all nodes small with
//!   static shapes), called directly through the kernel ABI. One indirect call per step.
//! * general path: the step graph compiled as a full inner `Program` (any ops, including
//!   numba fallbacks), executed with a fresh call state per Scan call.
//!
//! Buffer conventions follow PyTensor's Scan (and its numba implementation): every traced
//! output (mit-mot, mit-sot, sit-sot) has a buffer whose first rows hold the initial
//! taps; the step reads tap `k` at row `(i + lookback + k) % len` and writes at
//! `(i + lookback + out_tap) % len`. After the loop, buffers shorter than the full trace
//! are rotated so the last value is at the end, while-loops are truncated to the steps
//! taken, and unused rows of regular loops are zeroed. nit-sot buffers are allocated once
//! the first step's output shape is known.
//!
//! Step outputs are written into per-step temporaries and copied into their rows, so a
//! circular buffer whose output row coincides with an input tap is still correct.

use crate::abi::{PtArray, PtBuffer, PtCtx, KernelFn, MAX_NDIM};
use crate::dtype::DType;
use crate::ir::ScanInfo;
use crate::runtime::{CallState, ConstData, Program};
use std::ffi::c_void;
use std::sync::Arc;

pub enum StepExec {
    Kernel { func: KernelFn, consts: Vec<ConstData> },
    Program(Arc<Program>),
}

/// Type of an inner output: dtype and static shape (None where unknown).
#[derive(Clone)]
pub struct OutType {
    pub dtype: DType,
    pub shape: Vec<Option<i64>>,
}

pub struct ScanData {
    pub info: ScanInfo,
    pub step: StepExec,
    pub inner_outs: Vec<OutType>,
    pub outer_in_dtypes: Vec<DType>,
    pub desc: String,
}

unsafe impl Send for ScanData {}
unsafe impl Sync for ScanData {}

/// A C-contiguous buffer owned by the driver (until handed out as an output).
struct Buf {
    owner: *mut PtBuffer,
    data: *mut u8,
    shape: Vec<i64>,
    strides: Vec<i64>,
    itemsize: i64,
}

impl Buf {
    fn new(shape: Vec<i64>, itemsize: i64) -> Buf {
        let n: i64 = shape.iter().product();
        let owner = PtBuffer::new(n as usize * itemsize as usize, 64);
        let mut strides = vec![0; shape.len()];
        let mut acc = itemsize;
        for d in (0..shape.len()).rev() {
            strides[d] = acc;
            acc *= shape[d].max(1);
        }
        Buf { owner, data: unsafe { (*owner).data }, shape, strides, itemsize }
    }
    fn len(&self) -> i64 {
        self.shape[0]
    }
    fn row_bytes(&self) -> i64 {
        self.shape[1..].iter().product::<i64>() * self.itemsize
    }
    fn row(&self, r: i64) -> *mut u8 {
        unsafe { self.data.offset((r * self.row_bytes()) as isize) }
    }
    /// PtArray view of the whole buffer or of row `r`
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
}

/// Strided copy of `n`-dim data into a C-contiguous destination with the same shape.
unsafe fn copy_into(src: *const u8, shape: &[i64], src_strides: &[i64], dst: *mut u8, itemsize: i64) {
    let nd = shape.len();
    if shape.iter().any(|&s| s == 0) {
        return;
    }
    // fast path: source already C-contiguous
    let mut expect = itemsize;
    let mut contig = true;
    for d in (0..nd).rev() {
        if shape[d] != 1 && src_strides[d] != expect {
            contig = false;
        }
        expect *= shape[d];
    }
    if contig {
        std::ptr::copy(src, dst, expect as usize);
        return;
    }
    let mut idx = [0i64; MAX_NDIM];
    let (mut so, mut doff) = (0isize, 0isize);
    loop {
        std::ptr::copy_nonoverlapping(src.offset(so), dst.offset(doff), itemsize as usize);
        doff += itemsize as isize;
        let mut d = nd;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            so += src_strides[d] as isize;
            if idx[d] < shape[d] {
                break;
            }
            so -= (src_strides[d] * shape[d]) as isize;
            idx[d] = 0;
        }
    }
}

unsafe fn read_int(a: &PtArray, dt: DType) -> i64 {
    match dt {
        DType::Int64 => *(a.data as *const i64),
        DType::Int32 => *(a.data as *const i32) as i64,
        DType::Int16 => *(a.data as *const i16) as i64,
        DType::Int8 => *(a.data as *const i8) as i64,
        DType::Uint8 | DType::Bool => *a.data as i64,
        DType::Float64 => *(a.data as *const f64) as i64,
        DType::Float32 => *(a.data as *const f32) as i64,
    }
}

/// A traced buffer (mit-mot, mit-sot or sit-sot) during the loop.
struct Traced {
    buf: Buf,
    lookback: i64,
    in_taps: Vec<i64>,
    out_taps: Vec<i64>,
    post_process: bool,
}

/// args = outer inputs ++ outer outputs (all kernel-allocated). params -> &ScanData.
pub unsafe extern "C" fn ptl_scan(ctx: *mut PtCtx, args: *mut PtArray, params: *const c_void) -> i32 {
    let data = &*(*(params as *const usize) as *const ScanData);
    let mut owned: Vec<*mut PtBuffer> = Vec::new();
    let res = run(ctx, args, data, &mut owned);
    match res {
        Ok(()) => 0,
        Err(m) => {
            for b in owned {
                PtBuffer::decref(b);
            }
            PtCtx::fail(ctx, 2, &m)
        }
    }
}

unsafe fn run(ctx: *mut PtCtx, args: *mut PtArray, d: &ScanData, owned: &mut Vec<*mut PtBuffer>) -> Result<(), String> {
    let info = &d.info;
    let arg = |i: usize| *args.add(i);
    let (n_mm, n_ms, n_ss) = (info.mit_mot_in.len(), info.mit_sot_in.len(), info.sit_sot_in.len());
    let o_seq = 1;
    let o_mm = o_seq + info.n_seqs;
    let o_ms = o_mm + n_mm;
    let o_ss = o_ms + n_ms;
    let o_ut = o_ss + n_ss;
    let o_nit = o_ut + info.n_untraced;
    let o_ns = o_nit + info.n_nit_sot;
    let n_outer_in = o_ns + info.n_non_seqs;
    let n_steps = read_int(&arg(0), d.outer_in_dtypes[0]).max(0);

    // ---- buffers: copies of the outer buffers (the loop writes into them)
    let copy_of = |a: PtArray, dt: DType, owned: &mut Vec<*mut PtBuffer>| -> Buf {
        let b = Buf::new(a.shape().to_vec(), dt.itemsize() as i64);
        owned.push(b.owner);
        copy_into(a.data, a.shape(), a.strides(), b.data, b.itemsize);
        b
    };
    let mut traced: Vec<Traced> = Vec::new();
    let groups = [(o_mm, &info.mit_mot_in, true), (o_ms, &info.mit_sot_in, false), (o_ss, &info.sit_sot_in, false)];
    for (base, taps, is_mm) in groups {
        for (j, in_taps) in taps.iter().enumerate() {
            let a = arg(base + j);
            if a.ndim < 1 {
                return Err("Scan: state buffers need at least one dimension".into());
            }
            let buf = copy_of(a, d.outer_in_dtypes[base + j], owned);
            let lookback = -in_taps.iter().copied().min().unwrap_or(0).min(0);
            let out_taps = if is_mm { info.mit_mot_out[j].clone() } else { vec![0] };
            traced.push(Traced { buf, lookback, in_taps: in_taps.clone(), out_taps, post_process: !is_mm });
        }
    }
    let mut untraced: Vec<Buf> = (0..info.n_untraced).map(|j| copy_of(arg(o_ut + j), d.outer_in_dtypes[o_ut + j], owned)).collect();
    // nit-sot: length from the outer input, rows allocated now if the shape is static
    let n_mm_outs: usize = info.mit_mot_out.iter().map(|t| t.len()).sum();
    let first_nit_out = n_mm_outs + n_ms + n_ss;
    let mut nit: Vec<Option<Buf>> = Vec::new();
    let mut nit_len: Vec<i64> = Vec::new();
    for j in 0..info.n_nit_sot {
        let len = read_int(&arg(o_nit + j), d.outer_in_dtypes[o_nit + j]).max(0);
        nit_len.push(len);
        let t = &d.inner_outs[first_nit_out + j];
        nit.push(match t.shape.iter().copied().collect::<Option<Vec<i64>>>() {
            Some(s) => {
                let mut shape = vec![len];
                shape.extend(s);
                let b = Buf::new(shape, t.dtype.itemsize() as i64);
                owned.push(b.owner);
                Some(b)
            }
            None => None,
        });
    }

    // ---- step executor
    let mut temps: Vec<Buf> = Vec::new(); // fast path: step outputs (static shapes)
    let mut inner_state: Option<CallState> = None;
    match &d.step {
        StepExec::Kernel { .. } => {
            for t in &d.inner_outs {
                let shape: Vec<i64> = t.shape.iter().map(|s| s.expect("fast-path step outputs have static shapes")).collect();
                let b = Buf::new(shape, t.dtype.itemsize() as i64);
                temps.push(b);
            }
        }
        StepExec::Program(p) => inner_state = Some(p.new_state()),
    }
    let mut step_in: Vec<PtArray> = Vec::new();
    let mut step_args: Vec<PtArray> = Vec::new();
    // where each inner output's values are this step: (data, shape, strides) as raw
    // views, so the loop does no allocation
    let mut outs: Vec<PtArray> = Vec::with_capacity(d.inner_outs.len());
    let mut i: i64 = 0;
    let mut stop = false;
    let mut err: Option<String> = None;
    while i < n_steps && !stop {
        // inner inputs: sequences, taps, untraced states, non-sequences
        step_in.clear();
        for k in 0..info.n_seqs {
            let a = arg(o_seq + k);
            let (sh, st) = (a.shape(), a.strides());
            if sh[0] <= i {
                err = Some(format!("Scan: sequence {k} is shorter than n_steps"));
                break;
            }
            step_in.push(PtArray { data: a.data.offset((i * st[0]) as isize), shape: a.shape.add(1), strides: a.strides.add(1), owner: std::ptr::null_mut(), ndim: a.ndim - 1 });
        }
        if err.is_some() {
            break;
        }
        for t in traced.iter_mut() {
            let len = t.buf.len();
            for k in 0..t.in_taps.len() {
                let r = (i + t.lookback + t.in_taps[k]).rem_euclid(len.max(1));
                step_in.push(t.buf.view(Some(r)));
            }
        }
        for u in untraced.iter_mut() {
            step_in.push(u.view(None));
        }
        for k in 0..info.n_non_seqs {
            step_in.push(arg(o_ns + k));
        }

        // run the step; `outs` describes where each inner output's values are
        outs.clear();
        match &d.step {
            StepExec::Kernel { func, consts } => {
                step_args.clear();
                step_args.extend_from_slice(&step_in);
                for c in consts {
                    step_args.push(PtArray { data: (*c.buf).data, shape: c.shape.as_ptr() as *mut i64, strides: c.strides.as_ptr() as *mut i64, owner: std::ptr::null_mut(), ndim: c.shape.len() as i64 });
                }
                for t in temps.iter_mut() {
                    step_args.push(t.view(None));
                }
                let rc = func(ctx, step_args.as_mut_ptr(), std::ptr::null());
                (*ctx).end_kernel();
                if rc != 0 {
                    err = Some((*ctx).error.take().map(|(_, m)| m).unwrap_or_else(|| format!("Scan step failed (code {rc})")));
                    break;
                }
                for t in temps.iter_mut() {
                    outs.push(t.view(None));
                }
            }
            StepExec::Program(p) => {
                let st = inner_state.as_mut().unwrap();
                let mut bind_err = None;
                for (k, a) in step_in.iter().enumerate() {
                    if let Err(m) = p.set_input_raw(st, k, a.data, a.shape(), a.strides()) {
                        bind_err = Some(m);
                        break;
                    }
                }
                if let Some(m) = bind_err.or_else(|| p.execute(st, None).err()) {
                    err = Some(format!("in Scan step: {m}"));
                    break;
                }
                for o in st.outs.iter() {
                    outs.push(PtArray {
                        data: o.data,
                        shape: o.shape().as_ptr() as *mut i64,
                        strides: o.strides().as_ptr() as *mut i64,
                        owner: std::ptr::null_mut(),
                        ndim: o.ndim as i64,
                    });
                }
            }
        }

        // write the step outputs into their rows
        let mut q = 0;
        let write = |dst: *mut u8, dst_shape: &[i64], src: &PtArray, itemsize: i64| -> Result<(), String> {
            let (sh, st) = (src.shape(), src.strides());
            if sh.len() != dst_shape.len() || (0..sh.len()).any(|k| sh[k] != dst_shape[k]) {
                return Err(format!("Scan: step output has shape {sh:?}, expected {dst_shape:?}"));
            }
            copy_into(src.data, sh, st, dst, itemsize);
            Ok(())
        };
        for t in traced.iter_mut() {
            let len = t.buf.len().max(1);
            for k in 0..t.out_taps.len() {
                let r = (i + t.lookback + t.out_taps[k]).rem_euclid(len);
                if let Err(m) = write(t.buf.row(r), &t.buf.shape[1..], &outs[q], t.buf.itemsize) {
                    err = Some(m);
                }
                q += 1;
            }
        }
        for j in 0..info.n_nit_sot {
            let t = &d.inner_outs[first_nit_out + j];
            if nit[j].is_none() {
                let mut shape = vec![nit_len[j]];
                shape.extend(outs[q].shape());
                let b = Buf::new(shape, t.dtype.itemsize() as i64);
                owned.push(b.owner);
                nit[j] = Some(b);
            }
            let b = nit[j].as_mut().unwrap();
            if b.len() > 0 {
                let r = i.rem_euclid(b.len());
                if let Err(m) = write(b.row(r), &b.shape[1..], &outs[q], b.itemsize) {
                    err = Some(m);
                }
            }
            q += 1;
        }
        for u in untraced.iter_mut() {
            // untraced state: becomes the step output (its shape may change)
            if outs[q].shape() != &u.shape[..] {
                let nb = Buf::new(outs[q].shape().to_vec(), u.itemsize);
                owned.retain(|&x| x != u.owner);
                PtBuffer::decref(u.owner);
                owned.push(nb.owner);
                *u = nb;
            }
            copy_into(outs[q].data, outs[q].shape(), outs[q].strides(), u.data, u.itemsize);
            q += 1;
        }
        if info.as_while {
            stop = *outs[q].data != 0;
        }
        if let StepExec::Program(_) = &d.step {
            // the inner program handed us its outputs; release them
            for o in &inner_state.as_ref().unwrap().outs {
                if !o.owner.is_null() {
                    PtBuffer::decref(o.owner);
                }
            }
        }
        if err.is_some() {
            break;
        }
        i += 1;
    }
    for t in temps {
        PtBuffer::decref(t.owner);
    }
    if let Some(m) = err {
        return Err(m);
    }

    // ---- post-processing: rotate circular buffers, truncate (while) or zero the tail
    let finish = |b: &mut Buf, max_offset: i64, owned: &mut Vec<*mut PtBuffer>| {
        let len = b.len();
        let used = i + max_offset;
        if 1 < len && len < used {
            let shift = used.rem_euclid(len);
            if shift > 0 {
                let nb = Buf::new(b.shape.clone(), b.itemsize);
                let rb = b.row_bytes();
                std::ptr::copy_nonoverlapping(b.row(shift), nb.data, ((len - shift) * rb) as usize);
                std::ptr::copy_nonoverlapping(b.data, nb.row(len - shift), (shift * rb) as usize);
                owned.retain(|&x| x != b.owner);
                PtBuffer::decref(b.owner);
                owned.push(nb.owner);
                *b = nb;
            }
        } else if len > used {
            if info.as_while {
                b.shape[0] = used; // a shorter view of the same memory
            } else {
                std::ptr::write_bytes(b.row(used), 0, ((len - used) * b.row_bytes()) as usize);
            }
        }
    };
    for t in traced.iter_mut() {
        if t.post_process {
            let max_offset = t.lookback + t.out_taps.iter().copied().max().unwrap_or(0);
            finish(&mut t.buf, max_offset, owned);
        }
    }
    for (j, b) in nit.iter_mut().enumerate() {
        if b.is_none() {
            // no step ran and the shape wasn't static: an empty array
            let t = &d.inner_outs[first_nit_out + j];
            let mut shape = vec![nit_len[j]];
            shape.extend(t.shape.iter().map(|s| s.unwrap_or(0)));
            let nb = Buf::new(shape, t.dtype.itemsize() as i64);
            owned.push(nb.owner);
            *b = Some(nb);
        }
        finish(b.as_mut().unwrap(), 0, owned);
    }

    // ---- hand the buffers out as the outer outputs: mit-mot, mit-sot, sit-sot, nit-sot, untraced
    let mut k = n_outer_in;
    let mut emit = |b: &Buf| {
        let o = &mut *args.add(k);
        o.data = b.data;
        o.owner = b.owner;
        o.shape_mut().copy_from_slice(&b.shape);
        o.strides_mut().copy_from_slice(&b.strides);
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
    owned.clear(); // ownership moved to the outputs
    Ok(())
}
