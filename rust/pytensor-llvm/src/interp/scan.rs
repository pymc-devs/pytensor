//! Scan as a kernel: a Rust loop around the loop body, which is a subprogram.
//!
//! Buffer conventions follow PyTensor's Scan (and its numba implementation): every
//! traced output (mit-mot, mit-sot, sit-sot) has a buffer whose first rows hold the
//! initial taps; the body reads tap `k` at row `(i + lookback + k) % len` and writes at
//! `(i + lookback + out_tap) % len`. After the loop, buffers shorter than the full trace
//! are rotated so the last value is at the end, while-loops are truncated to the iterations
//! run, and unused rows of regular loops are zeroed. nit-sot buffers are allocated once
//! the first iteration's output shape is known.
//!
//! Body outputs with static shapes are written into memory this kernel provides; the
//! others come back as buffers of the loop body. Either way they are copied into
//! their rows, so a circular buffer whose output row coincides with an input tap is
//! still correct.

use super::program::Program;
use crate::array::{OwnedArray, PtArrayMut, PtArrayRef};
use crate::dtype::DType;
use crate::kernel::{Args, Kernel};
use crate::runtime::PtCtx;
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

pub struct ScanConfig {
    pub info: ScanInfo,
    pub body: Arc<Program>,
    pub outer_in_dtypes: Vec<DType>,
    pub inner_out_dtypes: Vec<DType>,
    /// the Python loop body: keeps its compiled code and kernels alive
    pub keepalive: pyo3::Py<pyo3::PyAny>,
}

/// A buffer of the loop: a C-contiguous array we hold alone, until it is handed out
/// as an output. Its first axis counts rows.
struct Buf(OwnedArray);

impl Buf {
    fn zeros(dtype: DType, shape: Vec<i64>) -> Result<Buf, String> {
        if shape.is_empty() {
            return Err("Scan: state buffers need at least one dimension".into());
        }
        Ok(Buf(OwnedArray::zeros(dtype, shape)?))
    }

    fn copy_of(source: &PtArrayRef) -> Result<Buf, String> {
        if source.ndim() == 0 {
            return Err("Scan: state buffers need at least one dimension".into());
        }
        Ok(Buf(OwnedArray::copy_of(source)?))
    }

    fn len(&self) -> i64 {
        self.0.rows().unwrap_or(0)
    }

    fn row(&self, r: i64) -> Result<PtArrayRef<'_>, String> {
        self.0
            .as_ref()
            .row(r)
            .ok_or_else(|| format!("Scan: no row {r} in a buffer of {}", self.len()))
    }

    fn row_mut(&mut self, r: i64) -> Result<PtArrayMut<'_>, String> {
        let len = self.len();
        self.0
            .as_mut()?
            .row(r)
            .ok_or_else(|| format!("Scan: no row {r} in a buffer of {len}"))
    }

    /// Copy an array of this buffer's row shape into row `r`.
    fn write_row(&mut self, r: i64, source: &PtArrayRef) -> Result<(), String> {
        self.row_mut(r)?
            .copy_from(source)
            .map_err(|e| format!("Scan: body output: {e}"))
    }
}

/// A state of the loop that isn't traced: the whole array is the state.
struct Untraced(OwnedArray);

struct Traced {
    buf: Buf,
    lookback: i64,
    in_taps: Vec<i64>,
    out_taps: Vec<i64>,
    post_process: bool,
}

/// An output of the loop body after an iteration: in the memory we provided for it,
/// or an array of its own.
enum BodyOutput {
    Provided,
    Taken(OwnedArray),
}

/// Scan as a kernel: args = outer inputs ++ outer outputs (all allocated here).
impl Kernel for ScanConfig {
    fn arity(&self) -> (usize, usize) {
        let info = &self.info;
        let n_states = info.mit_mot_in.len() + info.mit_sot_in.len() + info.sit_sot_in.len();
        (
            1 + info.n_seqs + n_states + info.n_untraced + info.n_nit_sot + info.n_non_seqs,
            n_states + info.n_nit_sot + info.n_untraced,
        )
    }

    fn run(&self, ctx: &mut PtCtx, mut args: Args) -> Result<(), String> {
        let d = self;
        let info = &d.info;
        let inputs = &args.inputs;
        let (n_mm, n_ms, n_ss) = (
            info.mit_mot_in.len(),
            info.mit_sot_in.len(),
            info.sit_sot_in.len(),
        );
        let o_seq = 1;
        let o_mm = o_seq + info.n_seqs;
        let o_ms = o_mm + n_mm;
        let o_ss = o_ms + n_ms;
        let o_ut = o_ss + n_ss;
        let o_nit = o_ut + info.n_untraced;
        let o_ns = o_nit + info.n_nit_sot;
        let n_steps = inputs.get(0)?.item_as_i64()?.max(0);
        let body = &*d.body;

        // ---- buffers: copies of the outer buffers (the loop writes into them)
        let mut traced: Vec<Traced> = Vec::new();
        let groups = [
            (o_mm, &info.mit_mot_in, true),
            (o_ms, &info.mit_sot_in, false),
            (o_ss, &info.sit_sot_in, false),
        ];
        for (base, taps, is_mm) in groups {
            for (j, in_taps) in taps.iter().enumerate() {
                let buf = Buf::copy_of(&inputs.get(base + j)?)?;
                let lookback = -in_taps.iter().copied().min().unwrap_or(0).min(0);
                let out_taps = if is_mm {
                    info.mit_mot_out[j].clone()
                } else {
                    vec![0]
                };
                traced.push(Traced {
                    buf,
                    lookback,
                    in_taps: in_taps.clone(),
                    out_taps,
                    post_process: !is_mm,
                });
            }
        }
        let mut untraced: Vec<Untraced> = (0..info.n_untraced)
            .map(|j| Ok(Untraced(OwnedArray::copy_of(&inputs.get(o_ut + j)?)?)))
            .collect::<Result<_, String>>()?;
        let n_mm_outs: usize = info.mit_mot_out.iter().map(|t| t.len()).sum();
        let first_nit_out = n_mm_outs + n_ms + n_ss;
        let mut nit: Vec<Option<Buf>> = Vec::new();
        let mut nit_len: Vec<i64> = Vec::new();
        for j in 0..info.n_nit_sot {
            let len = inputs.get(o_nit + j)?.item_as_i64()?.max(0);
            nit_len.push(len);
            nit.push(match body.output_static_shape(first_nit_out + j) {
                Some(s) => {
                    let mut shape = vec![len];
                    shape.extend(s);
                    Some(Buf::zeros(d.inner_out_dtypes[first_nit_out + j], shape)?)
                }
                None => None,
            });
        }

        // ---- body outputs with static shapes go into temporaries we provide
        let mut temps: Vec<Option<OwnedArray>> = Vec::new();
        for o in 0..body.n_outputs() {
            temps.push(match body.output_static_shape(o) {
                Some(s) => Some(OwnedArray::zeros(d.inner_out_dtypes[o], s)?),
                None => None,
            });
        }

        let mut st = body.take_memory();
        let mut i: i64 = 0;
        let mut stop = false;
        let mut result: Result<(), String> = Ok(());
        while i < n_steps && !stop {
            // run the body. Its inputs are rows of our buffers, borrowed until we
            // have its outputs: in the temporaries, or as arrays of their own.
            let outs = (|| -> Result<Vec<BodyOutput>, String> {
                let mut call = body.begin(&mut st)?;
                // (the body's inputs, in order)
                let mut k = 0..;
                for s in 0..info.n_seqs {
                    let row = inputs
                        .get(o_seq + s)?
                        .row(i)
                        .ok_or_else(|| format!("Scan: sequence {s} is shorter than n_steps"))?;
                    call.set_input(k.next().unwrap(), row)?;
                }
                for t in traced.iter() {
                    let len = t.buf.len().max(1);
                    for tap in &t.in_taps {
                        let r = (i + t.lookback + tap).rem_euclid(len);
                        call.set_input(k.next().unwrap(), t.buf.row(r)?)?;
                    }
                }
                for u in untraced.iter() {
                    call.set_input(k.next().unwrap(), u.0.as_ref())?;
                }
                for n in 0..info.n_non_seqs {
                    call.set_input(k.next().unwrap(), inputs.get(o_ns + n)?)?;
                }
                let mut outs = Vec::with_capacity(temps.len());
                for (o, temp) in temps.iter_mut().enumerate() {
                    let direct = match temp {
                        Some(temp) => call.provide(o, temp.as_mut()?)?.is_none(),
                        None => false,
                    };
                    outs.push(direct);
                }
                let done = call.run(ctx)?;
                outs.into_iter()
                    .enumerate()
                    .map(|(o, direct)| {
                        Ok(match direct {
                            true => BodyOutput::Provided,
                            false => BodyOutput::Taken(done.take_output(o)?),
                        })
                    })
                    .collect()
            })();
            let outs = match outs {
                Ok(outs) => outs,
                Err(m) => {
                    result = Err(match m.starts_with("Scan: ") {
                        true => m,
                        false => format!("in a Scan iteration: {m}"),
                    });
                    break;
                }
            };
            let out = |q: usize| -> Result<PtArrayRef<'_>, String> {
                match (&outs[q], &temps[q]) {
                    (BodyOutput::Taken(a), _) => Ok(a.as_ref()),
                    (BodyOutput::Provided, Some(temp)) => Ok(temp.as_ref()),
                    (BodyOutput::Provided, None) => Err("Scan: a body output is missing".into()),
                }
            };

            // write the body outputs into their rows
            let mut q = 0;
            let mut r: Result<(), String> = Ok(());
            for t in traced.iter_mut() {
                let len = t.buf.len().max(1);
                for tap in &t.out_taps {
                    let row = (i + t.lookback + tap).rem_euclid(len);
                    r = r.and(t.buf.write_row(row, &out(q)?));
                    q += 1;
                }
            }
            for j in 0..info.n_nit_sot {
                let value = out(q)?;
                if nit[j].is_none() {
                    let mut shape = vec![nit_len[j]];
                    shape.extend(value.shape());
                    nit[j] = Some(Buf::zeros(d.inner_out_dtypes[first_nit_out + j], shape)?);
                }
                let b = nit[j].as_mut().unwrap();
                if b.len() > 0 {
                    r = r.and(b.write_row(i.rem_euclid(b.len()), &value));
                }
                q += 1;
            }
            for u in untraced.iter_mut() {
                // an untraced state becomes the body output (its shape may change)
                let value = out(q)?;
                if value.shape() != u.0.shape() {
                    u.0 = OwnedArray::copy_of(&value)?;
                } else {
                    u.0.as_mut()?.copy_from(&value)?;
                }
                q += 1;
            }
            if info.as_while {
                stop = out(q)?.item_as_i64()? != 0;
            }
            if let Err(m) = r {
                result = Err(m);
                break;
            }
            i += 1;
        }
        body.put_memory(st);
        drop(temps);
        result?;

        // ---- post-processing: rotate circular buffers, truncate (while) or zero the tail
        let finish = |b: &mut Buf, max_offset: i64| -> Result<(), String> {
            let len = b.len();
            let used = i + max_offset;
            if 1 < len && len < used {
                let shift = used.rem_euclid(len);
                if shift > 0 {
                    let mut rotated = Buf::zeros(b.0.dtype(), b.0.shape().to_vec())?;
                    for r in 0..len {
                        rotated.write_row(r, &b.row((r + shift) % len)?)?;
                    }
                    *b = rotated;
                }
            } else if len > used {
                if info.as_while {
                    b.0.truncate(used); // a shorter view of the same memory
                } else {
                    for r in used.max(0)..len {
                        b.row_mut(r)?.zero();
                    }
                }
            }
            Ok(())
        };
        for t in traced.iter_mut() {
            if t.post_process {
                let max_offset = t.lookback + t.out_taps.iter().copied().max().unwrap_or(0);
                finish(&mut t.buf, max_offset)?;
            }
        }
        for (j, b) in nit.iter_mut().enumerate() {
            if b.is_none() {
                // no iteration ran and the shape isn't static: an empty array
                let mut shape = vec![nit_len[j]];
                shape.extend(std::iter::repeat_n(0, body.output_ndim(first_nit_out + j)));
                *b = Some(Buf::zeros(d.inner_out_dtypes[first_nit_out + j], shape)?);
            }
            finish(b.as_mut().unwrap(), 0)?;
        }

        // ---- hand the buffers out as the outer outputs: mit-mot, mit-sot, sit-sot,
        // nit-sot, untraced. (Buffers replaced during the loop were dropped.)
        let arrays = traced
            .into_iter()
            .map(|t| t.buf.0)
            .chain(nit.into_iter().map(|b| b.unwrap().0))
            .chain(untraced.into_iter().map(|u| u.0));
        for (k, array) in arrays.enumerate() {
            args.outputs.slot(k)?.set(array)?;
        }
        Ok(())
    }
}
