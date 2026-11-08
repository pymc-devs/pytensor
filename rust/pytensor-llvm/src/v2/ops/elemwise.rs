//! Elemwise and CAReduce: loop nests around a scalar op.

use crate::dtype::DType;
use crate::v2::emit::{e, Emitter, V};
use crate::v2::index::{Idx, IndexVec};
use crate::v2::op::{Generate, Key, TensorOp, R};
use crate::v2::scalar::ScalarOp;
use crate::v2::types::{Axis, AxisLen, InputIdx, OutputAlloc, OutputIdx, Signature};
use crate::v2::unit::{ArrVal, UnitCx};
use inkwell::values::{IntValue, PointerValue};
use inkwell::IntPredicate;
use std::sync::Arc;

// ------------------------------------------------------------------ loop nests

/// One array visited by a loop nest: element pointer = data + sum(index * stride).
#[derive(Clone)]
pub struct Operand<'c> {
    pub dtype: DType,
    pub data: PointerValue<'c>,
    /// byte strides, one per loop axis (0 where the operand is broadcast)
    pub strides: Vec<IntValue<'c>>,
}

/// Is the operand C-contiguous over `shape`? A constant where we can tell statically.
fn contiguous<'c>(em: &Emitter<'c>, shape: &[IntValue<'c>], op: &Operand<'c>) -> R<IntValue<'c>> {
    let bt = em.ctx.bool_type();
    let mut ok = bt.const_int(1, false);
    let mut expect = em.i64c(op.dtype.itemsize() as i64);
    for a in (0..shape.len()).rev() {
        if Emitter::const_int_value(shape[a]) == Some(1) {
            continue;
        }
        let st = op.strides[a];
        let same = match (Emitter::const_int_value(st), Emitter::const_int_value(expect)) {
            (Some(x), Some(y)) => bt.const_int((x == y) as u64, false),
            _ => {
                let eq = em.b.build_int_compare(IntPredicate::EQ, st, expect, "ceq").map_err(e)?;
                // a length-1 axis may have any stride
                let one = em.b.build_int_compare(IntPredicate::EQ, shape[a], em.i64c(1), "len1").map_err(e)?;
                em.b.build_or(eq, one, "cok").map_err(e)?
            }
        };
        ok = em.b.build_and(ok, same, "contig").map_err(e)?;
        expect = em.mul(expect, shape[a])?;
    }
    Ok(ok)
}

/// for every index of `shape`: body(element pointers of `ops`).
///
/// With all operands C-contiguous this is one flat loop over element pointers (what the
/// vectorizer likes best); otherwise a strided nest. When contiguity is only known at
/// run time both versions are generated.
pub fn for_each<'c>(
    em: &Emitter<'c>,
    shape: &[IntValue<'c>],
    ops: &[Operand<'c>],
    body: &mut dyn FnMut(&[PointerValue<'c>]) -> R<()>,
) -> R<()> {
    let mut contig = em.ctx.bool_type().const_int(1, false);
    for op in ops {
        let c = contiguous(em, shape, op)?;
        contig = em.b.build_and(contig, c, "allc").map_err(e)?;
    }
    let flat = |body: &mut dyn FnMut(&[PointerValue<'c>]) -> R<()>| -> R<()> {
        let mut n = em.i64c(1);
        for &s in shape {
            n = em.mul(n, s)?;
        }
        em.emit_loop(n, &mut |i| {
            let ptrs: Vec<PointerValue> = ops.iter().map(|op| em.gep_elem(op.dtype, op.data, i)).collect::<R<_>>()?;
            body(&ptrs)
        })
    };
    match contig.get_zero_extended_constant() {
        Some(1) => flat(body),
        Some(_) => strided(em, shape, ops, body),
        None => {
            let f = em.cur_fn();
            let fast = em.ctx.append_basic_block(f, "contig");
            let slow = em.ctx.append_basic_block(f, "strided");
            let done = em.ctx.append_basic_block(f, "nest.done");
            em.b.build_conditional_branch(contig, fast, slow).map_err(e)?;
            em.b.position_at_end(fast);
            flat(body)?;
            em.b.build_unconditional_branch(done).map_err(e)?;
            em.b.position_at_end(slow);
            strided(em, shape, ops, body)?;
            em.b.build_unconditional_branch(done).map_err(e)?;
            em.b.position_at_end(done);
            Ok(())
        }
    }
}

fn strided<'c>(
    em: &Emitter<'c>,
    shape: &[IntValue<'c>],
    ops: &[Operand<'c>],
    body: &mut dyn FnMut(&[PointerValue<'c>]) -> R<()>,
) -> R<()> {
    fn level<'c>(
        em: &Emitter<'c>,
        a: usize,
        shape: &[IntValue<'c>],
        ops: &[Operand<'c>],
        base: &[PointerValue<'c>],
        body: &mut dyn FnMut(&[PointerValue<'c>]) -> R<()>,
    ) -> R<()> {
        if a == shape.len() {
            return body(base);
        }
        em.emit_loop(shape[a], &mut |i| {
            let ptrs: Vec<PointerValue> = ops
                .iter()
                .zip(base)
                .map(|(op, &p)| {
                    let off = em.mul(i, op.strides[a])?;
                    em.gep_bytes(p, off)
                })
                .collect::<R<_>>()?;
            level(em, a + 1, shape, ops, &ptrs, body)
        })
    }
    let base: Vec<PointerValue> = ops.iter().map(|o| o.data).collect();
    level(em, 0, shape, ops, &base, body)
}

/// Copy (and convert) `src` into `dst`, which have the same shape.
pub fn copy_array<'c>(cx: &mut UnitCx<'c>, src: &ArrVal<'c>, dst: &ArrVal<'c>) -> R<()> {
    let ops = [
        Operand { dtype: src.dtype, data: src.data, strides: src.strides.clone() },
        Operand { dtype: dst.dtype, data: dst.data, strides: dst.strides.clone() },
    ];
    let em = &cx.e;
    for_each(em, &dst.shape, &ops, &mut |p| {
        let v = em.load_elem(p[0], src.dtype)?;
        let v = em.cast(v, src.dtype, dst.dtype)?;
        em.store_elem(p[1], v, dst.dtype)
    })
}

/// The loop shape of an elemwise over `ins` with output type lengths `out`: checks that
/// non-broadcast lengths agree (PyTensor doesn't broadcast at run time).
fn broadcast_shape<'c>(cx: &mut UnitCx<'c>, sig: &Signature, ins: &IndexVec<InputIdx, ArrVal<'c>>, out: &[AxisLen]) -> R<Vec<IntValue<'c>>> {
    let nd = out.len();
    let mut shape = Vec::with_capacity(nd);
    for a in 0..nd {
        let ax = Axis::new(a);
        let mut len: Option<IntValue> = out[a].known().map(|n| cx.e.i64c(n as i64));
        for (i, x) in ins.iter_enumerated() {
            if sig.input(i).broadcastable(ax) {
                continue;
            }
            let l = x.shape[a];
            match len {
                None => len = Some(l),
                Some(prev) => {
                    if Emitter::const_int_value(prev).is_some() && Emitter::const_int_value(prev) == Emitter::const_int_value(l) {
                        continue;
                    }
                    let ok = cx.e.b.build_int_compare(IntPredicate::EQ, prev, l, "bc").map_err(e)?;
                    cx.check(ok, &format!("Elemwise: shape mismatch: input {} has a different length on axis {a}", i.0))?;
                }
            }
        }
        shape.push(len.unwrap_or_else(|| cx.e.i64c(1)));
    }
    Ok(shape)
}

// ------------------------------------------------------------------ Elemwise

/// PyTensor's Elemwise. Broadcasting follows the static types.
pub struct Elemwise {
    pub scalar: Arc<dyn ScalarOp>,
}

impl std::fmt::Debug for Elemwise {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Elemwise{{{:?}}}", self.scalar)
    }
}

impl TensorOp for Elemwise {
    fn check(&self, sig: &Signature) -> R<()> {
        let (lo, hi) = self.scalar.arity();
        if sig.inputs.len() < lo || sig.inputs.len() > hi {
            return Err(format!("{} inputs for a scalar op taking {lo}..={hi}", sig.inputs.len()));
        }
        if sig.outputs.len() != self.scalar.n_outputs() {
            return Err("output count differs from the scalar op's".into());
        }
        let nd = sig.output(OutputIdx(0)).ndim();
        if sig.inputs.iter().any(|i| i.ty.ndim() != nd) || sig.outputs.iter().any(|o| o.ty.ndim() != nd) {
            return Err("Elemwise operands must have the same ndim".into());
        }
        if sig.outputs.iter().any(|o| matches!(o.alloc, OutputAlloc::ViewOf(_))) {
            return Err("Elemwise outputs can't be views".into());
        }
        Ok(())
    }

    fn generate(&self) -> Option<&dyn Generate> {
        Some(self)
    }
}

impl Generate for Elemwise {
    fn key(&self, k: &mut Key) {
        k.tag::<Self>();
        self.scalar.key(k);
    }

    fn emit<'c>(
        &self,
        cx: &mut UnitCx<'c>,
        sig: &Signature,
        ins: &IndexVec<InputIdx, ArrVal<'c>>,
        provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>> {
        let out_ty = sig.output(OutputIdx(0)).clone();
        let shape = broadcast_shape(cx, sig, ins, &out_ty.shape)?;
        let mut outs = IndexVec::new();
        for (o, os) in sig.outputs.iter_enumerated() {
            outs.push(match &provided[o] {
                Some(a) => a.clone(),
                None => cx.alloc_output(o, os.ty.dtype, &shape)?,
            });
        }
        // inputs that are one element for the whole loop are read once, before it
        let em = &cx.e;
        let mut fixed: Vec<Option<V>> = Vec::new();
        let mut operands: Vec<Operand> = Vec::new();
        let mut slot: Vec<usize> = Vec::new();
        for (i, x) in ins.iter_enumerated() {
            let ty = sig.input(i);
            if let Some((f, iv)) = x.known {
                fixed.push(Some(em.constant(x.dtype, f, iv)));
                slot.push(usize::MAX);
            } else if ty.axes().all(|a| ty.broadcastable(a)) {
                fixed.push(Some(em.load_elem(x.data, x.dtype)?));
                slot.push(usize::MAX);
            } else {
                let strides = ty
                    .axes()
                    .map(|a| if ty.broadcastable(a) { em.i64c(0) } else { x.strides[a.index()] })
                    .collect();
                slot.push(operands.len());
                operands.push(Operand { dtype: x.dtype, data: x.data, strides });
                fixed.push(None);
            }
        }
        let n_in_ops = operands.len();
        for o in outs.iter() {
            operands.push(Operand { dtype: o.dtype, data: o.data, strides: o.strides.clone() });
        }
        let in_dtypes: Vec<DType> = ins.iter().map(|x| x.dtype).collect();
        let out_dtypes: Vec<DType> = outs.iter().map(|o| o.dtype).collect();
        let scalar = &self.scalar;
        for_each(em, &shape, &operands, &mut |p| {
            let mut args = Vec::with_capacity(in_dtypes.len());
            for (k, dt) in in_dtypes.iter().enumerate() {
                let v = match fixed[k] {
                    Some(v) => v,
                    None => em.load_elem(p[slot[k]], *dt)?,
                };
                args.push((v, *dt));
            }
            let res = scalar.emit(em, &args, &out_dtypes)?;
            for (k, v) in res.into_iter().enumerate() {
                em.store_elem(p[n_in_ops + k], v, out_dtypes[k])?;
            }
            Ok(())
        })?;
        Ok(outs)
    }
}

// ------------------------------------------------------------------ CAReduce

/// PyTensor's CAReduce: combine elements along `axes` with a binary scalar op,
/// starting from `identity`, accumulating in `acc`.
#[derive(Debug)]
pub struct CAReduce {
    pub scalar: Arc<dyn ScalarOp>,
    pub axes: Vec<Axis>,
    pub identity: (f64, i64),
    pub acc: DType,
}

impl TensorOp for CAReduce {
    fn check(&self, sig: &Signature) -> R<()> {
        if sig.inputs.len() != 1 || sig.outputs.len() != 1 {
            return Err("CAReduce has one input and one output".into());
        }
        let nd = sig.input(InputIdx(0)).ndim();
        if self.axes.iter().any(|a| a.index() >= nd) {
            return Err("reduction axis out of range".into());
        }
        if sig.output(OutputIdx(0)).ndim() + self.axes.len() != nd {
            return Err("output ndim doesn't match the reduced axes".into());
        }
        let (lo, hi) = self.scalar.arity();
        if !(lo..=hi).contains(&2) {
            return Err(format!("{:?} is not a binary scalar op", self.scalar));
        }
        Ok(())
    }

    fn generate(&self) -> Option<&dyn Generate> {
        Some(self)
    }
}

impl CAReduce {
    fn combine<'c>(&self, em: &Emitter<'c>, acc: V<'c>, x: V<'c>, xdt: DType) -> R<V<'c>> {
        let x = em.cast(x, xdt, self.acc)?;
        Ok(self.scalar.emit(em, &[(acc, self.acc), (x, self.acc)], &[self.acc])?.remove(0))
    }
}

impl Generate for CAReduce {
    fn key(&self, k: &mut Key) {
        k.tag::<Self>();
        self.scalar.key(k);
        k.debug(&self.axes);
        k.f64(self.identity.0);
        k.i64(self.identity.1);
        k.debug(&self.acc);
    }

    fn emit<'c>(
        &self,
        cx: &mut UnitCx<'c>,
        _sig: &Signature,
        ins: &IndexVec<InputIdx, ArrVal<'c>>,
        provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>> {
        let x = &ins[InputIdx(0)];
        let nd = x.ndim();
        let reduced: Vec<bool> = (0..nd).map(|a| self.axes.contains(&Axis::new(a))).collect();
        let out_shape: Vec<IntValue> = (0..nd).filter(|&a| !reduced[a]).map(|a| x.shape[a]).collect();
        let out = match &provided[OutputIdx(0)] {
            Some(a) => a.clone(),
            None => cx.alloc_output(OutputIdx(0), _sig.output(OutputIdx(0)).dtype, &out_shape)?,
        };
        let odt = out.dtype;

        if out_shape.is_empty() {
            // full reduction: accumulate in a register
            let em = &cx.e;
            let init = em.constant(self.acc, self.identity.0, self.identity.1);
            let acc = reduce_all(em, x, init, &mut |em, acc, v| self.combine(em, acc, v, x.dtype))?;
            let v = em.cast(acc, self.acc, odt)?;
            em.store_elem(out.data, v, odt)?;
            return Ok(IndexVec::from_vec(vec![out]));
        }

        // partial: accumulate into an array of the accumulator dtype
        let acc_arr = if self.acc == odt { out.clone() } else { cx.alloc_temp(self.acc, &out_shape)? };
        let em = &cx.e;
        let init = em.constant(self.acc, self.identity.0, self.identity.1);
        let acc_op = Operand { dtype: self.acc, data: acc_arr.data, strides: acc_arr.strides.clone() };
        for_each(em, &out_shape, &[acc_op], &mut |p| em.store_elem(p[0], init, self.acc))?;

        // loop over x; the accumulator's stride is 0 on reduced axes. If the innermost
        // axis is reduced, it's accumulated in a register first.
        let mut acc_strides = Vec::with_capacity(nd);
        let mut k = 0;
        for &r in &reduced {
            if r {
                acc_strides.push(em.i64c(0));
            } else {
                acc_strides.push(acc_arr.strides[k]);
                k += 1;
            }
        }
        let inner_reduced = reduced[nd - 1];
        let outer = if inner_reduced { nd - 1 } else { nd };
        let ops = [
            Operand { dtype: x.dtype, data: x.data, strides: x.strides[..outer].to_vec() },
            Operand { dtype: self.acc, data: acc_arr.data, strides: acc_strides[..outer].to_vec() },
        ];
        strided(em, &x.shape[..outer], &ops, &mut |p| {
            let cur = em.load_elem(p[1], self.acc)?;
            let new = if inner_reduced {
                let st = x.strides[nd - 1];
                em.emit_acc_loop(x.shape[nd - 1], &[cur], &mut |i, acc| {
                    let ptr = em.gep_bytes(p[0], em.mul(i, st)?)?;
                    let v = em.load_elem(ptr, x.dtype)?;
                    Ok(vec![self.combine(em, acc[0], v, x.dtype)?])
                })?[0]
            } else {
                let v = em.load_elem(p[0], x.dtype)?;
                self.combine(em, cur, v, x.dtype)?
            };
            em.store_elem(p[1], new, self.acc)
        })?;
        if self.acc != odt {
            copy_array(cx, &acc_arr, &out)?;
        }
        Ok(IndexVec::from_vec(vec![out]))
    }
}

/// Combine all elements of `x` into one accumulator.
fn reduce_all<'c>(
    em: &Emitter<'c>,
    x: &ArrVal<'c>,
    init: V<'c>,
    f: &mut dyn FnMut(&Emitter<'c>, V<'c>, V<'c>) -> R<V<'c>>,
) -> R<V<'c>> {
    let op = Operand { dtype: x.dtype, data: x.data, strides: x.strides.clone() };
    let contig = contiguous(em, &x.shape, &op)?;
    let flat = |f: &mut dyn FnMut(&Emitter<'c>, V<'c>, V<'c>) -> R<V<'c>>| -> R<V<'c>> {
        let mut n = em.i64c(1);
        for &s in &x.shape {
            n = em.mul(n, s)?;
        }
        Ok(em.emit_acc_loop(n, &[init], &mut |i, acc| {
            let v = em.load_elem(em.gep_elem(x.dtype, x.data, i)?, x.dtype)?;
            Ok(vec![f(em, acc[0], v)?])
        })?[0])
    };
    let nest = |f: &mut dyn FnMut(&Emitter<'c>, V<'c>, V<'c>) -> R<V<'c>>| -> R<V<'c>> {
        fn level<'c>(
            em: &Emitter<'c>,
            a: usize,
            x: &ArrVal<'c>,
            p: PointerValue<'c>,
            acc: V<'c>,
            f: &mut dyn FnMut(&Emitter<'c>, V<'c>, V<'c>) -> R<V<'c>>,
        ) -> R<V<'c>> {
            if a == x.ndim() {
                let v = em.load_elem(p, x.dtype)?;
                return f(em, acc, v);
            }
            Ok(em.emit_acc_loop(x.shape[a], &[acc], &mut |i, acc| {
                let q = em.gep_bytes(p, em.mul(i, x.strides[a])?)?;
                Ok(vec![level(em, a + 1, x, q, acc[0], f)?])
            })?[0])
        }
        level(em, 0, x, x.data, init, f)
    };
    match contig.get_zero_extended_constant() {
        Some(1) => flat(f),
        Some(_) => nest(f),
        None => {
            let func = em.cur_fn();
            let fast = em.ctx.append_basic_block(func, "rcontig");
            let slow = em.ctx.append_basic_block(func, "rstrided");
            let done = em.ctx.append_basic_block(func, "rdone");
            em.b.build_conditional_branch(contig, fast, slow).map_err(e)?;
            em.b.position_at_end(fast);
            let a = flat(f)?;
            let fast_end = em.b.get_insert_block().unwrap();
            em.b.build_unconditional_branch(done).map_err(e)?;
            em.b.position_at_end(slow);
            let b = nest(f)?;
            let slow_end = em.b.get_insert_block().unwrap();
            em.b.build_unconditional_branch(done).map_err(e)?;
            em.b.position_at_end(done);
            let phi = em.b.build_phi(init.get_type(), "racc").map_err(e)?;
            phi.add_incoming(&[(&a, fast_end), (&b, slow_end)]);
            Ok(phi.as_basic_value())
        }
    }
}
