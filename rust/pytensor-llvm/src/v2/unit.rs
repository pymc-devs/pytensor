//! Compiled units: a graph generated as one LLVM function.
//!
//! Inside a unit, arrays use the internal convention (`ArrVal`): a data pointer plus one
//! LLVM value per axis length and stride, which are constants where the type knows them.
//! Nodes whose op can be generated are emitted inline; the others are called through
//! the kernel trampoline (`ptl_v2_call_kernel`) with a table of callees in the unit's
//! environment.
//!
//! The unit itself has the external ABI:
//!
//! ```c
//! int32_t unit(PtCtx *ctx, PtArray *args /* inputs ++ outputs */, const UnitEnv *env);
//! ```
//!
//! Temporaries with a static shape live on the stack (small) or in runtime buffers that
//! are released at the single exit block, also on errors.

use super::emit::{e, Emitter};
use super::graph::{ConstData, ConstIdx, Graph, Origin, ValueId};
use super::index::{Idx, IndexVec};
use super::op::{Kernel, Key, R};
use super::types::{InputIdx, Layout, OutputAlloc, OutputIdx, Signature, TensorTy};
use crate::abi::{PtArray, PtCtx, PTARRAY_SIZE};
use crate::define_index;
use crate::dtype::DType;
use inkwell::basic_block::BasicBlock;
use inkwell::types::BasicType;
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::IntPredicate;
use std::sync::Arc;

define_index!(
    /// An entry of a unit's callee table.
    CalleeIdx
);
define_index!(
    /// An entry of a unit's constant table.
    EnvConstIdx
);

/// Stack temporaries up to this size; larger ones come from the runtime.
const MAX_STACK_BYTES: u64 = 16 * 1024;

// ------------------------------------------------------------------ environment

pub struct Callee {
    pub kernel: Box<dyn Kernel>,
    pub sig: Signature,
}

/// What a compiled unit needs at run time besides its code.
#[derive(Default)]
pub struct Env {
    pub callees: IndexVec<CalleeIdx, Callee>,
    pub consts: IndexVec<EnvConstIdx, Arc<ConstData>>,
}

/// The C view of `Env` passed to the unit.
#[repr(C)]
pub struct EnvRepr {
    callees: *const Callee,
    consts: *const *const u8,
}

/// Owns an `Env` and its C view.
pub struct UnitEnv {
    env: Env,
    const_ptrs: Vec<*const u8>,
}

// SAFETY: the raw pointers point into `env` (owned, never mutated after construction);
// kernels are Send + Sync and constants are immutable.
unsafe impl Send for UnitEnv {}
// SAFETY: as above.
unsafe impl Sync for UnitEnv {}

impl UnitEnv {
    pub fn new(env: Env) -> Self {
        let const_ptrs = env.consts.iter().map(|c| c.data()).collect();
        UnitEnv { env, const_ptrs }
    }

    pub fn repr(&self) -> EnvRepr {
        EnvRepr { callees: self.env.callees.as_slice().as_ptr(), consts: self.const_ptrs.as_ptr() }
    }
}

/// Collects the environment in the same order the code generator consumes it.
#[derive(Default)]
pub struct EnvBuilder {
    pub env: Env,
}

impl EnvBuilder {
    /// Walk a graph that is generated inline (the unit itself, or a nested compiled
    /// graph). Must visit things in the order `UnitCx::emit_graph` does.
    pub fn collect_graph(&mut self, g: &Graph) -> R<()> {
        for c in g.consts.iter() {
            if !c.shape.is_empty() {
                self.env.consts.push(c.clone());
            }
        }
        for (n, node) in g.nodes.iter_enumerated() {
            let sig = g.signature(n);
            match node.op.generate() {
                Some(generator) => generator.collect_env(self, &sig)?,
                None => {
                    let kernel = node
                        .op
                        .kernel(&sig)?
                        .ok_or_else(|| format!("{:?} can neither be generated nor called", node.op))?;
                    self.env.callees.push(Callee { kernel, sig });
                }
            }
        }
        Ok(())
    }
}

/// Called from generated code for nodes that run a precompiled kernel.
///
/// # Safety
/// `ctx` is the live call context, `env` the environment the unit was built with and
/// `args` holds the callee's arguments as described by its signature (the generated
/// code builds them from values of the right types).
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn ptl_v2_call_kernel(ctx: *mut PtCtx, env: *const EnvRepr, idx: u64, args: *mut PtArray) -> i32 {
    // SAFETY: the function's contract.
    unsafe {
        let callee = &*(*env).callees.add(idx as usize);
        let n = callee.sig.inputs.len() + callee.sig.outputs.len();
        let arrays = std::slice::from_raw_parts_mut(args, n);
        let mut ka = super::op::KernelArgs::new(&callee.sig, arrays);
        match callee.kernel.call(&mut *ctx, &mut ka) {
            Ok(()) => 0,
            Err(msg) => PtCtx::fail(ctx, 1, &msg),
        }
    }
}

/// Allocate `bytes` from the runtime; stores the buffer in `*owner`. NULL on failure
/// (with the error set).
///
/// # Safety
/// `ctx` is the live call context and `owner` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn ptl_v2_alloc(ctx: *mut PtCtx, bytes: u64, owner: *mut *mut crate::abi::PtBuffer) -> *mut u8 {
    // SAFETY: the function's contract.
    unsafe {
        let (data, buf) = PtCtx::alloc(ctx, bytes as usize);
        *owner = buf;
        if data.is_null() {
            PtCtx::fail(ctx, 1, &format!("could not allocate {bytes} bytes"));
        }
        data
    }
}

/// Release a buffer reference (NULL is ignored).
///
/// # Safety
/// `b` is NULL or a buffer reference the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn ptl_v2_decref(b: *mut crate::abi::PtBuffer) {
    if !b.is_null() {
        // SAFETY: the function's contract.
        unsafe { crate::abi::PtBuffer::decref(b) }
    }
}

/// Take an additional reference (NULL is ignored).
///
/// # Safety
/// `b` is NULL or a live buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn ptl_v2_incref(b: *mut crate::abi::PtBuffer) {
    if !b.is_null() {
        // SAFETY: the function's contract.
        unsafe { crate::abi::PtBuffer::incref(b) }
    }
}

/// Report an error with a static message.
///
/// # Safety
/// `ctx` is the live call context; `msg` a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn ptl_v2_fail(ctx: *mut PtCtx, msg: *const std::ffi::c_char) {
    // SAFETY: the function's contract.
    unsafe {
        let m = std::ffi::CStr::from_ptr(msg).to_string_lossy();
        PtCtx::fail(ctx, 1, &m);
    }
}

/// Runtime entry points generated code links against.
pub fn runtime_symbols() -> Vec<(&'static str, usize)> {
    vec![
        ("ptl_v2_call_kernel", ptl_v2_call_kernel as *const () as usize),
        ("ptl_v2_alloc", ptl_v2_alloc as *const () as usize),
        ("ptl_v2_decref", ptl_v2_decref as *const () as usize),
        ("ptl_v2_incref", ptl_v2_incref as *const () as usize),
        ("ptl_v2_fail", ptl_v2_fail as *const () as usize),
    ]
}

// ------------------------------------------------------------------ arrays in units

/// An array inside generated code. Byte strides.
#[derive(Clone, Debug)]
pub struct ArrVal<'c> {
    pub dtype: DType,
    pub data: PointerValue<'c>,
    pub shape: Vec<IntValue<'c>>,
    pub strides: Vec<IntValue<'c>>,
    /// Known at compile time (from the type, or because we allocated it).
    pub layout: Option<Layout>,
    /// The element, for 0-d constants.
    pub known: Option<(f64, i64)>,
}

impl<'c> ArrVal<'c> {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// The shape as constants, if it is static.
    pub fn static_shape(&self) -> Option<Vec<i64>> {
        self.shape.iter().map(|v| Emitter::const_int_value(*v)).collect()
    }
}

/// Where a node output's memory goes.
#[derive(Clone, Copy, Debug)]
enum OutTarget {
    Temp,
    /// a unit output with Callee allocation: allocate into this argument
    UnitOutput(OutputIdx),
}

pub struct UnitCx<'c> {
    pub e: Emitter<'c>,
    func: FunctionValue<'c>,
    ctx_ptr: PointerValue<'c>,
    env_ptr: PointerValue<'c>,
    args_ptr: PointerValue<'c>,
    sig: Signature,
    alloca_block: BasicBlock<'c>,
    exit: BasicBlock<'c>,
    /// return code slot
    rc: PointerValue<'c>,
    /// buffers released at exit
    cleanups: Vec<PointerValue<'c>>,
    /// unit outputs we allocated: released at exit on error only
    owned_outputs: Vec<OutputIdx>,
    /// outputs of the node being emitted
    cur_targets: Vec<OutTarget>,
    next_callee: usize,
    next_const: usize,
    n_callees: usize,
    n_consts: usize,
}

impl<'c> UnitCx<'c> {
    // ------------------------------------------------------------ helpers for ops

    /// An alloca in the entry block.
    pub fn alloca(&self, ty: impl BasicType<'c>, n: u32, name: &str) -> R<PointerValue<'c>> {
        let b = self.e.ctx.create_builder();
        match self.alloca_block.get_terminator() {
            Some(t) => b.position_before(&t),
            None => b.position_at_end(self.alloca_block),
        }
        let p = if n == 1 {
            b.build_alloca(ty, name).map_err(e)?
        } else {
            b.build_array_alloca(ty, self.e.i64c(n as i64), name).map_err(e)?
        };
        if let Some(inst) = p.as_instruction() {
            let _ = inst.set_alignment(64);
        }
        Ok(p)
    }

    /// Report `msg` and leave the unit if `ok` is false.
    pub fn check(&mut self, ok: IntValue<'c>, msg: &str) -> R<()> {
        if let Some(c) = ok.get_zero_extended_constant() {
            if c != 0 {
                return Ok(());
            }
        }
        let fail = self.e.ctx.append_basic_block(self.func, "fail");
        let cont = self.e.ctx.append_basic_block(self.func, "ok");
        self.e.b.build_conditional_branch(ok, cont, fail).map_err(e)?;
        self.e.b.position_at_end(fail);
        let m = self.e.cstr(msg)?;
        let f = self.e.declare("ptl_v2_fail", None, &[self.e.ptrt().into(), self.e.ptrt().into()]);
        self.e.b.build_call(f, &[self.ctx_ptr.into(), m.into()], "").map_err(e)?;
        self.e.b.build_store(self.rc, self.e.i32t().const_int(1, false)).map_err(e)?;
        self.e.b.build_unconditional_branch(self.exit).map_err(e)?;
        self.e.b.position_at_end(cont);
        Ok(())
    }

    /// Leave the unit if `rc != 0` (the callee set the error).
    fn check_rc(&mut self, rc: IntValue<'c>) -> R<()> {
        let fail = self.e.ctx.append_basic_block(self.func, "kfail");
        let cont = self.e.ctx.append_basic_block(self.func, "kok");
        let z = rc.get_type().const_zero();
        let ok = self.e.b.build_int_compare(IntPredicate::EQ, rc, z, "ok").map_err(e)?;
        self.e.b.build_conditional_branch(ok, cont, fail).map_err(e)?;
        self.e.b.position_at_end(fail);
        self.e.b.build_store(self.rc, rc).map_err(e)?;
        self.e.b.build_unconditional_branch(self.exit).map_err(e)?;
        self.e.b.position_at_end(cont);
        Ok(())
    }

    /// Product of the axis lengths.
    pub fn size(&self, shape: &[IntValue<'c>]) -> R<IntValue<'c>> {
        let mut n = self.e.i64c(1);
        for &s in shape {
            n = self.e.mul(n, s)?;
        }
        Ok(n)
    }

    fn c_strides(&self, dt: DType, shape: &[IntValue<'c>]) -> R<Vec<IntValue<'c>>> {
        let mut st = vec![self.e.i64c(0); shape.len()];
        let mut acc = self.e.i64c(dt.itemsize() as i64);
        for a in (0..shape.len()).rev() {
            st[a] = acc;
            acc = self.e.mul(acc, shape[a])?;
        }
        Ok(st)
    }

    /// Runtime buffer of `bytes`; the reference is stored in `owner_slot`.
    fn runtime_alloc(&mut self, bytes: IntValue<'c>, owner_slot: PointerValue<'c>) -> R<PointerValue<'c>> {
        let p = self.e.ptrt();
        let f = self.e.declare("ptl_v2_alloc", Some(p.into()), &[p.into(), self.e.i64t().into(), p.into()]);
        let data = self
            .e
            .b
            .build_call(f, &[self.ctx_ptr.into(), bytes.into(), owner_slot.into()], "buf")
            .map_err(e)?
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_pointer_value();
        let ok = self.e.b.build_is_not_null(data, "allocok").map_err(e)?;
        self.check_rc_bool(ok)?;
        Ok(data)
    }

    /// Like `check`, for errors the callee already reported.
    fn check_rc_bool(&mut self, ok: IntValue<'c>) -> R<()> {
        let fail = self.e.ctx.append_basic_block(self.func, "afail");
        let cont = self.e.ctx.append_basic_block(self.func, "aok");
        self.e.b.build_conditional_branch(ok, cont, fail).map_err(e)?;
        self.e.b.position_at_end(fail);
        self.e.b.build_store(self.rc, self.e.i32t().const_int(1, false)).map_err(e)?;
        self.e.b.build_unconditional_branch(self.exit).map_err(e)?;
        self.e.b.position_at_end(cont);
        Ok(())
    }

    /// A C-contiguous temporary, released when the unit returns.
    pub fn alloc_temp(&mut self, dt: DType, shape: &[IntValue<'c>]) -> R<ArrVal<'c>> {
        let strides = self.c_strides(dt, shape)?;
        let n = self.size(shape)?;
        let bytes = self.e.mul(n, self.e.i64c(dt.itemsize() as i64))?;
        let data = match Emitter::const_int_value(bytes) {
            Some(b) if (b as u64) <= MAX_STACK_BYTES => {
                let words = (b.max(1) as u64).div_ceil(8) as u32;
                self.alloca(self.e.i64t(), words, "tmp")?
            }
            _ => {
                let slot = self.alloca(self.e.ptrt(), 1, "owner")?;
                self.store_null_at_entry(slot)?;
                self.cleanups.push(slot);
                self.runtime_alloc(bytes, slot)?
            }
        };
        Ok(ArrVal { dtype: dt, data, shape: shape.to_vec(), strides, layout: Some(Layout::C), known: None })
    }

    /// Initialize an owner slot to NULL in the entry block, so exit can always release it.
    fn store_null_at_entry(&self, slot: PointerValue<'c>) -> R<()> {
        let b = self.e.ctx.create_builder();
        match self.alloca_block.get_terminator() {
            Some(t) => b.position_before(&t),
            None => b.position_at_end(self.alloca_block),
        }
        b.build_store(slot, self.e.ptrt().const_null()).map_err(e)?;
        Ok(())
    }

    /// Memory for output `o` of the node being emitted (a `Callee` output).
    pub fn alloc_output(&mut self, o: OutputIdx, dt: DType, shape: &[IntValue<'c>]) -> R<ArrVal<'c>> {
        match self.cur_targets.get(o.index()).copied().unwrap_or(OutTarget::Temp) {
            OutTarget::Temp => self.alloc_temp(dt, shape),
            OutTarget::UnitOutput(uo) => self.alloc_unit_output(uo, dt, shape),
        }
    }

    /// A runtime buffer that becomes unit output `uo` (released on error).
    fn alloc_unit_output(&mut self, uo: OutputIdx, dt: DType, shape: &[IntValue<'c>]) -> R<ArrVal<'c>> {
        let a = self.arg_ptr(self.sig.arg_of_output(uo))?;
        let owner_slot = self.field_ptr(a, 3)?;
        let strides = self.c_strides(dt, shape)?;
        let n = self.size(shape)?;
        let bytes = self.e.mul(n, self.e.i64c(dt.itemsize() as i64))?;
        let data = self.runtime_alloc(bytes, owner_slot)?;
        self.owned_outputs.push(uo);
        self.write_array_arg(a, data, shape, &strides)?;
        Ok(ArrVal { dtype: dt, data, shape: shape.to_vec(), strides, layout: Some(Layout::C), known: None })
    }

    // ------------------------------------------------------------ ABI arrays

    fn arg_ptr(&self, k: usize) -> R<PointerValue<'c>> {
        self.e.gep_bytes(self.args_ptr, self.e.i64c((k * PTARRAY_SIZE) as i64))
    }

    /// Pointer to field `f` (data=0, shape=1, strides=2, owner=3, ndim=4) of a PtArray.
    fn field_ptr(&self, a: PointerValue<'c>, f: usize) -> R<PointerValue<'c>> {
        self.e.gep_bytes(a, self.e.i64c((f * 8) as i64))
    }

    fn load_ptr(&self, p: PointerValue<'c>) -> R<PointerValue<'c>> {
        Ok(self.e.b.build_load(self.e.ptrt(), p, "lp").map_err(e)?.into_pointer_value())
    }

    /// An ABI array as an `ArrVal` of type `ty`.
    fn read_array_arg(&self, a: PointerValue<'c>, ty: &TensorTy) -> R<ArrVal<'c>> {
        let data = self.load_ptr(self.field_ptr(a, 0)?)?;
        let shape_p = self.load_ptr(self.field_ptr(a, 1)?)?;
        let strides_p = self.load_ptr(self.field_ptr(a, 2)?)?;
        let mut shape = Vec::with_capacity(ty.ndim());
        for (k, len) in ty.shape.iter().enumerate() {
            shape.push(match len.known() {
                Some(n) => self.e.i64c(n as i64),
                None => self.e.load_i64(shape_p, k)?,
            });
        }
        let strides = if ty.layout == Some(Layout::C) {
            self.c_strides(ty.dtype, &shape)?
        } else {
            (0..ty.ndim()).map(|k| self.e.load_i64(strides_p, k)).collect::<R<_>>()?
        };
        Ok(ArrVal { dtype: ty.dtype, data, shape, strides, layout: ty.layout, known: None })
    }

    /// Write data, shape and strides into an ABI array (which has its own storage).
    fn write_array_arg(&self, a: PointerValue<'c>, data: PointerValue<'c>, shape: &[IntValue<'c>], strides: &[IntValue<'c>]) -> R<()> {
        self.e.b.build_store(self.field_ptr(a, 0)?, data).map_err(e)?;
        let shape_p = self.load_ptr(self.field_ptr(a, 1)?)?;
        let strides_p = self.load_ptr(self.field_ptr(a, 2)?)?;
        for (k, (&s, &st)) in shape.iter().zip(strides).enumerate() {
            self.e.store_i64(shape_p, k, s)?;
            self.e.store_i64(strides_p, k, st)?;
        }
        Ok(())
    }

    /// Call kernel `callee` with the given inputs; outputs are given (Caller) or filled
    /// in by the kernel. Returns the outputs and their owner references (NULL for
    /// provided memory and views).
    fn call_kernel(
        &mut self,
        callee: CalleeIdx,
        sig: &Signature,
        inputs: &IndexVec<InputIdx, ArrVal<'c>>,
        provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, (ArrVal<'c>, PointerValue<'c>)>> {
        let n = sig.inputs.len() + sig.outputs.len();
        let arr_ty = self.e.ctx.i8_type().array_type(PTARRAY_SIZE as u32);
        let args = self.alloca(arr_ty, n.max(1) as u32, "kargs")?;
        let fill = |cx: &Self, k: usize, a: &ArrVal<'c>| -> R<PointerValue<'c>> {
            let p = cx.e.gep_bytes(args, cx.e.i64c((k * PTARRAY_SIZE) as i64))?;
            let nd = a.ndim().max(1) as u32;
            let shape = cx.alloca(cx.e.i64t(), nd, "kshape")?;
            let strides = cx.alloca(cx.e.i64t(), nd, "kstrides")?;
            cx.e.b.build_store(cx.field_ptr(p, 1)?, shape).map_err(e)?;
            cx.e.b.build_store(cx.field_ptr(p, 2)?, strides).map_err(e)?;
            cx.e.b.build_store(cx.field_ptr(p, 3)?, cx.e.ptrt().const_null()).map_err(e)?;
            cx.e.b.build_store(cx.field_ptr(p, 4)?, cx.e.i64c(a.ndim() as i64)).map_err(e)?;
            cx.write_array_arg(p, a.data, &a.shape, &a.strides)?;
            Ok(p)
        };
        for (i, a) in inputs.iter_enumerated() {
            fill(self, i.index(), a)?;
        }
        let mut out_ptrs = Vec::new();
        for (o, os) in sig.outputs.iter_enumerated() {
            let k = sig.arg_of_output(o);
            let p = match &provided[o] {
                Some(a) => fill(self, k, a)?,
                None => {
                    let p = self.e.gep_bytes(args, self.e.i64c((k * PTARRAY_SIZE) as i64))?;
                    let nd = os.ty.ndim().max(1) as u32;
                    let shape = self.alloca(self.e.i64t(), nd, "kshape")?;
                    let strides = self.alloca(self.e.i64t(), nd, "kstrides")?;
                    self.e.b.build_store(self.field_ptr(p, 0)?, self.e.ptrt().const_null()).map_err(e)?;
                    self.e.b.build_store(self.field_ptr(p, 1)?, shape).map_err(e)?;
                    self.e.b.build_store(self.field_ptr(p, 2)?, strides).map_err(e)?;
                    self.e.b.build_store(self.field_ptr(p, 3)?, self.e.ptrt().const_null()).map_err(e)?;
                    self.e.b.build_store(self.field_ptr(p, 4)?, self.e.i64c(os.ty.ndim() as i64)).map_err(e)?;
                    p
                }
            };
            out_ptrs.push(p);
        }
        let ptr = self.e.ptrt();
        let f = self.e.declare(
            "ptl_v2_call_kernel",
            Some(self.e.i32t().into()),
            &[ptr.into(), ptr.into(), self.e.i64t().into(), ptr.into()],
        );
        let rc = self
            .e
            .b
            .build_call(
                f,
                &[self.ctx_ptr.into(), self.env_ptr.into(), self.e.i64c(callee.0 as i64).into(), args.into()],
                "rc",
            )
            .map_err(e)?
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_int_value();
        self.check_rc(rc)?;
        let mut outs = IndexVec::new();
        for (o, os) in sig.outputs.iter_enumerated() {
            let p = out_ptrs[o.index()];
            let mut a = self.read_array_arg(p, &os.ty)?;
            let owner = self.load_ptr(self.field_ptr(p, 3)?)?;
            if let Some(pr) = &provided[o] {
                a.layout = pr.layout;
            }
            if os.alloc == OutputAlloc::Callee {
                a.layout = Some(Layout::C);
            }
            outs.push((a, owner));
        }
        Ok(outs)
    }

    /// Release this reference when the unit returns.
    fn release_at_exit(&mut self, owner: PointerValue<'c>) -> R<()> {
        let slot = self.alloca(self.e.ptrt(), 1, "kowner")?;
        self.store_null_at_entry(slot)?;
        self.e.b.build_store(slot, owner).map_err(e)?;
        self.cleanups.push(slot);
        Ok(())
    }

    // ------------------------------------------------------------ graphs

    /// Emit a graph inline. `inputs` are its input arrays; `targets` says where each
    /// graph output's memory is (provided memory, or allocate into a unit output).
    /// Returns the output arrays.
    pub fn emit_graph(
        &mut self,
        g: &Graph,
        inputs: &IndexVec<InputIdx, ArrVal<'c>>,
        out_provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
        out_targets: &IndexVec<OutputIdx, Option<OutputIdx>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>> {
        let mut vals: IndexVec<ValueId, Option<ArrVal<'c>>> = g.values.iter().map(|_| None).collect();
        for (i, &v) in g.inputs.iter_enumerated() {
            vals[v] = Some(inputs[i].clone());
        }
        // constants, in the order EnvBuilder::collect_graph lists them
        for (ci, c) in g.consts.iter_enumerated() {
            let v = const_value(g, ci);
            vals[v] = Some(self.const_array(c)?);
        }
        // which graph output each value provides memory for (first occurrence)
        let mut provided_for: IndexVec<ValueId, Option<OutputIdx>> = g.values.iter().map(|_| None).collect();
        for (o, &v) in g.outputs.iter_enumerated() {
            if provided_for[v].is_none() && matches!(g.values[v].origin, Origin::Node(..)) {
                provided_for[v] = Some(o);
            }
        }
        let mut written: IndexVec<OutputIdx, bool> = g.outputs.iter().map(|_| false).collect();

        for (n, node) in g.nodes.iter_enumerated() {
            let sig = g.signature(n);
            let ins: IndexVec<InputIdx, ArrVal<'c>> =
                node.inputs.iter().map(|&v| vals[v].clone().ok_or_else(|| format!("value {v:?} used before definition"))).collect::<R<_>>()?;
            let mut provided: IndexVec<OutputIdx, Option<ArrVal<'c>>> = IndexVec::new();
            let mut targets = Vec::new();
            for (o, os) in sig.outputs.iter_enumerated() {
                let v = node.outputs[o];
                let go = provided_for[v];
                let (p, t) = match (os.alloc, go) {
                    (OutputAlloc::Caller, Some(go)) => match &out_provided[go] {
                        Some(a) => {
                            written[go] = true;
                            (Some(a.clone()), OutTarget::Temp)
                        }
                        None => (Some(self.alloc_temp(os.ty.dtype, &static_shape_vals(&self.e, &os.ty))?), OutTarget::Temp),
                    },
                    (OutputAlloc::Caller, None) => {
                        (Some(self.alloc_temp(os.ty.dtype, &static_shape_vals(&self.e, &os.ty))?), OutTarget::Temp)
                    }
                    (OutputAlloc::Callee, Some(go)) => match out_targets[go] {
                        Some(uo) if out_provided[go].is_none() => {
                            written[go] = true;
                            (None, OutTarget::UnitOutput(uo))
                        }
                        _ => (None, OutTarget::Temp),
                    },
                    _ => (None, OutTarget::Temp),
                };
                provided.push(p);
                targets.push(t);
            }
            let outs = match node.op.generate() {
                Some(generator) => {
                    let n_owned = self.owned_outputs.len();
                    self.cur_targets = targets.clone();
                    let r = generator.emit(self, &sig, &ins, &provided);
                    self.cur_targets.clear();
                    let outs = r.map_err(|m| format!("{:?}: {m}", node.op))?;
                    for t in &targets {
                        if let OutTarget::UnitOutput(uo) = *t {
                            if !self.owned_outputs[n_owned..].contains(&uo) {
                                return Err(format!("{:?} did not allocate its output with alloc_output", node.op));
                            }
                        }
                    }
                    outs
                }
                None => {
                    let callee = self.take_callee()?;
                    let res = self.call_kernel(callee, &sig, &ins, &provided)?;
                    let mut outs = IndexVec::new();
                    for (o, (a, owner)) in res.iter_enumerated() {
                        match targets[o.index()] {
                            OutTarget::UnitOutput(uo) => {
                                // the kernel's buffer becomes the unit output
                                let arg = self.arg_ptr(self.sig.arg_of_output(uo))?;
                                self.e.b.build_store(self.field_ptr(arg, 3)?, *owner).map_err(e)?;
                                self.owned_outputs.push(uo);
                                self.write_array_arg(arg, a.data, &a.shape, &a.strides)?;
                            }
                            OutTarget::Temp => {
                                if sig.outputs[o].alloc != OutputAlloc::Caller {
                                    self.release_at_exit(*owner)?;
                                }
                            }
                        }
                        outs.push(a.clone());
                    }
                    outs
                }
            };
            if outs.len() != node.outputs.len() {
                return Err(format!("{:?} produced {} outputs, expected {}", node.op, outs.len(), node.outputs.len()));
            }
            for (o, a) in outs.iter_enumerated() {
                vals[node.outputs[o]] = Some(a.clone());
            }
        }

        // graph outputs not written in place: copy
        let mut res = IndexVec::new();
        for (o, &v) in g.outputs.iter_enumerated() {
            let a = vals[v].clone().ok_or("graph output never computed")?;
            if written[o] {
                res.push(match &out_provided[o] {
                    Some(p) => p.clone(),
                    None => a,
                });
                continue;
            }
            let dst = match (&out_provided[o], out_targets[o]) {
                (Some(p), _) => p.clone(),
                (None, Some(uo)) => self.alloc_unit_output(uo, a.dtype, &a.shape.clone())?,
                (None, None) => {
                    // a nested graph's output that nobody provided memory for: views
                    // and values can be used as they are
                    res.push(a);
                    continue;
                }
            };
            super::ops::elemwise::copy_array(self, &a, &dst)?;
            res.push(dst);
        }
        Ok(res)
    }

    /// Emit a nested graph (a compiled OpFromGraph) in place of the current node.
    pub fn emit_nested_graph(
        &mut self,
        g: &Graph,
        inputs: &IndexVec<InputIdx, ArrVal<'c>>,
        provided: &IndexVec<OutputIdx, Option<ArrVal<'c>>>,
    ) -> R<IndexVec<OutputIdx, ArrVal<'c>>> {
        let targets: IndexVec<OutputIdx, Option<OutputIdx>> = provided
            .indices()
            .map(|o| match self.cur_targets.get(o.index()) {
                Some(OutTarget::UnitOutput(uo)) => Some(*uo),
                _ => None,
            })
            .collect();
        self.emit_graph(g, inputs, provided, &targets)
    }

    fn take_callee(&mut self) -> R<CalleeIdx> {
        if self.next_callee >= self.n_callees {
            return Err("callee table out of sync with the generated code".into());
        }
        self.next_callee += 1;
        Ok(CalleeIdx::new(self.next_callee - 1))
    }

    fn const_array(&mut self, c: &ConstData) -> R<ArrVal<'c>> {
        let shape: Vec<IntValue> = c.shape.iter().map(|&n| self.e.i64c(n)).collect();
        let strides: Vec<IntValue> = c.strides.iter().map(|&s| self.e.i64c(s)).collect();
        if c.shape.is_empty() {
            // folded into the code; memory only for ops that read through pointers
            let (f, i) = c.scalar().unwrap();
            let slot = self.alloca(self.e.memty(c.dtype), 1, "c0")?;
            let v = self.e.constant(c.dtype, f, i);
            self.e.store_elem(slot, v, c.dtype)?;
            return Ok(ArrVal { dtype: c.dtype, data: slot, shape, strides, layout: Some(Layout::C), known: Some((f, i)) });
        }
        if self.next_const >= self.n_consts {
            return Err("constant table out of sync with the generated code".into());
        }
        let k = self.next_const;
        self.next_const += 1;
        let consts_p = self.load_ptr(self.e.gep_bytes(self.env_ptr, self.e.i64c(8))?)?;
        let p = self.e.gep_elem(DType::Int64, consts_p, self.e.i64c(k as i64))?;
        let data = self.load_ptr(p)?;
        Ok(ArrVal { dtype: c.dtype, data, shape, strides, layout: Some(Layout::C), known: None })
    }
}

fn const_value(g: &Graph, c: ConstIdx) -> ValueId {
    g.values
        .iter_enumerated()
        .find(|(_, v)| v.origin == Origin::Const(c))
        .map(|(id, _)| id)
        .expect("constant without a value")
}

/// Static shape of a type as LLVM constants (the type must be static).
pub fn static_shape_vals<'c>(e: &Emitter<'c>, ty: &TensorTy) -> Vec<IntValue<'c>> {
    ty.shape.iter().map(|a| e.i64c(a.known().expect("static shape") as i64)).collect()
}

// ------------------------------------------------------------------ building a unit

/// A unit to compile: a graph with the signature it is called with.
pub struct UnitSpec {
    pub graph: Arc<Graph>,
    pub sig: Signature,
    pub fastmath: bool,
}

impl UnitSpec {
    /// Cache key: everything the generated code depends on.
    pub fn key(&self) -> R<String> {
        let mut k = Key::new();
        k.bool(self.fastmath);
        k.debug(&self.sig);
        graph_key(&self.graph, &mut k)?;
        Ok(k.finish())
    }

    pub fn env(&self) -> R<Env> {
        let mut b = EnvBuilder::default();
        b.collect_graph(&self.graph)?;
        Ok(b.env)
    }

    pub fn describe(&self) -> String {
        let ops: Vec<String> = self.graph.nodes.iter().map(|n| format!("{:?}", n.op)).collect();
        let mut s = ops.join(", ");
        if s.len() > 120 {
            s.truncate(117);
            s.push_str("...");
        }
        s
    }
}

/// Hash a graph's structure, types and generated ops.
pub fn graph_key(g: &Graph, k: &mut Key) -> R<()> {
    k.u64(g.values.len() as u64);
    for v in g.values.iter() {
        k.debug(&v.ty);
        k.debug(&v.origin);
    }
    for c in g.consts.iter() {
        k.debug(&c.dtype);
        k.debug(&c.shape);
        if c.shape.is_empty() {
            k.bytes(c.bytes());
        }
    }
    k.u64(g.nodes.len() as u64);
    for node in g.nodes.iter() {
        match node.op.generate() {
            Some(generator) => {
                k.u64(1);
                generator.key(k);
            }
            None => k.u64(0), // called through the callee table
        }
        k.debug(&node.inputs);
        k.debug(&node.outputs);
        k.debug(&node.views);
    }
    k.debug(&g.inputs);
    k.debug(&g.outputs);
    Ok(())
}

/// Generate the unit function `name` into the emitter's module.
pub fn generate_unit<'c>(em: Emitter<'c>, name: &str, spec: &UnitSpec, n_callees: usize, n_consts: usize) -> R<Emitter<'c>> {
    let ptr = em.ptrt();
    let fty = em.i32t().fn_type(&[ptr.into(), ptr.into(), ptr.into()], false);
    let func = em.module.add_function(name, fty, None);
    let alloca_block = em.ctx.append_basic_block(func, "entry");
    let body = em.ctx.append_basic_block(func, "body");
    let exit = em.ctx.append_basic_block(func, "exit");
    em.b.position_at_end(alloca_block);
    let rc = em.b.build_alloca(em.i32t(), "rc").map_err(e)?;
    em.b.build_store(rc, em.i32t().const_zero()).map_err(e)?;
    em.b.build_unconditional_branch(body).map_err(e)?;
    em.b.position_at_end(body);

    let p = |i: u32| func.get_nth_param(i).unwrap().into_pointer_value();
    let mut cx = UnitCx {
        e: em,
        func,
        ctx_ptr: p(0),
        args_ptr: p(1),
        env_ptr: p(2),
        sig: spec.sig.clone(),
        alloca_block,
        exit,
        rc,
        cleanups: vec![],
        owned_outputs: vec![],
        cur_targets: vec![],
        next_callee: 0,
        next_const: 0,
        n_callees,
        n_consts,
    };

    let g = &spec.graph;
    let mut inputs = IndexVec::new();
    for (i, is) in spec.sig.inputs.iter_enumerated() {
        let a = cx.arg_ptr(i.index())?;
        inputs.push(cx.read_array_arg(a, &is.ty)?);
    }
    let mut provided = IndexVec::new();
    let mut targets = IndexVec::new();
    for (o, os) in spec.sig.outputs.iter_enumerated() {
        match os.alloc {
            OutputAlloc::Caller => {
                let a = cx.arg_ptr(spec.sig.arg_of_output(o))?;
                let mut v = cx.read_array_arg(a, &os.ty)?;
                // the caller provides C-contiguous memory for Caller outputs
                v.strides = cx.c_strides(os.ty.dtype, &v.shape)?;
                v.layout = Some(Layout::C);
                provided.push(Some(v));
                targets.push(None);
            }
            OutputAlloc::Callee => {
                provided.push(None);
                targets.push(Some(o));
            }
            OutputAlloc::ViewOf(_) => {
                // only one-node units of view ops: the output stays a view, see below
                provided.push(None);
                targets.push(None);
            }
        }
    }
    let outs = cx.emit_graph(g, &inputs, &provided, &targets)?;
    // view outputs: point the argument at the view; the owner is the viewed input's
    for (o, os) in spec.sig.outputs.iter_enumerated() {
        if let OutputAlloc::ViewOf(i) = os.alloc {
            let arg = cx.arg_ptr(spec.sig.arg_of_output(o))?;
            let src = cx.arg_ptr(i.index())?;
            let owner = cx.load_ptr(cx.field_ptr(src, 3)?)?;
            cx.e.b.build_store(cx.field_ptr(arg, 3)?, owner).map_err(e)?;
            let a = &outs[o];
            cx.write_array_arg(arg, a.data, &a.shape, &a.strides)?;
        }
    }
    if spec.sig.outputs.iter().any(|os| matches!(os.alloc, OutputAlloc::ViewOf(_))) && g.nodes.len() != 1 {
        return Err("a unit with view outputs must be a single node".into());
    }
    cx.e.b.build_unconditional_branch(exit).map_err(e)?;

    // exit: release temporaries; on error also the outputs we allocated
    cx.e.b.position_at_end(exit);
    let dec = cx.e.declare("ptl_v2_decref", None, &[ptr.into()]);
    for slot in cx.cleanups.clone() {
        let b = cx.load_ptr(slot)?;
        cx.e.b.build_call(dec, &[b.into()], "").map_err(e)?;
    }
    let rcv = cx.e.b.build_load(cx.e.i32t(), rc, "rcv").map_err(e)?.into_int_value();
    if !cx.owned_outputs.is_empty() {
        let failed = cx.e.b.build_int_compare(IntPredicate::NE, rcv, cx.e.i32t().const_zero(), "failed").map_err(e)?;
        let owned = cx.owned_outputs.clone();
        let release = || -> R<()> {
            for &uo in &owned {
                let arg = cx.arg_ptr(spec.sig.arg_of_output(uo))?;
                let slot = cx.field_ptr(arg, 3)?;
                let b = cx.load_ptr(slot)?;
                cx.e.b.build_call(dec, &[b.into()], "").map_err(e)?;
                cx.e.b.build_store(slot, ptr.const_null()).map_err(e)?;
                cx.e.b.build_store(cx.field_ptr(arg, 0)?, ptr.const_null()).map_err(e)?;
            }
            Ok(())
        };
        let ee = &cx.e;
        let f = ee.cur_fn();
        let tb = ee.ctx.append_basic_block(f, "release");
        let done = ee.ctx.append_basic_block(f, "done");
        ee.b.build_conditional_branch(failed, tb, done).map_err(e)?;
        ee.b.position_at_end(tb);
        release()?;
        cx.e.b.build_unconditional_branch(done).map_err(e)?;
        cx.e.b.position_at_end(done);
    }
    cx.e.b.build_return(Some(&rcv)).map_err(e)?;
    if cx.next_callee != n_callees || cx.next_const != n_consts {
        return Err(format!(
            "environment mismatch: used {}/{} callees, {}/{} constants",
            cx.next_callee, n_callees, cx.next_const, n_consts
        ));
    }
    Ok(cx.e)
}
