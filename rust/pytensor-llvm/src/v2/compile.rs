//! Compiling units: in parallel, with the content-addressed object cache.

use super::interp::Options;
use super::op::{Kernel, KernelArgs, R};
use super::unit::{generate_unit, UnitEnv, UnitSpec};
use crate::abi::{PtArray, PtCtx};
use crate::compile::{cache_dir, init_llvm, target_cpu, target_machine, vfabi_variants};
use crate::jit::jit;
use inkwell::context::Context;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::FileType;
use rayon::prelude::*;
use std::time::Instant;

/// Bump when generated code changes in a way the cache key can't see.
pub const UNIT_CODEGEN_VERSION: u32 = 2;

#[derive(Clone, Debug, Default)]
pub struct CompileStats {
    pub n_units: usize,
    pub n_in_process: usize,
    pub n_disk_cache: usize,
    pub n_compiled: usize,
    /// summed over compiled units (CPU seconds)
    pub llvm_cpu_s: f64,
    /// wall time of compile_units
    pub wall_s: f64,
    /// Program::build, including compile_units
    pub build_s: f64,
}

impl CompileStats {
    pub fn add(&mut self, o: &CompileStats) {
        self.n_units += o.n_units;
        self.n_in_process += o.n_in_process;
        self.n_disk_cache += o.n_disk_cache;
        self.n_compiled += o.n_compiled;
        self.llvm_cpu_s += o.llvm_cpu_s;
        self.wall_s += o.wall_s;
    }
}

type UnitFn = unsafe extern "C-unwind" fn(*mut PtCtx, *mut PtArray, *const super::unit::EnvRepr) -> i32;

/// A compiled unit, callable as a kernel.
pub struct UnitKernel {
    f: UnitFn,
    env: UnitEnv,
    desc: String,
}

impl Kernel for UnitKernel {
    fn call(&self, ctx: &mut PtCtx, args: &mut KernelArgs<'_>) -> R<()> {
        let env = self.env.repr();
        // SAFETY: the unit was generated for exactly this signature and environment;
        // KernelArgs guarantees valid arguments for it.
        let rc = unsafe { (self.f)(ctx, args.raw().as_mut_ptr(), &env) };
        if rc != 0 {
            let msg = ctx.error.take().map(|(_, m)| m).unwrap_or_else(|| format!("error {rc}"));
            return Err(msg);
        }
        Ok(())
    }
}

impl std::fmt::Debug for UnitKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unit({})", self.desc)
    }
}

fn salt(opts: &Options) -> String {
    let (mut ma, mut mi, mut pa) = (0u32, 0u32, 0u32);
    // SAFETY: plain out-parameters.
    unsafe { llvm_sys::core::LLVMGetVersion(&mut ma, &mut mi, &mut pa) };
    let (cpu, features) = target_cpu();
    format!(
        "u{UNIT_CODEGEN_VERSION}|{}|llvm{ma}.{mi}.{pa}|{cpu}|{features}|O{}|veclib{:?}|fp{}",
        env!("CARGO_PKG_VERSION"),
        opts.opt_level,
        vfabi_variants(),
        crate::perf::enabled(),
    )
}

/// Object code for one unit.
fn compile_one(name: &str, spec: &UnitSpec, n_callees: usize, n_consts: usize, opts: &Options) -> R<Vec<u8>> {
    let ctx = Context::create();
    let em = super::emit::Emitter::new(&ctx, name, spec.fastmath, vfabi_variants());
    let tm = target_machine(opts.opt_level)?;
    em.module.set_triple(&tm.get_triple());
    em.module.set_data_layout(&tm.get_target_data().get_data_layout());
    let em = generate_unit(em, name, spec, n_callees, n_consts)?;
    em.finish();
    crate::perf::add_frame_pointers(&em.module);
    em.module.verify().map_err(|m| format!("invalid IR for {}: {}", spec.describe(), m))?;
    let pbo = PassBuilderOptions::create();
    pbo.set_loop_vectorization(true);
    pbo.set_loop_slp_vectorization(true);
    pbo.set_loop_interleaving(true);
    pbo.set_loop_unrolling(true);
    em.module.run_passes(&format!("default<O{}>", opts.opt_level.min(3)), &tm, pbo).map_err(|m| m.to_string())?;
    if let Ok(dir) = std::env::var("PTLLVM_DUMP_IR") {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ll")), em.module.print_to_string().to_string());
    }
    let buf = tm.write_to_memory_buffer(&em.module, FileType::Object).map_err(|m| m.to_string())?;
    Ok(buf.as_slice().to_vec())
}

/// Compile (or find) every unit. Returns kernels in the order of `specs`.
pub fn compile_units(specs: &[UnitSpec], opts: &Options) -> R<(Vec<UnitKernel>, CompileStats)> {
    let t0 = Instant::now();
    init_llvm();
    let salt = salt(opts);
    let mut stats = CompileStats { n_units: specs.len(), ..Default::default() };
    let envs: Vec<super::unit::Env> = specs.iter().map(|s| s.env()).collect::<R<_>>()?;
    let names: Vec<String> = specs
        .iter()
        .map(|s| {
            let mut k = super::op::Key::new();
            k.str(&salt);
            k.str(&s.key()?);
            Ok(format!("ptu_{}", k.finish()))
        })
        .collect::<R<_>>()?;

    let mut todo: Vec<usize> = Vec::new();
    {
        let j = jit().lock().unwrap_or_else(|e| e.into_inner());
        let mut seen = std::collections::HashSet::new();
        for (i, n) in names.iter().enumerate() {
            if !seen.insert(n.clone()) {
                continue;
            }
            if j.has(n).is_some() {
                stats.n_in_process += 1;
            } else {
                todo.push(i);
            }
        }
    }

    let dir = cache_dir();
    if opts.cache {
        let _ = std::fs::create_dir_all(&dir);
    }
    let results: Vec<R<(usize, Vec<u8>, bool, f64)>> = todo
        .par_iter()
        .map(|&i| {
            let path = dir.join(format!("{}.o", names[i]));
            if opts.cache {
                if let Ok(bytes) = std::fs::read(&path) {
                    return Ok((i, bytes, false, 0.0));
                }
            }
            let t = Instant::now();
            let obj = compile_one(&names[i], &specs[i], envs[i].callees.len(), envs[i].consts.len(), opts)?;
            let dt = t.elapsed().as_secs_f64();
            if opts.cache {
                // atomic publish so concurrent processes never see partial files
                let tmp = dir.join(format!("{}.{}.tmp", names[i], std::process::id()));
                if std::fs::write(&tmp, &obj).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
            Ok((i, obj, true, dt))
        })
        .collect();

    let mut j = jit().lock().unwrap_or_else(|e| e.into_inner());
    for r in results {
        let (i, obj, compiled, dt) = r?;
        if compiled {
            stats.n_compiled += 1;
            stats.llvm_cpu_s += dt;
        } else {
            stats.n_disk_cache += 1;
        }
        j.add_object(&names[i], &obj, &specs[i].describe())?;
    }
    let mut kernels = Vec::with_capacity(specs.len());
    for ((name, spec), env) in names.iter().zip(specs).zip(envs) {
        let addr = j.has(name).ok_or_else(|| format!("unit {name} missing"))?;
        // SAFETY: `addr` is the unit function generated by `generate_unit`, which has
        // the `UnitFn` signature.
        let f: UnitFn = unsafe { std::mem::transmute::<usize, UnitFn>(addr) };
        kernels.push(UnitKernel { f, env: UnitEnv::new(env), desc: spec.describe() });
    }
    stats.wall_s = t0.elapsed().as_secs_f64();
    Ok((kernels, stats))
}
