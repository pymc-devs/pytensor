//! Parallel kernel compilation with a content-addressed object cache.
//!
//! Each kernel is its own compilation unit: own LLVM context, own module, own object file.
//! Units are compiled on a rayon pool and cached on disk by spec hash, so a kernel that
//! shows up in many graphs (the same logp terms across models) is compiled once, ever.

use crate::codegen::Cg;
use crate::jit::jit;
use crate::spec::{KernelSpec, CODEGEN_VERSION};
use inkwell::context::Context;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::OptimizationLevel;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Options {
    pub fastmath: bool,
    /// Merge runs of statically small nodes into segments compiled as one unit.
    pub segments: bool,
    /// A node is small if all arrays it touches have at most this many elements.
    pub segment_max_elems: i64,
    pub opt_level: u32,
    pub cache: bool,
    pub dump_ir: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Options { fastmath: true, segments: true, segment_max_elems: 64, opt_level: 3, cache: true, dump_ir: None }
    }
}

#[derive(Clone, Debug, Default)]
pub struct KernelStats {
    pub n_kernels: usize,
    pub n_unique: usize,
    pub n_in_process: usize,
    pub n_disk_cache: usize,
    pub n_compiled: usize,
    /// Sum over kernels of codegen + optimization + emission time (CPU seconds).
    pub llvm_cpu_s: f64,
    pub wall_s: f64,
    pub link_s: f64,
}

static INIT: Once = Once::new();
static mut VECLIB: bool = false;

pub fn init_llvm() {
    INIT.call_once(|| {
        Target::initialize_native(&InitializationConfig::default()).expect("init native target");
        // Must happen before any pass pipeline is built. Process-global, so decided once.
        let veclib = std::env::var("PTLLVM_NO_VECLIB").is_err();
        if veclib {
            let args = [c"ptllvm".as_ptr(), c"-vector-library=LIBMVEC-X86".as_ptr()];
            unsafe {
                llvm_sys::support::LLVMParseCommandLineOptions(2, args.as_ptr(), std::ptr::null());
                VECLIB = true;
            }
        }
    });
}

pub fn veclib_enabled() -> bool {
    init_llvm();
    unsafe { VECLIB }
}

/// libmvec ISA variants usable on this host: SSE ('b', 2 x f64) and AVX2 ('d', 4 x f64).
fn vfabi_variants() -> Vec<(char, u32)> {
    if !veclib_enabled() {
        return vec![];
    }
    let features = TargetMachine::get_host_cpu_features().to_string();
    let mut v = vec![('b', 2)];
    if features.split(',').any(|f| f == "+avx2") {
        v.push(('d', 4));
    }
    v
}

fn target_machine(opt: u32) -> Result<TargetMachine, String> {
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let cpu = TargetMachine::get_host_cpu_name().to_string();
    let features = TargetMachine::get_host_cpu_features().to_string();
    let level = match opt {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        _ => OptimizationLevel::Aggressive,
    };
    target
        .create_target_machine(&triple, &cpu, &features, level, RelocMode::PIC, CodeModel::Default)
        .ok_or_else(|| "could not create target machine".to_string())
}

pub fn salt(opts: &Options) -> String {
    let (mut ma, mut mi, mut pa) = (0u32, 0u32, 0u32);
    unsafe { llvm_sys::core::LLVMGetVersion(&mut ma, &mut mi, &mut pa) };
    format!(
        "v{CODEGEN_VERSION}|{}|llvm{ma}.{mi}.{pa}|{}|{}|O{}|veclib{}|fp{}",
        env!("CARGO_PKG_VERSION"),
        TargetMachine::get_host_cpu_name().to_string(),
        TargetMachine::get_host_cpu_features().to_string(),
        opts.opt_level,
        veclib_enabled(),
        crate::perf::enabled(),
    )
}

fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PTLLVM_CACHE_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache").join("pytensor-llvm-poc")
}

/// Compile one kernel spec to an object file.
pub fn compile_one(name: &str, spec: &KernelSpec, opts: &Options) -> Result<Vec<u8>, String> {
    let fastmath = match spec {
        KernelSpec::Elemwise { fastmath, .. } | KernelSpec::Reduce { fastmath, .. } => *fastmath,
        KernelSpec::Segment(s) => s.fastmath,
        KernelSpec::Fused { fastmath, .. } => *fastmath,
    };
    let ctx = Context::create();
    let cg = Cg::new(&ctx, name, fastmath, vfabi_variants());
    let tm = target_machine(opts.opt_level)?;
    cg.module.set_triple(&tm.get_triple());
    cg.module.set_data_layout(&tm.get_target_data().get_data_layout());
    cg.gen_kernel(name, spec)?;
    crate::perf::add_frame_pointers(&cg.module);
    cg.module.verify().map_err(|e| format!("invalid IR for {}: {}", spec.describe(), e))?;
    let pbo = PassBuilderOptions::create();
    pbo.set_loop_vectorization(true);
    pbo.set_loop_slp_vectorization(true);
    pbo.set_loop_interleaving(true);
    pbo.set_loop_unrolling(true);
    let pipeline = format!("default<O{}>", opts.opt_level.min(3));
    cg.module.run_passes(&pipeline, &tm, pbo).map_err(|e| e.to_string())?;
    if let Some(dir) = &opts.dump_ir {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(format!("{name}.ll")), cg.module.print_to_string().to_string());
    }
    let buf = tm
        .write_to_memory_buffer(&cg.module, FileType::Object)
        .map_err(|e| e.to_string())?;
    Ok(buf.as_slice().to_vec())
}

/// Resolve function pointers for all specs, compiling what's missing in parallel.
pub fn get_kernels(specs: &[KernelSpec], opts: &Options) -> Result<(Vec<usize>, Vec<String>, KernelStats), String> {
    let t0 = Instant::now();
    init_llvm();
    let salt = salt(opts);
    let names: Vec<String> = specs.iter().map(|s| format!("ptk_{}", s.hash(&salt))).collect();
    let mut stats = KernelStats { n_kernels: specs.len(), ..Default::default() };

    // unique kernels not yet linked into this process
    let mut todo: Vec<usize> = Vec::new();
    {
        let j = jit().lock().unwrap();
        let mut seen = std::collections::HashSet::new();
        for (i, n) in names.iter().enumerate() {
            if !seen.insert(n.clone()) {
                continue;
            }
            stats.n_unique += 1;
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
    // (index, object bytes, compiled?, cpu seconds)
    let results: Vec<Result<(usize, Vec<u8>, bool, f64), String>> = todo
        .par_iter()
        .map(|&i| {
            let path = dir.join(format!("{}.o", names[i]));
            if opts.cache {
                if let Ok(bytes) = std::fs::read(&path) {
                    return Ok((i, bytes, false, 0.0));
                }
            }
            let t = Instant::now();
            let obj = compile_one(&names[i], &specs[i], opts)?;
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

    let tl = Instant::now();
    let mut j = jit().lock().unwrap();
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
    let addrs: Vec<usize> = names
        .iter()
        .map(|n| j.has(n).ok_or_else(|| format!("kernel {n} missing")))
        .collect::<Result<_, _>>()?;
    stats.link_s = tl.elapsed().as_secs_f64();
    stats.wall_s = t0.elapsed().as_secs_f64();
    Ok((addrs, names, stats))
}
