//! LLVM setup shared by the JIT and the compiler: target initialization, the target
//! machine, the vector math library, and the object cache directory.

use inkwell::OptimizationLevel;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();
static VECLIB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn init_llvm() {
    INIT.call_once(|| {
        Target::initialize_native(&InitializationConfig::default()).expect("init native target");
        // Must happen before any pass pipeline is built. Process-global, so decided once.
        let veclib = std::env::var("PYTENSOR_LLVM_NO_VECLIB").is_err();
        if veclib {
            let args = [c"ptllvm".as_ptr(), c"-vector-library=LIBMVEC-X86".as_ptr()];
            // SAFETY: two valid NUL-terminated strings; called once (inside INIT), before
            // any LLVM pass pipeline exists, as LLVM requires for global options.
            unsafe {
                llvm_sys::support::LLVMParseCommandLineOptions(2, args.as_ptr(), std::ptr::null())
            };
            VECLIB.store(true, std::sync::atomic::Ordering::Release);
        }
    });
}

pub fn veclib_enabled() -> bool {
    init_llvm();
    VECLIB.load(std::sync::atomic::Ordering::Acquire)
}

/// Target CPU and features: the host, or `PYTENSOR_LLVM_CPU` (e.g. "haswell" to run the
/// generated code under valgrind, which has no AVX-512). Part of the cache key.
pub fn target_cpu() -> (String, String) {
    match std::env::var("PYTENSOR_LLVM_CPU") {
        Ok(cpu) if !cpu.is_empty() => (cpu, String::new()),
        _ => (
            TargetMachine::get_host_cpu_name().to_string(),
            TargetMachine::get_host_cpu_features().to_string(),
        ),
    }
}

/// libmvec ISA variants usable on this target: SSE ('b', 2 x f64) and AVX2 ('d', 4 x f64).
pub fn vfabi_variants() -> Vec<(char, u32)> {
    if !veclib_enabled() {
        return vec![];
    }
    let (cpu, features) = target_cpu();
    if !features.is_empty() || cpu.is_empty() {
        let mut v = vec![('b', 2)];
        if features.split(',').any(|f| f == "+avx2") {
            v.push(('d', 4));
        }
        return v;
    }
    // named CPU: ask LLVM whether it has AVX2 via a target machine
    let features = String::from(
        if [
            "haswell",
            "broadwell",
            "skylake",
            "znver1",
            "znver2",
            "znver3",
            "x86-64-v3",
        ]
        .contains(&cpu.as_str())
        {
            "+avx2"
        } else {
            ""
        },
    );
    let mut v = vec![('b', 2)];
    if features.split(',').any(|f| f == "+avx2") {
        v.push(('d', 4));
    }
    v
}

pub fn target_machine(opt: u32) -> Result<TargetMachine, String> {
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let (cpu, features) = target_cpu();
    let level = match opt {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        _ => OptimizationLevel::Aggressive,
    };
    target
        .create_target_machine(
            &triple,
            &cpu,
            &features,
            level,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| "could not create target machine".to_string())
}

pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PYTENSOR_LLVM_CACHE_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache").join("pytensor-llvm")
}
