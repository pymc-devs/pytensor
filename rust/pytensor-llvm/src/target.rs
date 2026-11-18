//! LLVM setup shared by the JIT and the compiler: target initialization, the target
//! machine, the vector math library, and the object cache directory.

use inkwell::OptimizationLevel;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use std::path::PathBuf;
use std::sync::{Once, OnceLock};

static INIT: Once = Once::new();
static VECLIB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn init_llvm() {
    INIT.call_once(|| {
        Target::initialize_native(&InitializationConfig::default()).expect("init native target");
        // Process-global, so decided once. Whether libmvec variants are advertised to the
        // vectorizer (`vector-function-abi-variant` on libm calls, see `Builder::libm`).
        let veclib = std::env::var("PYTENSOR_LLVM_NO_VECLIB").is_err();
        if veclib {
            VECLIB.store(true, std::sync::atomic::Ordering::Release);
        }
    });
}

pub fn veclib_enabled() -> bool {
    init_llvm();
    VECLIB.load(std::sync::atomic::Ordering::Acquire)
}

/// Target CPU and features of the host. Part of the cache key.
pub fn target_cpu() -> (String, String) {
    (
        TargetMachine::get_host_cpu_name().to_string(),
        TargetMachine::get_host_cpu_features().to_string(),
    )
}

/// Load glibc's vector math library into the process, once. False where there is none
/// (macOS, Windows, musl).
fn libmvec_loaded() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    static LOADED: OnceLock<bool> = OnceLock::new();
    // SAFETY: a NUL-terminated library name; LLVM keeps the library loaded.
    *LOADED.get_or_init(|| unsafe {
        llvm_sys::support::LLVMLoadLibraryPermanently(c"libmvec.so.1".as_ptr()) == 0
    })
}

/// Whether the process (including libmvec) exports `name`. Which functions libmvec has
/// depends on the glibc version.
pub fn has_symbol(name: &str) -> bool {
    let Ok(name) = std::ffi::CString::new(name) else {
        return false;
    };
    // SAFETY: a NUL-terminated symbol name; only the address is checked.
    unsafe { !llvm_sys::support::LLVMSearchForAddressOfSymbol(name.as_ptr()).is_null() }
}

/// libmvec ISA variants usable on this target, as (VFABI ISA letter, f64 lanes):
/// x86-64 SSE ('b') and AVX2 ('d'), aarch64 AdvSIMD ('n').
pub fn vfabi_variants() -> Vec<(char, u32)> {
    if !veclib_enabled() || !libmvec_loaded() {
        return vec![];
    }
    match std::env::consts::ARCH {
        "x86_64" => {
            let (_, features) = target_cpu();
            let mut v = vec![('b', 2)];
            if features.split(',').any(|f| f == "+avx2") {
                v.push(('d', 4));
            }
            v
        }
        "aarch64" => vec![('n', 2)],
        _ => vec![],
    }
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
    // LLJIT links COFF objects with RuntimeDyld, which makes no stubs for far calls: with
    // the small code model, calls into DLLs and the Rust functions more than 2 GiB away
    // from the JIT'd code overflow their 32-bit displacements. JITLink (ELF, Mach-O)
    // makes stubs, so elsewhere the small model is fine.
    let code_model = if cfg!(all(windows, target_arch = "x86_64")) {
        CodeModel::Large
    } else {
        CodeModel::Default
    };
    target
        .create_target_machine(&triple, &cpu, &features, level, RelocMode::PIC, code_model)
        .ok_or_else(|| "could not create target machine".to_string())
}

/// `PYTENSOR_LLVM_CACHE_DIR`, else the platform's per-user cache directory.
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PYTENSOR_LLVM_CACHE_DIR") {
        return PathBuf::from(d);
    }
    let var = |k| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(windows) {
        var("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        var("HOME").map(|h| h.join("Library").join("Caches"))
    } else {
        var("HOME").map(|h| h.join(".cache"))
    };
    base.unwrap_or_else(std::env::temp_dir).join("pytensor-llvm")
}
