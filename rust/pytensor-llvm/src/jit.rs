//! Process-wide ORC LLJIT instance.
//!
//! We never hand LLVM IR to the JIT. Kernels are compiled to relocatable object files
//! (in parallel, one LLVM context per kernel) and only the finished objects are linked
//! here. That's what makes the disk cache trivial: a cache hit skips LLVM codegen
//! entirely, and the JIT only links.

use crate::native;
use llvm_sys::error::*;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::{Mutex, OnceLock};

pub struct Jit {
    jit: LLVMOrcLLJITRef,
    jd: LLVMOrcJITDylibRef,
    symbols: HashMap<String, usize>,
}

// The LLJIT is internally synchronized; we additionally guard it with a Mutex.
// SAFETY: the LLJIT is internally synchronized; we additionally only use it behind
// the Mutex in `jit()`.
unsafe impl Send for Jit {}

static JIT: OnceLock<Mutex<Jit>> = OnceLock::new();

/// SAFETY: `err` is a live LLVMErrorRef; this consumes it.
unsafe fn err_to_string(err: LLVMErrorRef) -> String {
    // SAFETY: the function's contract; the message is disposed once.
    unsafe {
        let msg = LLVMGetErrorMessage(err);
        let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(msg);
        s
    }
}

fn check(err: LLVMErrorRef, what: &str) -> Result<(), String> {
    if err.is_null() {
        Ok(())
    } else {
        // SAFETY: a non-null error returned by an LLVM call, consumed here.
        Err(format!("{what}: {}", unsafe { err_to_string(err) }))
    }
}

pub fn jit() -> &'static Mutex<Jit> {
    JIT.get_or_init(|| Mutex::new(Jit::new().expect("failed to create LLJIT")))
}

impl Jit {
    fn new() -> Result<Self, String> {
        crate::compile::init_llvm();
        // SAFETY: LLVM C API calls on handles created right here; every error is
        // checked before a handle is used.
        unsafe {
            let builder = LLVMOrcCreateLLJITBuilder();
            let mut jit: LLVMOrcLLJITRef = std::ptr::null_mut();
            check(LLVMOrcCreateLLJIT(&mut jit, builder), "create LLJIT")?;
            let jd = LLVMOrcLLJITGetMainJITDylib(jit);
            let prefix = LLVMOrcLLJITGetGlobalPrefix(jit);

            // libm etc. from the host process
            let mut gen1: LLVMOrcDefinitionGeneratorRef = std::ptr::null_mut();
            check(
                LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(&mut gen1, prefix, None, std::ptr::null_mut()),
                "process symbol generator",
            )?;
            LLVMOrcJITDylibAddGenerator(jd, gen1);

            // glibc's vector math library, for loops the vectorizer turned into _ZGV* calls
            let path = CString::new("libmvec.so.1").unwrap();
            let mut gen2: LLVMOrcDefinitionGeneratorRef = std::ptr::null_mut();
            let err = LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
                &mut gen2, path.as_ptr(), prefix, None, std::ptr::null_mut(),
            );
            if err.is_null() {
                LLVMOrcJITDylibAddGenerator(jd, gen2);
            } else {
                let _ = err_to_string(err);
            }

            // precompiled Rust scalar functions
            let es = LLVMOrcLLJITGetExecutionSession(jit);
            let mut pairs: Vec<LLVMOrcCSymbolMapPair> = native::scalar_symbols()
                .into_iter()
                .map(|(name, addr)| {
                    let c = CString::new(name).unwrap();
                    LLVMOrcCSymbolMapPair {
                        Name: LLVMOrcExecutionSessionIntern(es, c.as_ptr()),
                        Sym: LLVMJITEvaluatedSymbol {
                            Address: addr as u64,
                            Flags: LLVMJITSymbolFlags {
                                GenericFlags: (LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported as u8)
                                    | (LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8),
                                TargetFlags: 0,
                            },
                        },
                    }
                })
                .collect();
            let mu = LLVMOrcAbsoluteSymbols(pairs.as_mut_ptr(), pairs.len());
            check(LLVMOrcJITDylibDefine(jd, mu), "define Rust symbols")?;

            Ok(Jit { jit, jd, symbols: HashMap::new() })
        }
    }

    pub fn has(&self, name: &str) -> Option<usize> {
        self.symbols.get(name).copied()
    }

    /// Link an object file and return the address of `name`.
    pub fn add_object(&mut self, name: &str, obj: &[u8], label: &str) -> Result<usize, String> {
        if let Some(a) = self.symbols.get(name) {
            return Ok(*a);
        }
        // SAFETY: `self.jit`/`self.jd` are the live LLJIT handles (we hold &mut via the
        // Mutex); the object bytes are copied into an LLVM-owned buffer the JIT takes.
        unsafe {
            let bufname = CString::new(name).unwrap();
            let buf = llvm_sys::core::LLVMCreateMemoryBufferWithMemoryRangeCopy(
                obj.as_ptr() as *const _,
                obj.len(),
                bufname.as_ptr(),
            );
            check(LLVMOrcLLJITAddObjectFile(self.jit, self.jd, buf), "add object")?;
            let mut addr: LLVMOrcExecutorAddress = 0;
            check(LLVMOrcLLJITLookup(self.jit, &mut addr, bufname.as_ptr()), "lookup")?;
            self.symbols.insert(name.to_string(), addr as usize);
            crate::perf::register_object(obj, name, addr as usize, label);
            Ok(addr as usize)
        }
    }
}
