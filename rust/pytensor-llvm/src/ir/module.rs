//! Modules, functions and values; compiling a module.
//!
//! Thread safety: all LLVM state of a module (its context, the caches, its builders) is
//! in one `ModuleState` behind the module's mutex. The Python objects (`Module`,
//! `Function`, `Value`, `Builder`, ...) are plain data: an `Arc` of the module and
//! `Handle`s, raw LLVM handles that can only be read with the module's state locked.

use super::builder::{Builder, BuilderState};
use super::ty::{PyType, Ty};
use crate::target::{cache_dir, init_llvm, target_cpu, target_machine, vfabi_variants};
use llvm_sys::LLVMLinkage;
use llvm_sys::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_sys::core as c;
use llvm_sys::prelude::{LLVMContextRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef};
use llvm_sys::target::LLVMSetModuleDataLayout;
use llvm_sys::target_machine as tm;
use llvm_sys::transforms::pass_builder as pb;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use sha2::Digest;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Bump when the meaning of IR changes for the same text (runtime functions, ABI).
pub const IR_CACHE_VERSION: u32 = 1;

pub fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

pub fn cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "?")).unwrap()
}

/// A raw LLVM handle (value, type, block, ...) of the module `module`.
///
/// Handles are what the Python objects hold. The raw handle can only be read with
/// `get`, which takes the module's (locked) state and checks that it is the right one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Handle<T> {
    raw: T,
    module: u64,
}

// SAFETY: a handle is an identifier. Its raw handle reaches LLVM only through `get`,
// which needs its module's state, i.e. the module's lock held: LLVM objects of a module
// are only touched by one thread at a time.
unsafe impl<T> Send for Handle<T> {}
// SAFETY: as for `Send`.
unsafe impl<T> Sync for Handle<T> {}

impl<T: Copy> Handle<T> {
    /// The raw handle; `what` names it in the error for a handle of another module.
    pub fn get(&self, st: &ModuleState, what: &str) -> PyResult<T> {
        if self.module != st.id {
            return err(format!("{what} from a different module"));
        }
        Ok(self.raw)
    }
}

/// A module: its LLVM state behind a lock.
pub struct Shared {
    state: Mutex<ModuleState>,
}

impl Shared {
    /// The module's state, locked. Waiting for the lock detaches from Python, so a
    /// thread holding it while it runs Python code can't deadlock with this one.
    pub fn lock(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, ModuleState>> {
        self.state
            .lock_py_attached(py)
            .map_err(|_| PyRuntimeError::new_err("an earlier panic left the module broken"))
    }
}

/// An LLVM context and module, and everything built in it.
pub struct ModuleState {
    /// `Shared::id`: identifies the module's handles
    pub id: u64,
    pub ctx: LLVMContextRef,
    pub module: LLVMModuleRef,
    pub name: String,
    pub fastmath: bool,
    /// set once compiled: no more building
    pub finished: Cell<bool>,
    /// libmvec declarations that must survive until the vectorizer ran
    pub vector_decls: RefCell<Vec<LLVMValueRef>>,
    types: RefCell<HashMap<Ty, LLVMTypeRef>>,
    /// debug information, if asked for (`debug.rs`)
    pub debug: Option<super::debug::DebugInfo>,
    functions: RefCell<Vec<(String, LLVMValueRef, bool)>>,
    /// a readable name for the generated code in perf maps (not part of the IR, so not
    /// of the cache key)
    label: RefCell<Option<String>>,
    /// the builders of the module's functions (`Builder` holds the index)
    pub builders: RefCell<Vec<BuilderState>>,
}

// SAFETY: the state owns its LLVM context and everything in it, and LLVM contexts aren't
// tied to a thread: the state may move to another thread (it is only reached through the
// module's mutex, so one thread uses it at a time).
unsafe impl Send for ModuleState {}

impl ModuleState {
    pub fn llvm_type(&self, t: &Ty) -> LLVMTypeRef {
        if let Some(&r) = self.types.borrow().get(t) {
            return r;
        }
        let r = t.to_llvm(self.ctx);
        self.types.borrow_mut().insert(t.clone(), r);
        r
    }

    pub fn check_open(&self) -> PyResult<()> {
        if self.finished.get() {
            return err(format!("module {} was already compiled", self.name));
        }
        Ok(())
    }

    /// A handle of this module.
    pub fn handle<T>(&self, raw: T) -> Handle<T> {
        Handle {
            raw,
            module: self.id,
        }
    }
}

impl Drop for ModuleState {
    fn drop(&mut self) {
        // the builders, then the DIBuilder, refer to the module
        self.builders.get_mut().clear();
        drop(self.debug.take());
        // SAFETY: we own the module and its context; nothing refers to them anymore
        // (the Python objects hold an `Arc` of `Shared`, which owns this).
        unsafe {
            c::LLVMDisposeModule(self.module);
            c::LLVMContextDispose(self.ctx);
        }
    }
}

/// A value in a module: a constant, a function parameter, or an instruction result.
#[pyclass(module = "pytensor_llvm.ir", frozen, from_py_object)]
#[derive(Clone)]
pub struct Value {
    pub m: Arc<Shared>,
    pub raw: Handle<LLVMValueRef>,
    pub ty: Ty,
    /// the function this value lives in (None: constants and globals)
    pub func: Option<Handle<LLVMValueRef>>,
}

impl Value {
    /// The value if this is an integer constant (booleans: 0 or 1).
    pub fn const_int_value(&self, st: &ModuleState) -> PyResult<Option<i64>> {
        let raw = self.raw.get(st, "value")?;
        // SAFETY: `raw` is a live value of this (locked) module.
        unsafe {
            if self.ty.is_int() && !c::LLVMIsAConstantInt(raw).is_null() {
                if self.ty.is_bool() {
                    return Ok(Some(c::LLVMConstIntGetZExtValue(raw) as i64));
                }
                return Ok(Some(c::LLVMConstIntGetSExtValue(raw)));
            }
        }
        Ok(None)
    }

    /// `f(state, raw value)` with the module locked.
    fn with<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&ModuleState, LLVMValueRef) -> R,
    ) -> PyResult<R> {
        let st = self.m.lock(py)?;
        let raw = self.raw.get(&st, "value")?;
        Ok(f(&st, raw))
    }
}

#[pymethods]
impl Value {
    #[getter]
    fn r#type(&self) -> PyType {
        PyType(self.ty.clone())
    }

    /// The value if this is an integer constant.
    #[getter]
    fn const_int(&self, py: Python<'_>) -> PyResult<Option<i64>> {
        self.const_int_value(&*self.m.lock(py)?)
    }

    /// Name the value in the IR text (`%name` instead of `%7`), for reading dumps.
    /// Returns the value. Constants have no names.
    fn named(slf: Py<Self>, py: Python<'_>, name: &str) -> PyResult<Py<Self>> {
        slf.get().with(py, |_, raw| {
            // SAFETY: `raw` is a live value of the locked module; LLVM copies the name.
            unsafe {
                if c::LLVMIsConstant(raw) == 0 {
                    c::LLVMSetValueName2(raw, name.as_ptr() as *const _, name.len());
                }
            }
        })?;
        Ok(slf)
    }

    /// The value if this is a float constant.
    #[getter]
    fn const_float(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        let is_float = self.ty.is_float();
        // SAFETY: `raw` is a live value of the locked module.
        self.with(py, |_, raw| unsafe {
            if is_float && !c::LLVMIsAConstantFP(raw).is_null() {
                let mut lost = 0;
                return Some(c::LLVMConstRealGetDouble(raw, &mut lost));
            }
            None
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // SAFETY: `raw` is a live value of the locked module; the message is freed.
        self.with(py, |_, raw| unsafe {
            let s = c::LLVMPrintValueToString(raw);
            let r = CStr::from_ptr(s).to_string_lossy().trim().to_string();
            c::LLVMDisposeMessage(s);
            format!("<ir.Value {r}>")
        })
    }
}

#[pyclass(module = "pytensor_llvm.ir", eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Inline {
    Default,
    Always,
    Never,
}

/// A function of a module.
#[pyclass(module = "pytensor_llvm.ir", frozen)]
pub struct Function {
    pub m: Arc<Shared>,
    pub raw: Handle<LLVMValueRef>,
    pub fnty: Handle<LLVMTypeRef>,
    pub name: String,
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub exported: bool,
    has_builder: AtomicBool,
}

#[pymethods]
impl Function {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<Vec<Value>> {
        let st = self.m.lock(py)?;
        let f = self.raw.get(&st, "function")?;
        Ok((0..self.params.len())
            .map(|k| Value {
                m: self.m.clone(),
                // SAFETY: `f` is a function of the locked module with this many
                // parameters.
                raw: st.handle(unsafe { c::LLVMGetParam(f, k as u32) }),
                ty: self.params[k].clone(),
                func: Some(self.raw),
            })
            .collect())
    }

    #[getter]
    fn ret(&self) -> PyType {
        PyType(self.ret.clone())
    }

    /// A builder for the body. A function has one.
    fn builder(&self, py: Python<'_>) -> PyResult<Builder> {
        let st = self.m.lock(py)?;
        st.check_open()?;
        let f = self.raw.get(&st, "function")?;
        if self.has_builder.swap(true, Ordering::Relaxed) {
            return err(format!("function {} already has a builder", self.name));
        }
        let state = BuilderState::new(&st, f, self.ret.clone());
        let mut builders = st.builders.borrow_mut();
        builders.push(state);
        Ok(Builder::new(self.m.clone(), builders.len() - 1))
    }
}

impl Function {
    pub fn signature_matches(&self, args: &[Value]) -> PyResult<()> {
        if args.len() != self.params.len() {
            return err(format!(
                "{} takes {} arguments, got {}",
                self.name,
                self.params.len(),
                args.len()
            ));
        }
        for (k, (a, t)) in args.iter().zip(&self.params).enumerate() {
            if &a.ty != t {
                return err(format!(
                    "argument {k} of {}: expected {t}, got {}",
                    self.name, a.ty
                ));
            }
        }
        Ok(())
    }
}

/// An LLVM module under construction.
#[pyclass(module = "pytensor_llvm.ir", frozen)]
pub struct Module {
    pub m: Arc<Shared>,
}

#[pymethods]
impl Module {
    /// `debug_info`: give every instruction a debug location, the Python code that
    /// emitted it (see `debug.rs`); functions are named by their `debug_name`.
    #[new]
    #[pyo3(signature = (name, fastmath = true, debug_info = false))]
    fn new(name: &str, fastmath: bool, debug_info: bool) -> Self {
        static IDS: AtomicU64 = AtomicU64::new(0);
        init_llvm();
        // SAFETY: creating a fresh context and a module in it.
        let (ctx, module) = unsafe {
            let ctx = c::LLVMContextCreate();
            let n = cstring(name);
            (ctx, c::LLVMModuleCreateWithNameInContext(n.as_ptr(), ctx))
        };
        let id = IDS.fetch_add(1, Ordering::Relaxed);
        let state = ModuleState {
            id,
            ctx,
            module,
            name: name.to_string(),
            fastmath,
            finished: Cell::new(false),
            vector_decls: RefCell::new(vec![]),
            types: RefCell::new(HashMap::new()),
            debug: debug_info.then(|| super::debug::DebugInfo::new(ctx, module, name)),
            functions: RefCell::new(vec![]),
            label: RefCell::new(None),
            builders: RefCell::new(vec![]),
        };
        Module {
            m: Arc::new(Shared {
                state: Mutex::new(state),
            }),
        }
    }

    #[getter]
    fn name(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.m.lock(py)?.name.clone())
    }

    /// Readable name of the code in perf maps (default: the module name).
    #[getter]
    fn get_label(&self, py: Python<'_>) -> PyResult<String> {
        Ok(label_of(&*self.m.lock(py)?))
    }

    #[setter]
    fn set_label(&self, py: Python<'_>, label: String) -> PyResult<()> {
        *self.m.lock(py)?.label.borrow_mut() = Some(label);
        Ok(())
    }

    /// Add a function. `exported` functions keep a symbol after compilation (their
    /// address can be looked up); the others are internal. `debug_name`: what the
    /// function is called in debug information (default: `name`).
    #[pyo3(signature = (name, params, ret, exported = false, inline = Inline::Default, debug_name = None))]
    fn function(
        &self,
        py: Python<'_>,
        name: &str,
        params: Vec<PyType>,
        ret: PyType,
        exported: bool,
        inline: Inline,
        debug_name: Option<&str>,
    ) -> PyResult<Function> {
        let st = self.m.lock(py)?;
        st.check_open()?;
        if st.functions.borrow().iter().any(|(n, _, _)| n == name) {
            return err(format!("function {name} already exists"));
        }
        if name.is_empty()
            || !name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return err(format!("bad function name {name:?}"));
        }
        let params: Vec<Ty> = params.into_iter().map(|t| t.0).collect();
        if let Some(p) = params.iter().find(|t| !t.is_first_class()) {
            return err(format!("parameter of type {p}"));
        }
        let ret = ret.0;
        // SAFETY: the types belong to this module's context; the function is added to
        // this module.
        let (raw, fnty) = unsafe {
            let mut ps: Vec<LLVMTypeRef> = params.iter().map(|t| st.llvm_type(t)).collect();
            let fnty = c::LLVMFunctionType(st.llvm_type(&ret), ps.as_mut_ptr(), ps.len() as u32, 0);
            let n = cstring(name);
            let f = c::LLVMAddFunction(st.module, n.as_ptr(), fnty);
            c::LLVMSetLinkage(
                f,
                if exported {
                    LLVMLinkage::LLVMExternalLinkage
                } else {
                    LLVMLinkage::LLVMInternalLinkage
                },
            );
            match inline {
                Inline::Always => add_fn_attr(&st, f, "alwaysinline"),
                Inline::Never => add_fn_attr(&st, f, "noinline"),
                Inline::Default => {}
            }
            (f, fnty)
        };
        if let Some(d) = &st.debug {
            d.add_function(raw, debug_name.unwrap_or(name), name);
        }
        st.functions
            .borrow_mut()
            .push((name.to_string(), raw, exported));
        Ok(Function {
            m: self.m.clone(),
            raw: st.handle(raw),
            fnty: st.handle(fnty),
            name: name.to_string(),
            params,
            ret,
            exported,
            has_builder: AtomicBool::new(false),
        })
    }

    /// The module as LLVM IR text.
    fn ir(&self, py: Python<'_>) -> PyResult<String> {
        Ok(print_module(self.m.lock(py)?.module))
    }

    /// Check the module; raises with LLVM's message if it is invalid.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        verify(&*self.m.lock(py)?)
    }

    /// Verify, optimize, generate machine code and link it into the JIT. The object is
    /// cached by a hash of the IR and the target. Afterwards the module can't change.
    #[pyo3(signature = (opt_level = 3, cache = true))]
    fn compile(&self, py: Python<'_>, opt_level: u32, cache: bool) -> PyResult<CompiledModule> {
        let mut r = compile_modules(py, &[self], opt_level, cache)?;
        Ok(r.remove(0))
    }
}

fn label_of(st: &ModuleState) -> String {
    st.label.borrow().clone().unwrap_or_else(|| st.name.clone())
}

impl Module {
    /// Finish the module for compilation: verify, hash, give exported functions unique
    /// symbol names.
    fn prepare(&self, py: Python<'_>, opt_level: u32) -> PyResult<Prepared> {
        let st = self.m.lock(py)?;
        st.check_open()?;
        for (name, f, _) in st.functions.borrow().iter() {
            // SAFETY: `f` is a function of this module.
            if unsafe { c::LLVMCountBasicBlocks(*f) } == 0 {
                return err(format!("function {name} has no body"));
            }
        }
        finish_vector_decls(&st);
        if crate::perf::enabled() {
            for (_, f, _) in st.functions.borrow().iter() {
                add_fn_string_attr(&st, *f, "frame-pointer", "all");
            }
        }
        verify(&st)?;
        st.finished.set(true);
        let text = print_module(st.module);
        let mut h = sha2::Sha256::new();
        h.update(salt(opt_level).as_bytes());
        h.update(text.as_bytes());
        let key = hex::encode(&h.finalize()[..16]);
        let exported: Vec<(String, LLVMValueRef)> = st
            .functions
            .borrow()
            .iter()
            .filter(|(_, _, e)| *e)
            .map(|(n, f, _)| (n.clone(), *f))
            .collect();
        if exported.is_empty() {
            return err(format!("module {} exports no function", st.name));
        }
        let mut symbols = Vec::new();
        for (name, f) in &exported {
            let sym = format!("pt{key}_{name}");
            let s = cstring(&sym);
            // SAFETY: renaming a function of this (now immutable) module.
            unsafe { c::LLVMSetValueName2(*f, s.as_ptr(), sym.len()) };
            symbols.push((name.clone(), sym));
        }
        Ok(Prepared {
            key,
            symbols,
            name: label_of(&st),
            ir: text,
        })
    }
}

struct Prepared {
    key: String,
    symbols: Vec<(String, String)>,
    name: String,
    /// the IR before optimization
    ir: String,
}

/// Compile several modules: prepared one by one, optimized and code-generated in
/// parallel (without the GIL), linked one by one.
pub fn compile_modules(
    py: Python<'_>,
    modules: &[&Module],
    opt_level: u32,
    cache: bool,
) -> PyResult<Vec<CompiledModule>> {
    use rayon::prelude::*;
    let t0 = std::time::Instant::now();
    let prepared: Vec<Prepared> = modules
        .iter()
        .map(|m| m.prepare(py, opt_level))
        .collect::<PyResult<_>>()?;
    let in_process: Vec<bool> = {
        let j = crate::jit::jit().lock().unwrap_or_else(|e| e.into_inner());
        prepared
            .iter()
            .map(|p| j.has(&p.symbols[0].1).is_some())
            .collect()
    };
    let dir = cache_dir();
    if cache {
        let _ = std::fs::create_dir_all(&dir);
    }
    // the same module twice (same key) is compiled once
    let mut first_of: HashMap<&str, usize> = HashMap::new();
    let todo: Vec<usize> = (0..prepared.len())
        .filter(|&i| !in_process[i] && *first_of.entry(prepared[i].key.as_str()).or_insert(i) == i)
        .collect();
    // the workers lock the modules they compile (plainly: they don't run Python code)
    let jobs: Vec<(usize, &Shared, String)> = todo
        .iter()
        .map(|&i| (i, &*modules[i].m, prepared[i].key.clone()))
        .collect();
    let results: Vec<(usize, Result<(Vec<u8>, bool, f64), String>)> = py.detach(move || {
        jobs.into_par_iter()
            .map(|(i, m, key)| {
                let path = dir.join(format!("ir_{key}.o"));
                if cache {
                    if let Ok(o) = std::fs::read(&path) {
                        return (i, Ok((o, true, 0.0)));
                    }
                }
                let t = std::time::Instant::now();
                let r = match m.state.lock() {
                    Ok(st) => codegen(st.module, opt_level),
                    Err(_) => Err("an earlier panic left the module broken".to_string()),
                };
                let r = r.map(|obj| {
                    if cache {
                        let tmp = dir.join(format!("ir_{key}.{}.{i}.tmp", std::process::id()));
                        if std::fs::write(&tmp, &obj).is_ok() {
                            let _ = std::fs::rename(&tmp, &path);
                        }
                    }
                    (obj, false, t.elapsed().as_secs_f64())
                });
                (i, r)
            })
            .collect()
    });
    let mut stats: Vec<CompileStats> = in_process
        .iter()
        .map(|&p| CompileStats {
            in_process: p,
            ..Default::default()
        })
        .collect();
    let mut j = crate::jit::jit().lock().unwrap_or_else(|e| e.into_inner());
    for (i, r) in results {
        let (obj, disk, secs) = r.map_err(PyRuntimeError::new_err)?;
        stats[i].disk_cache = disk;
        stats[i].compile_s = secs;
        let p = &prepared[i];
        j.add_object(&p.symbols[0].1, &obj, &p.name)
            .map_err(PyRuntimeError::new_err)?;
    }
    let total = t0.elapsed().as_secs_f64();
    let mut out = Vec::with_capacity(prepared.len());
    for (i, p) in prepared.into_iter().enumerate() {
        let mut addresses = HashMap::new();
        for (name, sym) in p.symbols {
            addresses.insert(name, j.lookup(&sym).map_err(PyRuntimeError::new_err)?);
        }
        let mut st = stats[i].clone();
        st.total_s = total;
        out.push(CompiledModule {
            key: p.key,
            addresses,
            stats: st,
            ir: p.ir,
            opt_level,
        });
    }
    Ok(out)
}

/// Compile several modules in parallel (see `Module.compile`).
#[pyfunction]
#[pyo3(signature = (modules, opt_level = 3, cache = true))]
pub fn compile_many(
    py: Python<'_>,
    modules: Vec<PyRef<'_, Module>>,
    opt_level: u32,
    cache: bool,
) -> PyResult<Vec<CompiledModule>> {
    let refs: Vec<&Module> = modules.iter().map(|m| &**m).collect();
    compile_modules(py, &refs, opt_level, cache)
}

#[derive(Clone, Default)]
pub struct CompileStats {
    pub in_process: bool,
    pub disk_cache: bool,
    pub compile_s: f64,
    pub total_s: f64,
}

/// A compiled module: addresses of its exported functions.
#[pyclass(module = "pytensor_llvm.ir", frozen)]
pub struct CompiledModule {
    #[pyo3(get)]
    key: String,
    addresses: HashMap<String, usize>,
    stats: CompileStats,
    ir: String,
    opt_level: u32,
}

#[pymethods]
impl CompiledModule {
    fn address(&self, name: &str) -> PyResult<usize> {
        self.addresses
            .get(name)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("no exported function {name}")))
    }

    /// The module's LLVM IR as it was compiled, before optimization.
    #[getter]
    fn ir(&self) -> &str {
        &self.ir
    }

    /// The IR after the optimization pipeline the module was compiled with (or
    /// `opt_level`). Computed on demand from `ir`, so it is the same whether the object
    /// was compiled, read from the disk cache or already in the process. Exported
    /// functions keep their plain names here (compiled, they get a unique symbol).
    ///
    /// `remarks`: start with LLVM's optimization remarks (of every pass: what the loop
    /// vectorizer did or why it didn't, what was inlined, unrolled...), as comments.
    /// Each names the instruction of `ir` it is about (e.g. the branch into a loop).
    #[pyo3(signature = (opt_level = None, remarks = false))]
    fn optimized_ir(
        &self,
        py: Python<'_>,
        opt_level: Option<u32>,
        remarks: bool,
    ) -> PyResult<String> {
        let level = opt_level.unwrap_or(self.opt_level);
        py.detach(|| optimized_text(&self.ir, level, remarks))
            .map_err(PyRuntimeError::new_err)
    }

    /// How the object was obtained: "compiled", "disk cache" or "in process".
    #[getter]
    fn source(&self) -> &'static str {
        if self.stats.in_process {
            "in process"
        } else if self.stats.disk_cache {
            "disk cache"
        } else {
            "compiled"
        }
    }

    #[getter]
    fn compile_s(&self) -> f64 {
        self.stats.compile_s
    }

    #[getter]
    fn total_s(&self) -> f64 {
        self.stats.total_s
    }
}

// ------------------------------------------------------------------ helpers

pub fn add_fn_attr(m: &ModuleState, f: LLVMValueRef, name: &str) {
    // SAFETY: `f` is a function of `m`; enum attribute kinds are looked up by name.
    unsafe {
        let kind = c::LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
        let a = c::LLVMCreateEnumAttribute(m.ctx, kind, 0);
        c::LLVMAddAttributeAtIndex(f, llvm_sys::LLVMAttributeFunctionIndex, a);
    }
}

pub fn add_fn_string_attr(m: &ModuleState, f: LLVMValueRef, k: &str, v: &str) {
    // SAFETY: as above; the attribute copies the strings.
    unsafe {
        let a = c::LLVMCreateStringAttribute(
            m.ctx,
            k.as_ptr() as *const _,
            k.len() as u32,
            v.as_ptr() as *const _,
            v.len() as u32,
        );
        c::LLVMAddAttributeAtIndex(f, llvm_sys::LLVMAttributeFunctionIndex, a);
    }
}

fn print_module(m: LLVMModuleRef) -> String {
    // SAFETY: printing a live module; the message is freed.
    unsafe {
        let s = c::LLVMPrintModuleToString(m);
        let r = CStr::from_ptr(s).to_string_lossy().into_owned();
        c::LLVMDisposeMessage(s);
        r
    }
}

fn verify(m: &ModuleState) -> PyResult<()> {
    if let Some(d) = &m.debug {
        d.finalize(m);
    }
    // SAFETY: verifying a live module; the message is freed.
    unsafe {
        let mut msg = std::ptr::null_mut();
        let bad = LLVMVerifyModule(
            m.module,
            LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &mut msg,
        );
        let text = if msg.is_null() {
            String::new()
        } else {
            CStr::from_ptr(msg).to_string_lossy().into_owned()
        };
        if !msg.is_null() {
            c::LLVMDisposeMessage(msg);
        }
        if bad != 0 {
            return err(format!("invalid IR in module {}: {text}", m.name));
        }
    }
    Ok(())
}

/// Keep libmvec declarations alive until the loop vectorizer has seen them.
fn finish_vector_decls(m: &ModuleState) {
    let decls = m.vector_decls.borrow();
    if decls.is_empty() {
        return;
    }
    // SAFETY: building a global in this module from its own function declarations.
    unsafe {
        let p = m.llvm_type(&Ty::Ptr);
        let mut vals = decls.clone();
        let arr = c::LLVMConstArray2(p, vals.as_mut_ptr(), vals.len() as u64);
        let name = cstring("llvm.compiler.used");
        let g = c::LLVMAddGlobal(m.module, c::LLVMTypeOf(arr), name.as_ptr());
        c::LLVMSetLinkage(g, LLVMLinkage::LLVMAppendingLinkage);
        let sec = cstring("llvm.metadata");
        c::LLVMSetSection(g, sec.as_ptr());
        c::LLVMSetInitializer(g, arr);
    }
}

fn salt(opt_level: u32) -> String {
    let (mut ma, mut mi, mut pa) = (0u32, 0u32, 0u32);
    // SAFETY: plain out-parameters.
    unsafe { c::LLVMGetVersion(&mut ma, &mut mi, &mut pa) };
    let (cpu, features) = target_cpu();
    format!(
        "ir{IR_CACHE_VERSION}|{}|llvm{ma}.{mi}.{pa}|{cpu}|{features}|O{opt_level}|veclib{:?}|fp{}",
        env!("CARGO_PKG_VERSION"),
        vfabi_variants(),
        crate::perf::enabled(),
    )
}

/// Compilations hold this shared. Collecting optimization remarks holds it alone: that
/// switches on LLVM's process-wide `-pass-remarks*` options, and a compilation running
/// meanwhile would print its remarks to stderr.
static REMARKS: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Set LLVM's `-pass-remarks`, `-pass-remarks-missed` and `-pass-remarks-analysis` to
/// `pattern` (passes whose remarks are produced).
fn set_remark_passes(pattern: &str) {
    let args: Vec<CString> = [
        "pytensor-llvm",
        "-pass-remarks",
        "-pass-remarks-missed",
        "-pass-remarks-analysis",
    ]
    .iter()
    .enumerate()
    .map(|(k, a)| {
        cstring(&if k == 0 {
            a.to_string()
        } else {
            format!("{a}={pattern}")
        })
    })
    .collect();
    let ptrs: Vec<*const std::ffi::c_char> = args.iter().map(|a| a.as_ptr()).collect();
    // SAFETY: NUL-terminated arguments that outlive the call.
    unsafe {
        llvm_sys::support::LLVMParseCommandLineOptions(
            ptrs.len() as i32,
            ptrs.as_ptr(),
            c"".as_ptr(),
        )
    };
}

/// Diagnostic handler of the context remarks are collected in: `out` is a `Vec<String>`.
extern "C" fn collect_diagnostic(
    d: llvm_sys::prelude::LLVMDiagnosticInfoRef,
    out: *mut std::ffi::c_void,
) {
    // SAFETY: `out` is the vector `optimized_text` set up for this context; the
    // description is freed.
    unsafe {
        let s = c::LLVMGetDiagInfoDescription(d);
        (*(out as *mut Vec<String>)).push(CStr::from_ptr(s).to_string_lossy().trim().to_string());
        c::LLVMDisposeMessage(s);
    }
}

/// "@function %block: instruction" for every instruction of `m`, in the order the
/// `debugify` pass numbers them (its line numbers, from 1).
///
/// # Safety
/// `m` is a live module used only by this thread.
unsafe fn instruction_lines(m: LLVMModuleRef) -> Vec<String> {
    let name = |v: LLVMValueRef| {
        let mut n = 0;
        // SAFETY: names of values of `m`, copied.
        String::from_utf8_lossy(unsafe {
            std::slice::from_raw_parts(c::LLVMGetValueName2(v, &mut n) as *const u8, n)
        })
        .into_owned()
    };
    let mut lines = Vec::new();
    // SAFETY: walking the functions, blocks and instructions of `m`; printed values are
    // freed.
    unsafe {
        let mut f = c::LLVMGetFirstFunction(m);
        while !f.is_null() {
            let mut bb = c::LLVMGetFirstBasicBlock(f);
            while !bb.is_null() {
                let block = name(c::LLVMBasicBlockAsValue(bb));
                let mut i = c::LLVMGetFirstInstruction(bb);
                while !i.is_null() {
                    let s = c::LLVMPrintValueToString(i);
                    let text = CStr::from_ptr(s).to_string_lossy().trim().to_string();
                    c::LLVMDisposeMessage(s);
                    let text: String = text
                        .split(", !dbg")
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(80)
                        .collect();
                    let loc = llvm_sys::debuginfo::LLVMInstructionGetDebugLoc(i);
                    let at = if loc.is_null() {
                        String::new()
                    } else {
                        let ctx = c::LLVMGetModuleContext(m);
                        format!("\n;     at {}", super::debug::describe(ctx, loc))
                    };
                    lines.push(format!("@{} %{block}: {text}{at}", name(f)));
                    i = c::LLVMGetNextInstruction(i);
                }
                bb = c::LLVMGetNextBasicBlock(bb);
            }
            f = c::LLVMGetNextFunction(f);
        }
    }
    lines
}

/// A remark ("ir:<line>:<col>: <message>", the line from `debugify`) with the
/// instruction it is about.
fn located(remark: &str, lines: &[String]) -> String {
    let mut parts = remark.splitn(4, ':');
    if let (Some("ir"), Some(line), Some(_col), Some(msg)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
        && let Some(at) = line
            .parse::<usize>()
            .ok()
            .and_then(|l| lines.get(l.wrapping_sub(1)))
    {
        return format!("{at}\n;     {}", msg.trim());
    }
    remark.to_string()
}

/// Set the target of `m` and run the optimization pipeline of `opt_level` on it.
fn optimize(m: LLVMModuleRef, tmr: tm::LLVMTargetMachineRef, opt_level: u32) -> Result<(), String> {
    run_pipeline(m, tmr, opt_level, false)
}

/// `optimize`, after `debugify` (which gives every instruction a line number: its
/// position in the module) if `debugify`.
fn run_pipeline(
    m: LLVMModuleRef,
    tmr: tm::LLVMTargetMachineRef,
    opt_level: u32,
    debugify: bool,
) -> Result<(), String> {
    // SAFETY: `m` is a module used only by this thread; `tmr` is a live target machine;
    // messages are freed.
    unsafe {
        let triple = tm::LLVMGetTargetMachineTriple(tmr);
        c::LLVMSetTarget(m, triple);
        c::LLVMDisposeMessage(triple);
        let dl = tm::LLVMCreateTargetDataLayout(tmr);
        LLVMSetModuleDataLayout(m, dl);
        llvm_sys::target::LLVMDisposeTargetData(dl);

        let opts = pb::LLVMCreatePassBuilderOptions();
        pb::LLVMPassBuilderOptionsSetLoopVectorization(opts, 1);
        pb::LLVMPassBuilderOptionsSetSLPVectorization(opts, 1);
        pb::LLVMPassBuilderOptionsSetLoopInterleaving(opts, 1);
        pb::LLVMPassBuilderOptionsSetLoopUnrolling(opts, 1);
        let passes = format!("default<O{}>", opt_level.min(3));
        let passes = cstring(&if debugify {
            format!("debugify,{passes}")
        } else {
            passes
        });
        let e = pb::LLVMRunPasses(m, passes.as_ptr(), tmr, opts);
        pb::LLVMDisposePassBuilderOptions(opts);
        if !e.is_null() {
            let msg = llvm_sys::error::LLVMGetErrorMessage(e);
            let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
            llvm_sys::error::LLVMDisposeErrorMessage(msg);
            return Err(format!("optimization failed: {s}"));
        }
    }
    Ok(())
}

/// The IR of a module as the compiler sees it after optimization: `ir` (as printed
/// before compiling) parsed into a fresh context and run through the same pipeline.
/// With `remarks`, LLVM's optimization remarks first, as comments (see `optimized_ir`).
fn optimized_text(ir: &str, opt_level: u32, remarks: bool) -> Result<String, String> {
    init_llvm();
    let machine = target_machine(opt_level)?;
    let tmr = machine.as_mut_ptr();
    let mut text = String::new();
    if remarks {
        text.push_str("; optimization remarks, each after the instruction of the IR before\n");
        text.push_str("; optimization it is about\n");
        for remark in with_module(ir, |ctx, m| collect_remarks(ctx, m, tmr, opt_level))? {
            text.push_str(&format!("; {remark}\n"));
        }
        text.push('\n');
    }
    with_module(ir, |_, m| {
        let _shared = REMARKS.read().unwrap_or_else(|e| e.into_inner());
        optimize(m, tmr, opt_level).map(|()| text + &print_module(m))
    })
}

/// `f(ctx, module)` with `ir` parsed into a fresh context, disposed after.
fn with_module<T>(
    ir: &str,
    f: impl FnOnce(LLVMContextRef, LLVMModuleRef) -> Result<T, String>,
) -> Result<T, String> {
    let text = cstring(ir);
    let name = cstring("ir");
    // SAFETY: a private context and module, disposed on every path; parsing consumes the
    // memory buffer; messages are freed.
    unsafe {
        let ctx = c::LLVMContextCreate();
        let buf = c::LLVMCreateMemoryBufferWithMemoryRangeCopy(
            text.as_ptr(),
            text.as_bytes().len(),
            name.as_ptr(),
        );
        let mut module = std::ptr::null_mut();
        let mut msg = std::ptr::null_mut();
        if llvm_sys::ir_reader::LLVMParseIRInContext(ctx, buf, &mut module, &mut msg) != 0 {
            let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
            c::LLVMDisposeMessage(msg);
            c::LLVMContextDispose(ctx);
            return Err(format!("can't parse the IR: {s}"));
        }
        let r = f(ctx, module);
        c::LLVMDisposeModule(module);
        c::LLVMContextDispose(ctx);
        r
    }
}

/// The optimization remarks of running the pipeline on `m` (in `ctx`, used for nothing
/// else), each with the instruction it is about.
fn collect_remarks(
    ctx: LLVMContextRef,
    m: LLVMModuleRef,
    tmr: tm::LLVMTargetMachineRef,
    opt_level: u32,
) -> Result<Vec<String>, String> {
    // SAFETY: `m` is a live module of `ctx`, both only used here. (`debugify` only
    // numbers the instructions of a module without debug information.)
    let lines = unsafe {
        let lines = instruction_lines(m);
        llvm_sys::debuginfo::LLVMStripModuleDebugInfo(m);
        lines
    };
    let mut collected: Vec<String> = Vec::new();
    // SAFETY: `collected` outlives the pipeline run, the only use of the handler.
    unsafe {
        c::LLVMContextSetDiagnosticHandler(
            ctx,
            Some(collect_diagnostic),
            &mut collected as *mut Vec<String> as *mut std::ffi::c_void,
        )
    };
    {
        let _alone = REMARKS.write().unwrap_or_else(|e| e.into_inner());
        set_remark_passes(".*");
        let r = run_pipeline(m, tmr, opt_level, true);
        // a pattern nothing matches: off again
        set_remark_passes("a^");
        r?;
    }
    Ok(collected.iter().map(|r| located(r, &lines)).collect())
}

/// Optimize a module and generate an object file.
fn codegen(m: LLVMModuleRef, opt_level: u32) -> Result<Vec<u8>, String> {
    let _shared = REMARKS.read().unwrap_or_else(|e| e.into_inner());
    let machine = target_machine(opt_level)?;
    let tmr = machine.as_mut_ptr();
    optimize(m, tmr, opt_level)?;
    // SAFETY: `m` is a finished module used only by this thread; `tmr` is a live target
    // machine; buffers and messages are freed.
    unsafe {
        if let Ok(dir) = std::env::var("PYTENSOR_LLVM_DUMP_IR") {
            let _ = std::fs::create_dir_all(&dir);
            let name = CStr::from_ptr(c::LLVMGetModuleIdentifier(m, &mut 0))
                .to_string_lossy()
                .into_owned();
            let _ = std::fs::write(
                std::path::Path::new(&dir).join(format!("{name}.opt.ll")),
                print_module(m),
            );
        }
        let mut msg = std::ptr::null_mut();
        let mut buf = std::ptr::null_mut();
        let bad = tm::LLVMTargetMachineEmitToMemoryBuffer(
            tmr,
            m,
            tm::LLVMCodeGenFileType::LLVMObjectFile,
            &mut msg,
            &mut buf,
        );
        if bad != 0 {
            let s = CStr::from_ptr(msg).to_string_lossy().into_owned();
            c::LLVMDisposeMessage(msg);
            return Err(format!("code generation failed: {s}"));
        }
        let start = c::LLVMGetBufferStart(buf) as *const u8;
        let len = c::LLVMGetBufferSize(buf);
        let obj = std::slice::from_raw_parts(start, len).to_vec();
        c::LLVMDisposeMemoryBuffer(buf);
        Ok(obj)
    }
}

/// glibc libmvec variants to advertise (ISA letter, f64 lanes).
pub fn vector_variants() -> Vec<(char, u32)> {
    vfabi_variants()
}
