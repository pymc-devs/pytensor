//! Modules, functions and values; compiling a module.

use super::builder::Builder;
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
use sha2::Digest;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::rc::Rc;

/// Bump when the meaning of IR changes for the same text (runtime functions, ABI).
pub const IR_CACHE_VERSION: u32 = 1;

pub fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

pub fn cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "?")).unwrap()
}

/// An LLVM context and module. Everything built in a module refers to it.
pub struct ModuleInner {
    pub ctx: LLVMContextRef,
    pub module: LLVMModuleRef,
    pub name: String,
    pub fastmath: bool,
    /// set once compiled: no more building
    pub finished: Cell<bool>,
    /// libmvec declarations that must survive until the vectorizer ran
    pub vector_decls: RefCell<Vec<LLVMValueRef>>,
    types: RefCell<HashMap<Ty, LLVMTypeRef>>,
}

impl ModuleInner {
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
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        // SAFETY: we own the module and its context; nothing refers to them anymore
        // (every Value/Function/Builder holds an Rc to this).
        unsafe {
            c::LLVMDisposeModule(self.module);
            c::LLVMContextDispose(self.ctx);
        }
    }
}

/// A value in a module: a constant, a function parameter, or an instruction result.
#[pyclass(module = "pytensor_llvm.ir", unsendable, from_py_object)]
#[derive(Clone)]
pub struct Value {
    pub m: Rc<ModuleInner>,
    pub raw: LLVMValueRef,
    pub ty: Ty,
    /// the function this value lives in (None: constants and globals)
    pub func: Option<LLVMValueRef>,
}

impl Value {
    /// The value if this is an integer constant (booleans: 0 or 1).
    pub fn const_int_value(&self) -> Option<i64> {
        // SAFETY: `raw` is a live value of this module.
        unsafe {
            if self.ty.is_int() && !c::LLVMIsAConstantInt(self.raw).is_null() {
                if self.ty.is_bool() {
                    return Some(c::LLVMConstIntGetZExtValue(self.raw) as i64);
                }
                return Some(c::LLVMConstIntGetSExtValue(self.raw));
            }
        }
        None
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
    fn const_int(&self) -> Option<i64> {
        self.const_int_value()
    }

    /// Name the value in the IR text (`%name` instead of `%7`), for reading dumps.
    /// Returns the value. Constants have no names.
    fn named(slf: Py<Self>, py: Python<'_>, name: &str) -> Py<Self> {
        {
            let v = slf.borrow(py);
            // SAFETY: `raw` is a live value; LLVM copies the name.
            unsafe {
                if c::LLVMIsConstant(v.raw) == 0 {
                    c::LLVMSetValueName2(v.raw, name.as_ptr() as *const _, name.len());
                }
            }
        }
        slf
    }

    /// The value if this is a float constant.
    #[getter]
    fn const_float(&self) -> Option<f64> {
        // SAFETY: `raw` is a live value of this module.
        unsafe {
            if self.ty.is_float() && !c::LLVMIsAConstantFP(self.raw).is_null() {
                let mut lost = 0;
                return Some(c::LLVMConstRealGetDouble(self.raw, &mut lost));
            }
        }
        None
    }

    fn __repr__(&self) -> String {
        // SAFETY: `raw` is a live value; the returned message is freed.
        unsafe {
            let s = c::LLVMPrintValueToString(self.raw);
            let r = CStr::from_ptr(s).to_string_lossy().trim().to_string();
            c::LLVMDisposeMessage(s);
            format!("<ir.Value {r}>")
        }
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
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct Function {
    pub m: Rc<ModuleInner>,
    pub raw: LLVMValueRef,
    pub fnty: LLVMTypeRef,
    pub name: String,
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub exported: bool,
    has_builder: Cell<bool>,
}

#[pymethods]
impl Function {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn params(&self) -> Vec<Value> {
        (0..self.params.len())
            .map(|k| Value {
                m: self.m.clone(),
                // SAFETY: `raw` is a function with this many parameters.
                raw: unsafe { c::LLVMGetParam(self.raw, k as u32) },
                ty: self.params[k].clone(),
                func: Some(self.raw),
            })
            .collect()
    }

    #[getter]
    fn ret(&self) -> PyType {
        PyType(self.ret.clone())
    }

    /// A builder for the body. A function has one.
    fn builder(&self) -> PyResult<Builder> {
        self.m.check_open()?;
        if self.has_builder.replace(true) {
            return err(format!("function {} already has a builder", self.name));
        }
        Builder::new(self.m.clone(), self.raw, self.ret.clone())
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
#[pyclass(module = "pytensor_llvm.ir", unsendable)]
pub struct Module {
    pub m: Rc<ModuleInner>,
    functions: RefCell<Vec<(String, LLVMValueRef, bool)>>,
    /// a readable name for the generated code in perf maps (not part of the IR, so not
    /// of the cache key)
    label: RefCell<Option<String>>,
}

#[pymethods]
impl Module {
    #[new]
    #[pyo3(signature = (name, fastmath = true))]
    fn new(name: &str, fastmath: bool) -> Self {
        init_llvm();
        // SAFETY: creating a fresh context and a module in it.
        let (ctx, module) = unsafe {
            let ctx = c::LLVMContextCreate();
            let n = cstring(name);
            (ctx, c::LLVMModuleCreateWithNameInContext(n.as_ptr(), ctx))
        };
        let m = ModuleInner {
            ctx,
            module,
            name: name.to_string(),
            fastmath,
            finished: Cell::new(false),
            vector_decls: RefCell::new(vec![]),
            types: RefCell::new(HashMap::new()),
        };
        Module {
            m: Rc::new(m),
            functions: RefCell::new(vec![]),
            label: RefCell::new(None),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.m.name
    }

    /// Readable name of the code in perf maps (default: the module name).
    #[getter]
    fn get_label(&self) -> String {
        self.label
            .borrow()
            .clone()
            .unwrap_or_else(|| self.m.name.clone())
    }

    #[setter]
    fn set_label(&self, label: String) {
        *self.label.borrow_mut() = Some(label);
    }

    /// Add a function. `exported` functions keep a symbol after compilation (their
    /// address can be looked up); the others are internal.
    #[pyo3(signature = (name, params, ret, exported = false, inline = Inline::Default))]
    fn function(
        &self,
        name: &str,
        params: Vec<PyType>,
        ret: PyType,
        exported: bool,
        inline: Inline,
    ) -> PyResult<Function> {
        self.m.check_open()?;
        if self.functions.borrow().iter().any(|(n, _, _)| n == name) {
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
            let mut ps: Vec<LLVMTypeRef> = params.iter().map(|t| self.m.llvm_type(t)).collect();
            let fnty =
                c::LLVMFunctionType(self.m.llvm_type(&ret), ps.as_mut_ptr(), ps.len() as u32, 0);
            let n = cstring(name);
            let f = c::LLVMAddFunction(self.m.module, n.as_ptr(), fnty);
            c::LLVMSetLinkage(
                f,
                if exported {
                    LLVMLinkage::LLVMExternalLinkage
                } else {
                    LLVMLinkage::LLVMInternalLinkage
                },
            );
            match inline {
                Inline::Always => add_fn_attr(&self.m, f, "alwaysinline"),
                Inline::Never => add_fn_attr(&self.m, f, "noinline"),
                Inline::Default => {}
            }
            (f, fnty)
        };
        self.functions
            .borrow_mut()
            .push((name.to_string(), raw, exported));
        Ok(Function {
            m: self.m.clone(),
            raw,
            fnty,
            name: name.to_string(),
            params,
            ret,
            exported,
            has_builder: Cell::new(false),
        })
    }

    /// The module as LLVM IR text.
    fn ir(&self) -> String {
        print_module(self.m.module)
    }

    /// Check the module; raises with LLVM's message if it is invalid.
    fn verify(&self) -> PyResult<()> {
        verify(&self.m)
    }

    /// Verify, optimize, generate machine code and link it into the JIT. The object is
    /// cached by a hash of the IR and the target. Afterwards the module can't change.
    #[pyo3(signature = (opt_level = 3, cache = true))]
    fn compile(&self, py: Python<'_>, opt_level: u32, cache: bool) -> PyResult<CompiledModule> {
        let mut r = compile_modules(py, &[self], opt_level, cache)?;
        Ok(r.remove(0))
    }
}

impl Module {
    /// Finish the module for compilation: verify, hash, give exported functions unique
    /// symbol names.
    fn prepare(&self, opt_level: u32) -> PyResult<Prepared> {
        self.m.check_open()?;
        for (name, f, _) in self.functions.borrow().iter() {
            // SAFETY: `f` is a function of this module.
            if unsafe { c::LLVMCountBasicBlocks(*f) } == 0 {
                return err(format!("function {name} has no body"));
            }
        }
        finish_vector_decls(&self.m);
        if crate::perf::enabled() {
            for (_, f, _) in self.functions.borrow().iter() {
                add_fn_string_attr(&self.m, *f, "frame-pointer", "all");
            }
        }
        verify(&self.m)?;
        self.m.finished.set(true);
        let text = print_module(self.m.module);
        let mut h = sha2::Sha256::new();
        h.update(salt(opt_level).as_bytes());
        h.update(text.as_bytes());
        let key = hex::encode(&h.finalize()[..16]);
        let exported: Vec<(String, LLVMValueRef)> = self
            .functions
            .borrow()
            .iter()
            .filter(|(_, _, e)| *e)
            .map(|(n, f, _)| (n.clone(), *f))
            .collect();
        if exported.is_empty() {
            return err(format!("module {} exports no function", self.m.name));
        }
        let mut symbols = Vec::new();
        for (name, f) in &exported {
            let sym = format!("pt{key}_{name}");
            let s = cstring(&sym);
            // SAFETY: renaming a function of this (now immutable) module.
            unsafe { c::LLVMSetValueName2(*f, s.as_ptr(), sym.len()) };
            symbols.push((name.clone(), sym));
        }
        let name = self
            .label
            .borrow()
            .clone()
            .unwrap_or_else(|| self.m.name.clone());
        Ok(Prepared {
            key,
            symbols,
            module: ModPtr(self.m.module),
            name,
            ir: text,
        })
    }
}

struct Prepared {
    key: String,
    symbols: Vec<(String, String)>,
    module: ModPtr,
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
        .map(|m| m.prepare(opt_level))
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
    struct Job(usize, LLVMModuleRef, String);
    // SAFETY: each module is finished (no Python code touches it) and used by exactly
    // one worker; distinct modules have distinct LLVM contexts.
    unsafe impl Send for Job {}
    let jobs: Vec<Job> = todo
        .iter()
        .map(|&i| Job(i, prepared[i].module.0, prepared[i].key.clone()))
        .collect();
    let results: Vec<(usize, Result<(Vec<u8>, bool, f64), String>)> = py.detach(move || {
        jobs.into_par_iter()
            .map(|Job(i, m, key)| {
                let path = dir.join(format!("ir_{key}.o"));
                if cache {
                    if let Ok(o) = std::fs::read(&path) {
                        return (i, Ok((o, true, 0.0)));
                    }
                }
                let t = std::time::Instant::now();
                let r = codegen(m, opt_level).map(|obj| {
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

struct ModPtr(LLVMModuleRef);

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
    #[pyo3(signature = (opt_level = None))]
    fn optimized_ir(&self, py: Python<'_>, opt_level: Option<u32>) -> PyResult<String> {
        let level = opt_level.unwrap_or(self.opt_level);
        py.detach(|| optimized_text(&self.ir, level))
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

pub fn add_fn_attr(m: &ModuleInner, f: LLVMValueRef, name: &str) {
    // SAFETY: `f` is a function of `m`; enum attribute kinds are looked up by name.
    unsafe {
        let kind = c::LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
        let a = c::LLVMCreateEnumAttribute(m.ctx, kind, 0);
        c::LLVMAddAttributeAtIndex(f, llvm_sys::LLVMAttributeFunctionIndex, a);
    }
}

pub fn add_fn_string_attr(m: &ModuleInner, f: LLVMValueRef, k: &str, v: &str) {
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

fn verify(m: &ModuleInner) -> PyResult<()> {
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
fn finish_vector_decls(m: &ModuleInner) {
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

/// Set the target of `m` and run the optimization pipeline of `opt_level` on it.
fn optimize(m: LLVMModuleRef, tmr: tm::LLVMTargetMachineRef, opt_level: u32) -> Result<(), String> {
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
        let passes = cstring(&format!("default<O{}>", opt_level.min(3)));
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
fn optimized_text(ir: &str, opt_level: u32) -> Result<String, String> {
    init_llvm();
    let machine = target_machine(opt_level)?;
    let tmr = machine.as_mut_ptr();
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
        let r = optimize(module, tmr, opt_level).map(|()| print_module(module));
        c::LLVMDisposeModule(module);
        c::LLVMContextDispose(ctx);
        r
    }
}

/// Optimize a module and generate an object file.
fn codegen(m: LLVMModuleRef, opt_level: u32) -> Result<Vec<u8>, String> {
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
