//! Debug information for generated code (`ir.Module(debug_info=True)`): which Python
//! code emitted each instruction, and for which graph node.
//!
//! * Every LLVM function (and parallel body) gets a `DISubprogram` named after what it
//!   computes (`Module.function(debug_name=...)`, e.g. the node), with the function's
//!   name as linkage name. Node functions are inlined; LLVM keeps the inlined-at chain,
//!   so after optimization every instruction still knows its node.
//! * An instruction's location is the Python call stack that emitted it, as a chain of
//!   Python functions "inlined" into each other: the innermost frame's line, in a
//!   subprogram for its Python function, inlined at its caller's line, and so on up to
//!   the frame that built the LLVM function (`build_*_function`), inlined at line 0 of
//!   the LLVM function. Tools (remarks, `perf`, debuggers) see Python source lines and
//!   the Python call stack.

use super::module::ModuleState;
use llvm_sys::core as c;
use llvm_sys::debuginfo as di;
use llvm_sys::prelude::{LLVMDIBuilderRef, LLVMMetadataRef, LLVMValueRef};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

/// Python frames past these (the functions that build an LLVM function) belong to
/// other functions.
const BOUNDARIES: [&str; 4] = [
    "build_node_function",
    "build_unit_function",
    "build_kernel_wrapper",
    "compile_node",
];
/// At most this many frames are looked at.
const MAX_FRAMES: usize = 48;

pub struct DebugInfo {
    dib: LLVMDIBuilderRef,
    cu: LLVMMetadataRef,
    /// the file the LLVM functions are in (the module)
    module_file: LLVMMetadataRef,
    subroutine_type: LLVMMetadataRef,
    files: RefCell<HashMap<String, LLVMMetadataRef>>,
    /// (file, qualified name, first line) -> the subprogram of a Python function
    python: RefCell<HashMap<(String, String, u32), LLVMMetadataRef>>,
    /// LLVM function -> its subprogram, and its name there
    functions: RefCell<HashMap<usize, (LLVMMetadataRef, String)>>,
    finalized: Cell<bool>,
}

/// One Python frame: (file, qualified name, first line, current line).
type Frame = (String, String, u32, u32);

impl DebugInfo {
    /// Debug information for `m` (a fresh module named `name`): the compile unit and
    /// the module flags.
    pub fn new(
        ctx: llvm_sys::prelude::LLVMContextRef,
        m: llvm_sys::prelude::LLVMModuleRef,
        name: &str,
    ) -> Self {
        // SAFETY: a DIBuilder of `m`, and metadata in its context.
        unsafe {
            let dib = di::LLVMCreateDIBuilder(m);
            let module_file = file_of(dib, name, "");
            let producer = "pytensor-llvm";
            let cu = di::LLVMDIBuilderCreateCompileUnit(
                dib,
                di::LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguagePython,
                module_file,
                producer.as_ptr() as *const _,
                producer.len(),
                1,
                c"".as_ptr(),
                0,
                0,
                c"".as_ptr(),
                0,
                di::LLVMDWARFEmissionKind::LLVMDWARFEmissionKindFull,
                0,
                0,
                0,
                c"".as_ptr(),
                0,
                c"".as_ptr(),
                0,
            );
            let subroutine_type = di::LLVMDIBuilderCreateSubroutineType(
                dib,
                module_file,
                std::ptr::null_mut(),
                0,
                di::LLVMDIFlagZero,
            );
            let i32t = c::LLVMInt32TypeInContext(ctx);
            for (key, value) in [
                ("Debug Info Version", di::LLVMDebugMetadataVersion()),
                ("Dwarf Version", 5),
            ] {
                let v = c::LLVMValueAsMetadata(c::LLVMConstInt(i32t, value as u64, 0));
                c::LLVMAddModuleFlag(
                    m,
                    llvm_sys::LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorWarning,
                    key.as_ptr() as *const _,
                    key.len(),
                    v,
                );
            }
            DebugInfo {
                dib,
                cu,
                module_file,
                subroutine_type,
                files: RefCell::new(HashMap::new()),
                python: RefCell::new(HashMap::new()),
                functions: RefCell::new(HashMap::new()),
                finalized: Cell::new(false),
            }
        }
    }

    fn subprogram(
        &self,
        name: &str,
        linkage: &str,
        file: LLVMMetadataRef,
        line: u32,
    ) -> LLVMMetadataRef {
        // SAFETY: a subprogram (a definition) in this compile unit.
        unsafe {
            di::LLVMDIBuilderCreateFunction(
                self.dib,
                file,
                name.as_ptr() as *const _,
                name.len(),
                linkage.as_ptr() as *const _,
                linkage.len(),
                file,
                line,
                self.subroutine_type,
                1,
                1,
                line,
                di::LLVMDIFlagZero,
                1,
            )
        }
    }

    /// Give the LLVM function `f` (named `linkage`) a subprogram named `name`.
    pub fn add_function(&self, f: LLVMValueRef, name: &str, linkage: &str) {
        let sp = self.subprogram(name, linkage, self.module_file, 0);
        // SAFETY: `f` is a function of this module.
        unsafe { di::LLVMSetSubprogram(f, sp) };
        self.functions
            .borrow_mut()
            .insert(f as usize, (sp, name.to_string()));
    }

    pub fn subprogram_of(&self, f: LLVMValueRef) -> Option<LLVMMetadataRef> {
        self.functions
            .borrow()
            .get(&(f as usize))
            .map(|(sp, _)| *sp)
    }

    /// The name of the LLVM function `f` in debug information.
    pub fn name_of(&self, f: LLVMValueRef) -> Option<String> {
        self.functions
            .borrow()
            .get(&(f as usize))
            .map(|(_, n)| n.clone())
    }

    fn file(&self, path: &str) -> LLVMMetadataRef {
        if let Some(&f) = self.files.borrow().get(path) {
            return f;
        }
        let (dir, name) = path.rsplit_once('/').unwrap_or(("", path));
        let f = file_of(self.dib, name, dir);
        self.files.borrow_mut().insert(path.to_string(), f);
        f
    }

    fn python_function(&self, file: &str, name: &str, first: u32) -> LLVMMetadataRef {
        let key = (file.to_string(), name.to_string(), first);
        if let Some(&sp) = self.python.borrow().get(&key) {
            return sp;
        }
        let sp = self.subprogram(name, "", self.file(file), first);
        self.python.borrow_mut().insert(key, sp);
        sp
    }

    /// The location of what the Python code running now emits into the LLVM function
    /// `f`: its call stack (see the module docs), or line 0 of `f`.
    pub fn location(
        &self,
        ctx: llvm_sys::prelude::LLVMContextRef,
        f: LLVMValueRef,
    ) -> Option<LLVMMetadataRef> {
        let sp = self.subprogram_of(f)?;
        let frames = Python::attach(|py| python_frames(py).unwrap_or_default());
        // SAFETY: locations in this module's context, scopes of this compile unit.
        unsafe {
            let mut at = di::LLVMDIBuilderCreateDebugLocation(ctx, 0, 0, sp, std::ptr::null_mut());
            for (file, name, first, line) in frames.iter().rev() {
                let scope = self.python_function(file, name, *first);
                at = di::LLVMDIBuilderCreateDebugLocation(ctx, *line, 0, scope, at);
            }
            Some(at)
        }
    }

    /// Give every instruction of the functions with a subprogram a location: those
    /// built without one (allocas, captured values, ...) take the one before them in
    /// their block, else line 0 of the function. Then finish the debug information.
    pub fn finalize(&self, m: &ModuleState) {
        if self.finalized.replace(true) {
            return;
        }
        // SAFETY: walking the functions, blocks and instructions of the module.
        unsafe {
            for (&f, &(sp, _)) in self.functions.borrow().iter() {
                let f = f as LLVMValueRef;
                let base =
                    di::LLVMDIBuilderCreateDebugLocation(m.ctx, 0, 0, sp, std::ptr::null_mut());
                let mut bb = c::LLVMGetFirstBasicBlock(f);
                while !bb.is_null() {
                    let mut last = base;
                    let mut i = c::LLVMGetFirstInstruction(bb);
                    while !i.is_null() {
                        let loc = di::LLVMInstructionGetDebugLoc(i);
                        if loc.is_null() {
                            di::LLVMInstructionSetDebugLoc(i, last);
                        } else {
                            last = loc;
                        }
                        i = c::LLVMGetNextInstruction(i);
                    }
                    bb = c::LLVMGetNextBasicBlock(bb);
                }
            }
            di::LLVMDIBuilderFinalize(self.dib);
        }
    }
}

impl Drop for DebugInfo {
    fn drop(&mut self) {
        // SAFETY: we own the DIBuilder (the module outlives it: `ModuleState` drops this
        // first).
        unsafe { di::LLVMDisposeDIBuilder(self.dib) };
        let _ = self.cu;
    }
}

fn file_of(dib: LLVMDIBuilderRef, name: &str, dir: &str) -> LLVMMetadataRef {
    // SAFETY: a file descriptor of the DIBuilder's module; LLVM copies the strings.
    unsafe {
        di::LLVMDIBuilderCreateFile(
            dib,
            name.as_ptr() as *const _,
            name.len(),
            dir.as_ptr() as *const _,
            dir.len(),
        )
    }
}

/// The directory of the `pytensor` package (with a trailing slash).
fn pytensor_root(py: Python<'_>) -> PyResult<String> {
    let file: String = py.import("pytensor")?.getattr("__file__")?.extract()?;
    let dir = file.rsplit_once('/').map_or("", |(d, _)| d);
    Ok(format!("{dir}/"))
}

/// The Python frames of pytensor emitting code now, innermost first, up to the one that
/// builds the LLVM function (see `BOUNDARIES`). Others (functools' wrappers, tests)
/// are left out.
fn python_frames(py: Python<'_>) -> PyResult<Vec<Frame>> {
    static ROOT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let root = match ROOT.get() {
        Some(r) => r,
        None => {
            let r = pytensor_root(py)?;
            ROOT.get_or_init(|| r)
        }
    };
    let mut frames = Vec::new();
    let mut f = py.import("sys")?.call_method1("_getframe", (0,))?;
    for _ in 0..MAX_FRAMES {
        if f.is_none() {
            break;
        }
        let code = f.getattr("f_code")?;
        let file: String = code.getattr("co_filename")?.extract()?;
        if !file.starts_with(root.as_str()) {
            f = f.getattr("f_back")?;
            continue;
        }
        let name: String = code
            .getattr("co_qualname")
            .or_else(|_| code.getattr("co_name"))?
            .extract()?;
        let first: u32 = code.getattr("co_firstlineno")?.extract()?;
        let line: u32 = f
            .getattr("f_lineno")?
            .extract::<Option<u32>>()?
            .unwrap_or(first);
        let boundary = BOUNDARIES.contains(&name.as_str());
        frames.push((file, name, first, line));
        if boundary {
            break;
        }
        f = f.getattr("f_back")?;
    }
    Ok(frames)
}

/// A readable form of the location `loc` of an instruction: the Python frames
/// innermost first ("file:line function"), then the LLVM function's subprogram.
///
/// # Safety
/// `loc` is a `DILocation` of a live module whose context is `ctx`.
pub unsafe fn describe(ctx: llvm_sys::prelude::LLVMContextRef, mut loc: LLVMMetadataRef) -> String {
    let mut parts = Vec::new();
    // SAFETY: walking a location's inlined-at chain; names are read from printed metadata.
    unsafe {
        while !loc.is_null() {
            let scope = di::LLVMDILocationGetScope(loc);
            let line = di::LLVMDILocationGetLine(loc);
            let name = metadata_field(ctx, scope, "name").unwrap_or_default();
            let file = di::LLVMDIScopeGetFile(scope);
            let mut n = 0;
            let fname = if file.is_null() {
                String::new()
            } else {
                let p = di::LLVMDIFileGetFilename(file, &mut n);
                String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, n as usize))
                    .into_owned()
            };
            let next = di::LLVMDILocationGetInlinedAt(loc);
            if next.is_null() {
                parts.push(name);
            } else {
                parts.push(format!("{fname}:{line} {name}"));
            }
            loc = next;
        }
    }
    parts.join(" < ")
}

/// The string field `field` of the metadata node `md` (from its printed form).
unsafe fn metadata_field(
    ctx: llvm_sys::prelude::LLVMContextRef,
    md: LLVMMetadataRef,
    field: &str,
) -> Option<String> {
    // SAFETY: printing a metadata node of `ctx`; the message is freed.
    let text = unsafe {
        let v = c::LLVMMetadataAsValue(ctx, md);
        let s = c::LLVMPrintValueToString(v);
        let t = std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned();
        c::LLVMDisposeMessage(s);
        t
    };
    let key = format!("{field}: \"");
    let start = text.find(&key)? + key.len();
    let end = text[start..].find('"')?;
    Some(text[start..start + end].to_string())
}
