//! Support for profiling generated code with Linux `perf`.
//!
//! Enabled by `PYTENSOR_ENABLE_PERF=1` (read once per process). Then:
//!
//! * generated functions keep a frame pointer (`"frame-pointer"="all"`), so
//!   `perf record -g` can walk the stack through them; this is part of the
//!   object cache key, so profiling and normal objects don't mix;
//! * every JIT-linked function is appended to `/tmp/perf-<pid>.map`, the file
//!   `perf report` reads to name samples in anonymous executable memory.
//!
//! The map holds names and address ranges only: `perf annotate` can't show the
//! instructions of JIT code (that would need jitdump).

use object::{Object, ObjectSection, ObjectSymbol, SymbolKind};
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("PYTENSOR_ENABLE_PERF").as_deref(), Ok(v) if !v.is_empty() && v != "0"))
}

fn map_file() -> Option<&'static Mutex<File>> {
    static MAP: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    MAP.get_or_init(|| {
        let path = format!("/tmp/perf-{}.map", std::process::id());
        std::fs::OpenOptions::new().create(true).append(true).open(path).ok().map(Mutex::new)
    })
    .as_ref()
}

/// Record the functions of a just-linked object file.
///
/// `name` is the symbol we looked up and `addr` its address in the JIT. In a
/// relocatable object a symbol's value is its offset in its section, so every
/// function in the same section sits at `addr - value(name) + value(f)`.
/// Functions in other sections are skipped (codegen puts everything in .text).
pub fn register_object(obj: &[u8], name: &str, addr: usize, label: &str) {
    if !enabled() {
        return;
    }
    let Some(file) = map_file() else { return };
    let Ok(o) = object::File::parse(obj) else { return };
    // (the module's FILE symbol has the same name as the kernel)
    let Some(anchor) = o.symbols().find(|s| s.kind() == SymbolKind::Text && s.name() == Ok(name)) else { return };
    let Some(section) = anchor.section_index() else { return };
    let base = (addr as u64).wrapping_sub(anchor.address());
    let section_name = o.section_by_index(section).ok().and_then(|s| s.name().ok().map(str::to_owned));

    let mut lines = String::new();
    for s in o.symbols() {
        if s.kind() != SymbolKind::Text || s.section_index() != Some(section) || s.size() == 0 {
            continue;
        }
        let Ok(sym) = s.name() else { continue };
        // Readable names: the wrapper gets the kernel description, helpers keep
        // their own name, both with the kernel hash for matching against the plan.
        let pretty = if sym == name { format!("{name} [{label}]") } else { format!("{name}::{sym}") };
        lines.push_str(&format!("{:x} {:x} {}\n", base + s.address(), s.size(), pretty));
    }
    if lines.is_empty() {
        // no sized function symbols: at least name the entry point
        lines = format!("{addr:x} 1 {name} [{label}] ({})\n", section_name.unwrap_or_default());
    }
    if let Ok(mut f) = file.lock() {
        let _ = f.write_all(lines.as_bytes());
        let _ = f.flush();
    }
}

/// Function attributes for generated code when profiling.
pub fn add_frame_pointers(module: &inkwell::module::Module) {
    if !enabled() {
        return;
    }
    let ctx = module.get_context();
    let attr = ctx.create_string_attribute("frame-pointer", "all");
    for f in module.get_functions() {
        if f.count_basic_blocks() > 0 {
            f.add_attribute(inkwell::attributes::AttributeLoc::Function, attr);
        }
    }
}

