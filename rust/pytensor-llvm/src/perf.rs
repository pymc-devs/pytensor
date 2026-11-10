//! Support for profiling generated code with Linux `perf`.
//!
//! Enabled by `PYTENSOR_LLVM_PERF=1` (read once per process). Then:
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
    *ON.get_or_init(|| matches!(std::env::var("PYTENSOR_LLVM_PERF").as_deref(), Ok(v) if !v.is_empty() && v != "0"))
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
        let pretty = pretty_name(name, sym, label);
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

/// Readable perf-map names. Code from `pytensor_llvm.ir` has symbols `pt<hash>_<fn>`:
/// shown as `<label>::<fn> [<hash prefix>]`. Other code shows the entry point with its
/// label, helpers as `<entry>::<helper>`.
fn pretty_name(entry: &str, sym: &str, label: &str) -> String {
    let label: String = label.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    if let Some((hash, func)) = split_ir_symbol(sym) {
        return format!("{label}::{func} [{}]", &hash[..8]);
    }
    if let Some((hash, _)) = split_ir_symbol(entry) {
        return format!("{label}::{sym} [{}]", &hash[..8]);
    }
    if sym == entry {
        format!("{entry} [{label}]")
    } else {
        format!("{entry}::{sym}")
    }
}

/// `pt<32 hex>_<fn>` -> (hash, fn)
fn split_ir_symbol(sym: &str) -> Option<(&str, &str)> {
    let rest = sym.strip_prefix("pt")?;
    let (hash, func) = (rest.get(..32)?, rest.get(32..)?.strip_prefix('_')?);
    hash.bytes().all(|b| b.is_ascii_hexdigit()).then_some((hash, func))
}

