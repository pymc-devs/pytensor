//! Linux `perf` integration: with `PYTENSOR_LLVM_PERF=1`,
//!
//! * every JIT-linked function is appended to `/tmp/perf-<pid>.map`, the file
//!   `perf report` reads to name samples in anonymous executable memory;
//! * and to a jitdump, `jit-<pid>.dump` in `PYTENSOR_LLVM_JITDUMP_DIR` (default `/tmp`):
//!   the function's name, address and machine code, and, for code compiled with debug
//!   information (`debug_info`), the Python line each instruction comes from. With
//!   `perf record -k 1` (the jitdump's clock), `perf inject --jit -i perf.data -o
//!   jit.data` makes an ELF file per function, so `perf report -i jit.data` and `perf
//!   annotate -i jit.data` show the instructions next to the Python code.
//!
//! A jitdump has one source line per address. The debug information has the whole
//! Python call stack (`ir/debug.rs`); the line given is that of the innermost frame
//! outside the generic helpers (`HELPER_FILES`), e.g. in the emitter of the op.

use super::enabled;
use object::{Object, ObjectSection, ObjectSymbol, SymbolKind};
use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

fn map_file() -> Option<&'static Mutex<File>> {
    static MAP: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    MAP.get_or_init(|| {
        let path = format!("/tmp/perf-{}.map", std::process::id());
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(Mutex::new)
    })
    .as_ref()
}

/// A function of a linked object: its address, size and name.
struct Linked {
    /// its offset in its section (the line tables' addresses)
    offset: u64,
    addr: u64,
    size: u64,
    name: String,
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
    let Ok(o) = object::File::parse(obj) else {
        return;
    };
    // (the module's FILE symbol has the same name as the kernel)
    let Some(anchor) = o
        .symbols()
        .find(|s| s.kind() == SymbolKind::Text && s.name() == Ok(name))
    else {
        return;
    };
    let Some(section) = anchor.section_index() else {
        return;
    };
    let base = (addr as u64).wrapping_sub(anchor.address());
    let functions: Vec<Linked> = o
        .symbols()
        .filter(|s| {
            s.kind() == SymbolKind::Text && s.section_index() == Some(section) && s.size() > 0
        })
        .filter_map(|s| {
            Some(Linked {
                offset: s.address(),
                addr: base + s.address(),
                size: s.size(),
                name: pretty_name(name, s.name().ok()?, label),
            })
        })
        .collect();

    if let Some(file) = map_file() {
        let mut lines: String = functions
            .iter()
            .map(|f| format!("{:x} {:x} {}\n", f.addr, f.size, f.name))
            .collect();
        if lines.is_empty() {
            // no sized function symbols: at least name the entry point
            let section_name = o
                .section_by_index(section)
                .ok()
                .and_then(|s| s.name().ok().map(str::to_owned));
            lines = format!(
                "{addr:x} 1 {name} [{label}] ({})\n",
                section_name.unwrap_or_default()
            );
        }
        if let Ok(mut f) = file.lock() {
            let _ = f.write_all(lines.as_bytes());
            let _ = f.flush();
        }
    }

    if let Some(dump) = jitdump() {
        let lines = LineTables::new(&o);
        if let Ok(mut d) = dump.lock() {
            for f in &functions {
                if let Some(t) = &lines {
                    d.debug_info(f.addr, &t.entries(f.offset, f.size, f.addr - f.offset));
                }
                // SAFETY: the JIT linked `f.size` bytes of code at `f.addr` (readable
                // and never freed: the JIT keeps everything it linked).
                let code =
                    unsafe { std::slice::from_raw_parts(f.addr as *const u8, f.size as usize) };
                d.code_load(f.addr, code, &f.name);
            }
            let _ = d.file.flush();
        }
    }
}

// ------------------------------------------------------------------ jitdump
// The format of perf's tools/perf/Documentation/jitdump-specification.txt.

const JITDUMP_MAGIC: u32 = 0x4A69_5444;
const JIT_CODE_LOAD: u32 = 0;
const JIT_CODE_DEBUG_INFO: u32 = 2;

struct JitDump {
    file: std::io::BufWriter<File>,
    code_index: u64,
}

/// The process's jitdump, created with its header (and mapped executable, which is how
/// `perf record` notices it) on first use.
fn jitdump() -> Option<&'static Mutex<JitDump>> {
    static DUMP: OnceLock<Option<Mutex<JitDump>>> = OnceLock::new();
    DUMP.get_or_init(|| {
        let dir = std::env::var("PYTENSOR_LLVM_JITDUMP_DIR").unwrap_or_else(|_| "/tmp".into());
        let path = format!("{dir}/jit-{}.dump", std::process::id());
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .ok()?;
        // SAFETY: mapping the first page of a file we opened; the mapping is never
        // used and stays for the life of the process (perf only needs to see it).
        let mapped = unsafe {
            use std::os::fd::AsRawFd;
            libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_EXEC,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return None;
        }
        let mut d = JitDump {
            file: std::io::BufWriter::new(file),
            code_index: 0,
        };
        let mut h = Vec::with_capacity(40);
        for v in [JITDUMP_MAGIC, 1, 40, elf_machine(), 0, std::process::id()] {
            h.extend_from_slice(&v.to_ne_bytes());
        }
        h.extend_from_slice(&timestamp().to_ne_bytes());
        h.extend_from_slice(&0u64.to_ne_bytes()); // flags
        d.file.write_all(&h).ok()?;
        Some(Mutex::new(d))
    })
    .as_ref()
}

fn elf_machine() -> u32 {
    match std::env::consts::ARCH {
        "x86_64" => 62,
        "aarch64" => 183,
        _ => 0,
    }
}

/// The jitdump's clock: CLOCK_MONOTONIC in ns (`perf record -k 1`).
fn timestamp() -> u64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime writes the timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

impl JitDump {
    fn record(&mut self, id: u32, body: &[u8]) {
        let mut r = Vec::with_capacity(16 + body.len());
        r.extend_from_slice(&id.to_ne_bytes());
        r.extend_from_slice(&(16 + body.len() as u32).to_ne_bytes());
        r.extend_from_slice(&timestamp().to_ne_bytes());
        r.extend_from_slice(body);
        let _ = self.file.write_all(&r);
    }

    fn code_load(&mut self, addr: u64, code: &[u8], name: &str) {
        let mut b = Vec::with_capacity(40 + name.len() + 1 + code.len());
        b.extend_from_slice(&std::process::id().to_ne_bytes());
        // SAFETY: gettid has no preconditions (the syscall: glibc before 2.30 has no
        // wrapper).
        b.extend_from_slice(&(unsafe { libc::syscall(libc::SYS_gettid) } as u32).to_ne_bytes());
        for v in [addr, addr, code.len() as u64, self.code_index] {
            b.extend_from_slice(&v.to_ne_bytes());
        }
        b.extend_from_slice(name.as_bytes());
        b.push(0);
        b.extend_from_slice(code);
        self.record(JIT_CODE_LOAD, &b);
        self.code_index += 1;
    }

    /// The source lines of the function at `addr`: (address, line, file), each until
    /// the next. Written before the function's `code_load`, as perf wants.
    fn debug_info(&mut self, addr: u64, entries: &[(u64, u32, String)]) {
        if entries.is_empty() {
            return;
        }
        let mut b = Vec::new();
        b.extend_from_slice(&addr.to_ne_bytes());
        b.extend_from_slice(&(entries.len() as u64).to_ne_bytes());
        for (a, line, file) in entries {
            b.extend_from_slice(&a.to_ne_bytes());
            b.extend_from_slice(&line.to_ne_bytes());
            b.extend_from_slice(&0u32.to_ne_bytes()); // discriminator
            b.extend_from_slice(file.as_bytes());
            b.push(0);
        }
        self.record(JIT_CODE_DEBUG_INFO, &b);
    }
}

// ------------------------------------------------------------------ line tables

/// Generic helpers of the code generator: their lines say little about where time
/// goes, so a jitdump gives the line of the innermost frame outside them.
const HELPER_FILES: [&str; 3] = ["emit.py", "num.py", "types.py"];

/// Relocations of a DWARF section of a relocatable object.
#[derive(Clone, Copy, Debug)]
struct Relocations<'a>(&'a object::read::RelocationMap);

impl gimli::read::Relocate<usize> for Relocations<'_> {
    fn relocate_address(&self, offset: usize, value: u64) -> gimli::Result<u64> {
        Ok(self.0.relocate(offset as u64, value))
    }

    fn relocate_offset(&self, offset: usize, value: usize) -> gimli::Result<usize> {
        let v = self.0.relocate(offset as u64, value as u64);
        usize::try_from(v).map_err(|_| gimli::Error::OffsetOutOfBounds(v))
    }
}

type Reader<'a> =
    gimli::read::RelocateReader<gimli::EndianSlice<'a, gimli::RunTimeEndian>, Relocations<'a>>;

/// The DWARF sections of an object, with their relocations.
struct LineTables {
    sections: Vec<(
        gimli::SectionId,
        Cow<'static, [u8]>,
        object::read::RelocationMap,
    )>,
    endian: gimli::RunTimeEndian,
}

impl LineTables {
    /// None if the object has no debug information.
    fn new(o: &object::File) -> Option<Self> {
        o.section_by_name(".debug_info")?;
        let endian = if o.is_little_endian() {
            gimli::RunTimeEndian::Little
        } else {
            gimli::RunTimeEndian::Big
        };
        let mut sections = Vec::new();
        for id in [
            gimli::SectionId::DebugAbbrev,
            gimli::SectionId::DebugAddr,
            gimli::SectionId::DebugInfo,
            gimli::SectionId::DebugLine,
            gimli::SectionId::DebugLineStr,
            gimli::SectionId::DebugRanges,
            gimli::SectionId::DebugRngLists,
            gimli::SectionId::DebugStr,
            gimli::SectionId::DebugStrOffsets,
        ] {
            if let Some(s) = o.section_by_name(id.name()) {
                let data = s.uncompressed_data().ok()?.into_owned();
                let relocations = object::read::RelocationMap::new(o, &s).ok()?;
                sections.push((id, Cow::Owned(data), relocations));
            }
        }
        Some(LineTables { sections, endian })
    }

    /// The source lines of the code at section offsets `offset..offset + size`, as
    /// jitdump entries at `offset + shift` (see `HELPER_FILES` for which frame's line).
    fn entries(&self, offset: u64, size: u64, shift: u64) -> Vec<(u64, u32, String)> {
        self.lines(offset, size, shift).unwrap_or_default()
    }

    fn lines(&self, offset: u64, size: u64, shift: u64) -> Option<Vec<(u64, u32, String)>> {
        let dwarf = gimli::Dwarf::load(|id| -> gimli::Result<Reader<'_>> {
            let (data, relocations): (&[u8], _) =
                match self.sections.iter().find(|(s, _, _)| *s == id) {
                    Some((_, d, r)) => (d, Some(r)),
                    None => (&[], None),
                };
            static EMPTY: OnceLock<object::read::RelocationMap> = OnceLock::new();
            let relocations = relocations.unwrap_or_else(|| EMPTY.get_or_init(Default::default));
            Ok(gimli::read::RelocateReader::new(
                gimli::EndianSlice::new(data, self.endian),
                Relocations(relocations),
            ))
        })
        .ok()?;
        let ctx = addr2line::Context::from_dwarf(dwarf).ok()?;
        let mut entries: Vec<(u64, u32, String)> = Vec::new();
        for (start, _len, _innermost) in ctx.find_location_range(offset, offset + size).ok()? {
            let Ok(mut frames) = ctx.find_frames(start).skip_all_loads() else {
                continue;
            };
            let mut chosen = None;
            while let Ok(Some(frame)) = frames.next() {
                let Some(loc) = frame.location else { continue };
                let (Some(file), Some(line)) = (loc.file, loc.line) else {
                    continue;
                };
                if line == 0 {
                    continue;
                }
                let helper = HELPER_FILES
                    .iter()
                    .any(|h| file.ends_with(&format!("/{h}")));
                if chosen.is_none() || !helper {
                    chosen = Some((line, file.to_string()));
                }
                if !helper {
                    break;
                }
            }
            let Some((line, file)) = chosen else { continue };
            if entries
                .last()
                .is_some_and(|(_, l, f)| *l == line && *f == file)
            {
                continue;
            }
            entries.push((start + shift, line, file));
        }
        Some(entries)
    }
}

/// Readable perf-map names. Code from `pytensor_llvm.ir` has symbols `pt<hash>_<fn>`:
/// shown as `<label>::<fn> [<hash prefix>]`. Other code shows the entry point with its
/// label, helpers as `<entry>::<helper>`.
fn pretty_name(entry: &str, sym: &str, label: &str) -> String {
    let label: String = label
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
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
    hash.bytes()
        .all(|b| b.is_ascii_hexdigit())
        .then_some((hash, func))
}
