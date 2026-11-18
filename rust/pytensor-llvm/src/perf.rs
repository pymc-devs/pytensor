//! Support for profiling generated code.
//!
//! Enabled by `PYTENSOR_LLVM_PERF=1` (read once per process). Then generated functions
//! keep a frame pointer (`"frame-pointer"="all"`), so sampling profilers can walk the
//! stack through them; this is part of the object cache key, so profiling and normal
//! objects don't mix. On Linux, JIT-linked functions are also named for `perf` (a perf
//! map and a jitdump, see `linux`).

use std::sync::OnceLock;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::register_object;

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("PYTENSOR_LLVM_PERF").as_deref(), Ok(v) if !v.is_empty() && v != "0"))
}

/// Record the functions of a just-linked object file: only for Linux `perf`.
#[cfg(not(target_os = "linux"))]
pub fn register_object(_obj: &[u8], _name: &str, _addr: usize, _label: &str) {}
