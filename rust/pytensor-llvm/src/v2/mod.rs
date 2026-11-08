//! The function-based backend (see DESIGN.md, "v2").
//!
//! * `graph`: tensor graphs with typed values and op objects;
//! * `op`: the traits an op implements (`TensorOp`, `Kernel`, `Generate`);
//! * `scalar`: scalar ops and Composite;
//! * `unit`: compiled units, the internal array convention, the kernel trampoline;
//! * `interp`: the interpreter;
//! * `ops`: tensor op implementations.

pub mod compile;
pub mod emit;
pub mod graph;
pub mod index;
pub mod interp;
pub mod kernels;
pub mod op;
pub mod ops;
pub mod py;
pub mod scalar;
pub mod types;
pub mod unit;
