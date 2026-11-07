#![allow(unsafe_op_in_unsafe_fn)]
//! Native code generation and runtime for PyTensor's LLVM backend.

pub mod abi;
pub mod capi;
pub mod codegen;
pub mod compile;
pub mod dtype;
pub mod ir;
pub mod jit;
pub mod native;
pub mod runtime;
pub mod segment;
pub mod spec;

mod builder;
mod py;
