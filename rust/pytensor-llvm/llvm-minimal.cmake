# Initial-cache file for a minimal, static LLVM to link into pytensor-llvm.
#
# What the crate needs from LLVM (see src/jit.rs, src/target.rs, src/ir/):
#   Core, IRReader/AsmParser (LLVMParseIRInContext), Analysis, DebugInfo (DIBuilder),
#   Passes (new pass manager, LLVMRunPasses), Target/TargetMachine + native codegen
#   (object emission to memory), OrcJIT (LLJIT, object files, process symbol generator).
# All of those are LLVM-proper libraries, so no subprojects/runtimes are needed; the
# big savings come from building only the host backend and no tools/tests/optional deps.
#
# llvm-sys links *every* library listed by `llvm-config --libnames`, so whatever is
# installed here is what ends up on the link line (unused objects are dropped by the
# linker, but the libraries must all exist).
#
# CI builds it with ci/build-llvm.sh (version: llvm-version.txt), which also sets the
# Windows specifics (cl, static CRT). By hand (from an llvm-project checkout matching
# llvm-sys, i.e. LLVM 23.1):
#
#   cmake -G Ninja -S llvm -B build-llvm \
#       -C /path/to/pytensor/rust/pytensor-llvm/llvm-minimal.cmake \
#       -DCMAKE_INSTALL_PREFIX=$HOME/opt/llvm-23-min
#   cmake --build build-llvm --target install-distribution
#
#   export LLVM_SYS_231_PREFIX=$HOME/opt/llvm-23-min
#   maturin build --release
#
# Override the target set with e.g. -DLLVM_TARGETS_TO_BUILD="X86;AArch64" if needed
# (e.g. when cross-compiling, where "host" would be wrong).

set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# Only the backend of the machine we build for (Target::initialize_native).
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "")
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "" CACHE STRING "")

set(LLVM_ENABLE_PROJECTS "" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "" CACHE STRING "")

# Static archives, position independent (they go into a Python extension .so).
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_PIC ON CACHE BOOL "")

set(LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "")
set(LLVM_ENABLE_RTTI OFF CACHE BOOL "")
set(LLVM_ENABLE_EH OFF CACHE BOOL "")
# ORC needs threads.
set(LLVM_ENABLE_THREADS ON CACHE BOOL "")

# No optional system dependencies: keeps `llvm-config --system-libs` down to
# -lrt -ldl -lm -lpthread, so nothing extra has to be present at build or load time.
set(LLVM_ENABLE_ZLIB OFF CACHE STRING "")
set(LLVM_ENABLE_ZSTD OFF CACHE STRING "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE STRING "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")
set(LLVM_ENABLE_CURL OFF CACHE BOOL "")
set(LLVM_ENABLE_HTTPLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_FFI OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")  # ignored by newer LLVM, harmless
set(LLVM_ENABLE_DIA_SDK OFF CACHE BOOL "")

set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_BUILD_UTILS OFF CACHE BOOL "")
set(LLVM_INSTALL_UTILS OFF CACHE BOOL "")

# Tools stay *enabled* (otherwise llvm-config gets no install rule), but only what is
# listed here is built and installed by the `install-distribution` target:
# llvm-config (llvm-sys needs it), the headers, and the static libraries.
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_DISTRIBUTION_COMPONENTS
  llvm-config
  llvm-headers
  llvm-libraries
  CACHE STRING "")
