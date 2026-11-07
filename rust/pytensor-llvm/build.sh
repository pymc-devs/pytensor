#!/bin/sh
# Build the extension and install it into the current Python environment.
# Requires LLVM 18 development files (llvm-config on PATH) and a Rust toolchain.
set -e
cd "$(dirname "$0")"
rm -rf target/wheels
maturin build --release --auditwheel skip
pip install --force-reinstall --no-deps target/wheels/*.whl "$@"
