#!/usr/bin/env bash
# Build the minimal static LLVM of llvm-minimal.cmake, the version in llvm-version.txt,
# and install it into the prefix given as the only argument.
#
# Used by .github/workflows/llvm.yml: in the manylinux containers, on macOS, and in Git
# Bash with the MSVC environment on Windows. CI caches the result keyed on this script,
# llvm-minimal.cmake and llvm-version.txt.
set -euo pipefail

prefix=$1
here=$(cd "$(dirname "$0")/.." && pwd)
version=$(tr -d '[:space:]' < "$here/llvm-version.txt")
work=llvm-work # relative and short: Windows path length

args=()
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
    # Static CRT, like the Rust side (+crt-static): Python doesn't ship msvcp140.dll.
    args+=(
        -DCMAKE_C_COMPILER=cl
        -DCMAKE_CXX_COMPILER=cl
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
    )
    ;;
esac

# Only the parts of llvm-project LLVM itself needs (the rest also has symlinks, which
# break extraction on Windows). Releases no longer have per-project source tarballs.
url=https://github.com/llvm/llvm-project/releases/download/llvmorg-$version
top=llvm-project-$version.src
mkdir -p "$work"
curl -fsSL --retry 3 "$url/$top.tar.xz" |
    tar -xJf - -C "$work" --strip-components=1 \
        "$top/llvm" "$top/cmake" "$top/third-party" "$top/libc"

cmake -G Ninja -S "$work/llvm" -B "$work/build" \
    -C "$here/llvm-minimal.cmake" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    ${args[@]+"${args[@]}"} # (empty-array safe in macOS's bash 3.2)
cmake --build "$work/build" --target install-distribution
rm -rf "$work"
