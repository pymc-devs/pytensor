#!/usr/bin/env python
"""Check whether Numba/LLVM vectorizes transcendental math calls in this environment.

PyTensor's Numba backend lowers ``exp``/``log`` (and, when ``config.numba__veclib`` is
set, ``log1p``/``expm1``) to scalar libm calls that LLVM's loop vectorizer can replace
with SIMD calls into a *vector math library* -- glibc ``libmvec``, Intel SVML, or AMD
AMDLIBM. That substitution only happens when such a library is wired into LLVM.

Run this to confirm whether your environment picks one up::

    python scripts/check_numba_veclib.py

If it reports ``VECTORIZED``, enable the SIMD ``log1p``/``expm1`` lowerings by setting
``pytensor.config.numba__veclib`` to the name of the library it found -- ``"libmvec"``,
``"svml"``, or ``"amdlibm"`` (or the ``PYTENSOR_FLAGS`` / ``.pytensorrc`` equivalent).
Otherwise keep the default (``""``): without a vector library those lowerings only add
work over the scalar libm calls.

Wiring up glibc ``libmvec`` (Linux/glibc) looks like this, *before* importing numba::

    import llvmlite.binding as llvm

    llvm.set_option("", "-vector-library=LIBMVEC-X86")  # "LIBMVEC" on LLVM >= 21
    llvm.load_library_permanently("libmvec.so.1")
"""

import re

import numba
import numpy as np

import pytensor
import pytensor.tensor as pt


# Maps an assembly symbol prefix to (numba__veclib config value, human description)
# for the library that exports it.
VECLIB_SYMBOLS = {
    "_ZGV": ("libmvec", "glibc libmvec / GNU vector ABI"),
    "__svml_": ("svml", "Intel SVML"),
    "amd_vr": ("amdlibm", "AMD AMDLIBM"),
}


def detect_vectorized_math() -> dict[str, list[str]]:
    """Compile an ``exp`` loop and report any vector-math symbols in its assembly.

    This mirrors how PyTensor's Numba ``Elemwise`` lowers a transcendental: a scalar
    libm call inside a contiguous loop, which the loop vectorizer rewrites to a packed
    call only when a vector library is available.
    """

    @numba.njit
    def exp_loop(x):
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = np.exp(x[i])
        return out

    exp_loop.compile((numba.float64[::1],))
    asm = exp_loop.inspect_asm(exp_loop.signatures[0])
    return {
        cfg: (desc, sorted(set(re.findall(rf"{re.escape(prefix)}\w*", asm))))
        for prefix, (cfg, desc) in VECLIB_SYMBOLS.items()
        if prefix in asm
    }


def main() -> int:
    # Sanity check that PyTensor's Numba backend itself works end to end.
    x = pt.vector("x")
    fn = pytensor.function([x], pt.exp(x), mode="NUMBA")
    np.testing.assert_allclose(fn(np.linspace(-1, 1, 8)), np.exp(np.linspace(-1, 1, 8)))

    found = detect_vectorized_math()
    print(f"current pytensor.config.numba__veclib = {pytensor.config.numba__veclib}\n")

    if found:
        print("VECTORIZED: exp lowered to SIMD vector-math calls:")
        for desc, syms in found.values():
            print(f"  - {desc}: {', '.join(syms[:4])}")
        cfg = next(iter(found))
        print("\nA vector math library is wired into LLVM. Enable the SIMD")
        print(
            f'log1p/expm1 lowerings with:\n    pytensor.config.numba__veclib = "{cfg}"'
        )
        return 0

    print("SCALAR: exp stayed a scalar libm call -- no vector math library is wired")
    print("into LLVM, so the log1p/expm1 SIMD lowerings would only add overhead.")
    print('Keep pytensor.config.numba__veclib = "" (the default).\n')
    print("To wire up glibc libmvec (Linux/glibc), before importing numba:")
    print("    import llvmlite.binding as llvm")
    print(
        '    llvm.set_option("", "-vector-library=LIBMVEC-X86")  # "LIBMVEC" on LLVM >= 21'
    )
    print('    llvm.load_library_permanently("libmvec.so.1")')
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
