import functools
import logging
import shlex
from pathlib import Path

import numpy as np

from pytensor.configdefaults import config
from pytensor.graph import Variable


_logger = logging.getLogger("pytensor.tensor.blas")


def view_roots(node: Variable) -> list[Variable]:
    """Return the leaves from a search through consecutive view-maps."""
    owner = node.owner
    if owner is not None:
        try:
            vars_to_views = {owner.outputs[o]: i for o, i in owner.op.view_map.items()}
        except AttributeError:
            return [node]
        if node in vars_to_views:
            answer = []
            for i in vars_to_views[node]:
                answer += view_roots(owner.inputs[i])
            return answer
        else:
            return [node]
    else:
        return [node]


def must_initialize_y_gemv():
    # Check whether Scipy GEMV could output nan if y in not initialized
    from scipy.linalg.blas import get_blas_funcs

    if must_initialize_y_gemv._result is None:
        y = np.full((2,), np.nan)
        x = np.ones((2,))
        A = np.ones((2, 2))
        gemv = get_blas_funcs("gemv", dtype=y.dtype)
        gemv(1.0, A.T, x, 0.0, y, overwrite_y=True, trans=True)
        must_initialize_y_gemv._result = np.isnan(y).any()

    return must_initialize_y_gemv._result


must_initialize_y_gemv._result = None  # type: ignore


def ldflags(libs=True, flags=False, libs_dir=False, include_dir=False):
    """Extract a list of compilation flags from config.blas__ldflags.

    Depending on the options, different type of flags will be kept.
    It returns a list of libraries against which an Op's object file
    should be linked to benefit from a BLAS implementation.

    Parameters
    ----------
    libs : bool, optional
        Extract flags starting with "-l" (the default is True).
    libs_dir : bool, optional
        Extract flags starting with "-L" (the default is False).
    include_dir : bool, optional
        Extract flags starting with "-I" (the default is False).
    flags: bool, optional
        Extract all the other flags (the default is False).

    Returns
    -------
    list of strings
        Extracted flags.

    """
    ldflags_str = config.blas__ldflags
    return _ldflags(
        ldflags_str=ldflags_str,
        libs=libs,
        flags=flags,
        libs_dir=libs_dir,
        include_dir=include_dir,
    )


@functools.cache
def _ldflags(
    ldflags_str: str, libs: bool, flags: bool, libs_dir: bool, include_dir: bool
) -> list[str]:
    """Extract list of compilation flags from a string.

    Depending on the options, different type of flags will be kept.

    Parameters
    ----------
    ldflags_str : string
        The string to process. Typically, this will be the content of
        `config.blas__ldflags`.
    libs : bool
        Extract flags starting with "-l".
    flags: bool
        Extract all the other flags.
    libs_dir: bool
        Extract flags starting with "-L".
    include_dir: bool
        Extract flags starting with "-I".

    Returns
    -------
    list of strings
        Extracted flags.

    """
    rval = []
    if libs_dir:
        found_dyn = False
        dirs = [x[2:] for x in shlex.split(ldflags_str) if x.startswith("-L")]
        l = _ldflags(
            ldflags_str=ldflags_str,
            libs=True,
            flags=False,
            libs_dir=False,
            include_dir=False,
        )
        for d in dirs:
            for f in Path(d.strip('"')).iterdir():
                if f.suffix in {".so", ".dylib", ".dll"}:
                    if any(f.stem.find(ll) >= 0 for ll in l):
                        found_dyn = True
        # Special treatment of clang framework. Specifically for MacOS Accelerate
        if "-framework" in l and "Accelerate" in l:
            found_dyn = True
        if not found_dyn and dirs:
            _logger.warning(
                "We did not find a dynamic library in the "
                "library_dir of the library we use for blas. If you use "
                "ATLAS, make sure to compile it with dynamics library."
            )

    split_flags = shlex.split(ldflags_str)
    skip = False
    for pos, t in enumerate(split_flags):
        if skip:
            skip = False
            continue
        # Remove extra quote.
        if (t.startswith("'") and t.endswith("'")) or (
            t.startswith('"') and t.endswith('"')
        ):
            t = t[1:-1]

        try:
            t0, t1 = t[0], t[1]
            assert t0 == "-" or Path(t).exists()
        except Exception:
            raise ValueError(f'invalid token "{t}" in ldflags_str: "{ldflags_str}"')
        if t == "-framework":
            skip = True
            # Special treatment of clang framework. Specifically for MacOS Accelerate
            # The clang framework implicitly adds: header dirs, libraries, and library dirs.
            # If we choose to always return these flags, we run into a huge deal amount of
            # incompatibilities. For this reason, we only return the framework if libs are
            # requested.
            if (
                libs
                and len(split_flags) >= pos
                and split_flags[pos + 1] == "Accelerate"
            ):
                # We only add the Accelerate framework, but in the future we could extend it to
                # other frameworks
                rval.append(t)
                rval.append(split_flags[pos + 1])
        elif libs_dir and t1 == "L":
            rval.append(t[2:])
        elif include_dir and t1 == "I":
            raise ValueError(
                "Include dirs are not used for blas. We disable"
                " this as this can hide other headers and this"
                " is not wanted.",
                t,
            )
        elif libs and t1 == "l":  # example -lmkl
            rval.append(t[2:])
        elif flags and t1 not in ("L", "I", "l"):  # example -openmp
            rval.append(t)
        elif flags and t1 == "L":
            # to find it when we load the compiled op if the env of the
            # used is not well configured.
            rval.append("-Wl,-rpath," + t[2:])
    return rval
