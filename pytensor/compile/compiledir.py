"""
This module contains housekeeping functions for cleaning/purging the "compiledir".
It is used by the "pytensor-cache" CLI tool, located in the /bin folder of the repository.
"""

import logging
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pytensor.configdefaults import config
from pytensor.graph.op import Op
from pytensor.link.c.type import CType
from pytensor.link.utils import clear_generated_src
from pytensor.utils import flatten


_logger = logging.getLogger("pytensor.compile.compiledir")


class CompiledirContent(NamedTuple):
    """Contents of the C module cache in `config.compiledir`.

    The Numba and generated-source caches live elsewhere and are not
    included.

    Attributes:
        compiledir: The cache directory that was scanned.
        max_key_file_size: Threshold above which a key file is
            reported in `big_key_files`.
        table: One entry per module with a single Op:
            (dir, op, types, compile_time).
        table_multiple_ops: One entry per module with several Ops:
            (dir, ops_str, types_str, compile_time).
        table_op_class: Compiled count per Op class.
        zeros_op: Number of key files containing zero Ops.
        missing_modules: Unpicklable keys per missing module name.
        unreadable_keys: Number of key files that failed to unpickle.
        big_key_files: (dir, size, ops) for key files above
            `max_key_file_size`.
        total_key_sizes: Total size in bytes of all key files.
        nb_keys: Histogram mapping keys-per-module to the number of
            modules with that count.
    """

    compiledir: Path
    max_key_file_size: int
    table: list[tuple[Path, Op, list[CType], float]]
    table_multiple_ops: list[tuple[Path, str, str, float]]
    table_op_class: Counter[type[Op]]
    zeros_op: int
    missing_modules: Counter[str]
    unreadable_keys: int
    big_key_files: list[tuple[Path, int, list[Op]]]
    total_key_sizes: int
    nb_keys: Counter[int]


def remove_extra_caches(base_compiledir: Path) -> None:
    """Remove the Numba cache and the generated source files."""
    shutil.rmtree(base_compiledir / "numba", ignore_errors=True)
    clear_generated_src()


def cleanup():
    """
    Delete keys in old format from the compiledir.

    Old clean up include key in old format or with old version of the c_code:
    1) keys that have an ndarray in them.
       Now we use a hash in the keys of the constant data.
    2) key that don't have the numpy ABI version in them
    3) They do not have a compile version string

    If there is no key left for a compiled module, we delete the module.

    """
    for directory in config.compiledir.iterdir():
        try:
            filename = directory / "key.pkl"
            # print file
            with filename.open("rb") as file:
                try:
                    keydata = pickle.load(file)

                    for key in list(keydata.keys):
                        have_npy_abi_version = False
                        have_c_compiler = False
                        for obj in flatten(key):
                            if isinstance(obj, np.ndarray):
                                # Reuse have_npy_abi_version to
                                # force the removing of key
                                have_npy_abi_version = False
                                break
                            elif isinstance(obj, str):
                                if obj.startswith("NPY_ABI_VERSION=0x"):
                                    have_npy_abi_version = True
                                elif obj.startswith("c_compiler_str="):
                                    have_c_compiler = True
                            elif isinstance(obj, Op | CType) and hasattr(
                                obj, "c_code_cache_version"
                            ):
                                v = obj.c_code_cache_version()
                                if v not in [(), None] and v not in key[0]:
                                    # Reuse have_npy_abi_version to
                                    # force the removing of key
                                    have_npy_abi_version = False
                                    break

                        if not (have_npy_abi_version and have_c_compiler):
                            try:
                                # This can happen when we move the compiledir.
                                if keydata.key_pkl != filename:
                                    keydata.key_pkl = filename
                                keydata.remove_key(key)
                            except OSError:
                                _logger.error(
                                    f"Could not remove file '{filename}'. To complete "
                                    "the clean-up, please remove manually "
                                    "the directory containing it."
                                )
                    if len(keydata.keys) == 0:
                        shutil.rmtree(directory)

                except (EOFError, AttributeError):
                    _logger.error(
                        f"Could not read key file '{filename}'. To complete "
                        "the clean-up, please remove manually "
                        "the directory containing it."
                    )
        except OSError:
            _logger.error(
                f"Could not clean up this directory: '{directory}'. To complete "
                "the clean-up, please remove it manually."
            )


def print_title(title, overline="", underline=""):
    len_title = len(title)
    if overline:
        print(str(overline) * len_title)  # noqa: T201
    print(title)  # noqa: T201
    if underline:
        print(str(underline) * len_title)  # noqa: T201


def collect_compiledir_content() -> CompiledirContent:
    """Scan `config.compiledir` and return what is in the cache; prints nothing."""
    max_key_file_size = 1 * 1024 * 1024  # 1M

    compiledir = config.compiledir
    table = []
    table_multiple_ops = []
    table_op_class: Counter[type[Op]] = Counter()
    zeros_op = 0
    missing_modules: Counter[str] = Counter()
    unreadable_keys = 0
    big_key_files = []
    total_key_sizes = 0
    nb_keys: Counter[int] = Counter()
    for dir in config.compiledir.iterdir():
        filename = dir / "key.pkl"
        if not filename.exists():
            continue
        with filename.open("rb") as file:
            try:
                keydata = pickle.load(file)
                ops = list({x for x in flatten(keydata.keys) if isinstance(x, Op)})
                # Whatever the case, we count compilations for OP classes.
                table_op_class.update({op.__class__ for op in ops})
                if len(ops) == 0:
                    zeros_op += 1
                else:
                    types = list(
                        {x for x in flatten(keydata.keys) if isinstance(x, CType)}
                    )
                    compile_start = compile_end = float("nan")
                    for fn in dir.iterdir():
                        if fn.name == "mod.c":
                            compile_start = fn.stat().st_mtime
                        elif fn.suffix == ".so":
                            compile_end = fn.stat().st_mtime
                    compile_time = compile_end - compile_start
                    if len(ops) == 1:
                        table.append((dir, ops[0], types, compile_time))
                    else:
                        ops_to_str = f"[{', '.join(sorted(str(op) for op in ops))}]"
                        types_to_str = f"[{', '.join(sorted(str(t) for t in types))}]"
                        table_multiple_ops.append(
                            (dir, ops_to_str, types_to_str, compile_time)
                        )

                size = filename.stat().st_size
                total_key_sizes += size
                if size > max_key_file_size:
                    big_key_files.append((dir, size, ops))

                nb_keys[len(keydata.keys)] += 1
            except ImportError as exc:
                missing_modules[(exc.name or "unknown").split(".")[0]] += 1
            except (OSError, AttributeError, EOFError, pickle.UnpicklingError):
                unreadable_keys += 1
                _logger.error(f"Could not read key file '{filename}'.")
    return CompiledirContent(
        compiledir=compiledir,
        max_key_file_size=max_key_file_size,
        table=table,
        table_multiple_ops=table_multiple_ops,
        table_op_class=table_op_class,
        zeros_op=zeros_op,
        missing_modules=missing_modules,
        unreadable_keys=unreadable_keys,
        big_key_files=big_key_files,
        total_key_sizes=total_key_sizes,
        nb_keys=nb_keys,
    )


def render_compiledir_content(content: CompiledirContent) -> None:
    print_title(f"PyTensor cache: {content.compiledir}", overline="=", underline="=")
    print()  # noqa: T201

    print_title(f"List of {len(content.table)} compiled individual ops", underline="+")
    print_title(
        "sub dir/compiletime/Op/set of different associated PyTensor types",
        underline="-",
    )
    table = sorted(content.table, key=lambda t: str(t[1]))
    for dir, op, types, compile_time in table:
        print(dir, f"{compile_time:.3f}s", op, types)  # noqa: T201

    print()  # noqa: T201
    print_title(
        f"List of {len(content.table_multiple_ops)} compiled sets of ops", underline="+"
    )
    print_title(
        "sub dir/compiletime/Set of ops/set of different associated PyTensor types",
        underline="-",
    )
    table_multiple_ops = sorted(content.table_multiple_ops, key=lambda t: (t[1], t[2]))
    for dir, ops_to_str, types_to_str, compile_time in table_multiple_ops:
        print(dir, f"{compile_time:.3f}s", ops_to_str, types_to_str)  # noqa: T201

    print()  # noqa: T201
    print_title(
        (
            f"List of {len(content.table_op_class)} compiled Op classes and "
            "the number of times they got compiled"
        ),
        underline="+",
    )
    for op_class, nb in reversed(content.table_op_class.most_common()):
        print(op_class, nb)  # noqa: T201

    if content.big_key_files:
        big_key_files = sorted(content.big_key_files, key=lambda t: str(t[1]))
        big_total_size = sum(sz for _, sz, _ in big_key_files)
        print(  # noqa: T201
            f"There are directories with key files bigger than {int(content.max_key_file_size)} bytes "
            "(they probably contain big tensor constants)"
        )
        print(  # noqa: T201
            f"They use {int(big_total_size)} bytes out of {int(content.total_key_sizes)} (total size "
            "used by all key files)"
        )

        for dir, size, ops in big_key_files:
            print(dir, size, ops)  # noqa: T201

    nb_keys = sorted(content.nb_keys.items())
    print()  # noqa: T201
    print_title("Number of keys for a compiled module", underline="+")
    print_title(
        "number of keys/number of modules with that number of keys", underline="-"
    )
    for n_k, n_m in nb_keys:
        print(n_k, n_m)  # noqa: T201
    print()  # noqa: T201
    print(  # noqa: T201
        f"Skipped {int(content.zeros_op)} files that contained 0 op "
        "(are they always pytensor.scalar ops?)"
    )
    if content.missing_modules:
        modules = ", ".join(
            f"{name} ({count})"
            for name, count in sorted(content.missing_modules.items())
        )
        print(  # noqa: T201
            f"Skipped {sum(content.missing_modules.values())} key files from unavailable modules: {modules}"
        )
    if content.unreadable_keys:
        print(f"Skipped {content.unreadable_keys} unreadable key files")  # noqa: T201


def print_compiledir_content() -> None:
    """
    print list of %d compiled individual ops in the "pytensor.config.compiledir"
    """
    render_compiledir_content(collect_compiledir_content())


def compiledir_purge():
    shutil.rmtree(config.compiledir)


def basecompiledir_ls():
    """
    Return the sorted sub-directories and other files in the "pytensor.config.base_compiledir"
    """
    subdirs = []
    others = []
    for f in config.base_compiledir.iterdir():
        if f.is_dir():
            subdirs.append(f)
        else:
            others.append(f)
    return sorted(subdirs), sorted(others)


def render_basecompiledir_ls(subdirs, others) -> None:
    print(f"Base compile dir is {config.base_compiledir}")  # noqa: T201
    print("Sub-directories (possible compile caches):")  # noqa: T201
    for d in subdirs:
        print(f"    {d}")  # noqa: T201
    if not subdirs:
        print("    (None)")  # noqa: T201

    if others:
        print()  # noqa: T201
        print("Other files in base_compiledir:")  # noqa: T201
        for f in others:
            print(f"    {f}")  # noqa: T201


def basecompiledir_purge():
    shutil.rmtree(config.base_compiledir)
