"""
This module contains housekeeping functions for cleaning/purging the "compiledir".
It is used by the "pytensor-cache" CLI tool, located in the /bin folder of the repository.
"""

import logging
import pickle
import shutil
from collections import Counter

import numpy as np

from pytensor.configdefaults import config
from pytensor.graph.op import Op
from pytensor.link.c.type import CType
from pytensor.utils import flatten


_logger = logging.getLogger("pytensor.compile.compiledir")


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


def print_compiledir_content():
    """
    print list of %d compiled individual ops in the "pytensor.config.compiledir"
    """
    max_key_file_size = 1 * 1024 * 1024  # 1M

    compiledir = config.compiledir
    table = []
    table_multiple_ops = []
    table_op_class = Counter()
    zeros_op = 0
    big_key_files = []
    total_key_sizes = 0
    nb_keys = Counter()
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
            except OSError:
                pass
            except AttributeError:
                _logger.error(f"Could not read key file '{filename}'.")

    print_title(f"PyTensor cache: {compiledir}", overline="=", underline="=")
    print()  # noqa: T201

    print_title(f"List of {len(table)} compiled individual ops", underline="+")
    print_title(
        "sub dir/compiletime/Op/set of different associated PyTensor types",
        underline="-",
    )
    table = sorted(table, key=lambda t: str(t[1]))
    for dir, op, types, compile_time in table:
        print(dir, f"{compile_time:.3f}s", op, types)  # noqa: T201

    print()  # noqa: T201
    print_title(
        f"List of {len(table_multiple_ops)} compiled sets of ops", underline="+"
    )
    print_title(
        "sub dir/compiletime/Set of ops/set of different associated PyTensor types",
        underline="-",
    )
    table_multiple_ops = sorted(table_multiple_ops, key=lambda t: (t[1], t[2]))
    for dir, ops_to_str, types_to_str, compile_time in table_multiple_ops:
        print(dir, f"{compile_time:.3f}s", ops_to_str, types_to_str)  # noqa: T201

    print()  # noqa: T201
    print_title(
        (
            f"List of {len(table_op_class)} compiled Op classes and "
            "the number of times they got compiled"
        ),
        underline="+",
    )
    for op_class, nb in reversed(table_op_class.most_common()):
        print(op_class, nb)  # noqa: T201

    if big_key_files:
        big_key_files = sorted(big_key_files, key=lambda t: str(t[1]))
        big_total_size = sum(sz for _, sz, _ in big_key_files)
        print(  # noqa: T201
            f"There are directories with key files bigger than {int(max_key_file_size)} bytes "
            "(they probably contain big tensor constants)"
        )
        print(  # noqa: T201
            f"They use {int(big_total_size)} bytes out of {int(total_key_sizes)} (total size "
            "used by all key files)"
        )

        for dir, size, ops in big_key_files:
            print(dir, size, ops)  # noqa: T201

    nb_keys = sorted(nb_keys.items())
    print()  # noqa: T201
    print_title("Number of keys for a compiled module", underline="+")
    print_title(
        "number of keys/number of modules with that number of keys", underline="-"
    )
    for n_k, n_m in nb_keys:
        print(n_k, n_m)  # noqa: T201
    print()  # noqa: T201
    print(  # noqa: T201
        f"Skipped {int(zeros_op)} files that contained 0 op "
        "(are they always pytensor.scalar ops?)"
    )


def compiledir_purge():
    shutil.rmtree(config.compiledir)


def basecompiledir_ls():
    """
    Print list of files in the "pytensor.config.base_compiledir"
    """
    subdirs = []
    others = []
    for f in config.base_compiledir.iterdir():
        if f.is_dir():
            subdirs.append(f)
        else:
            others.append(f)

    subdirs = sorted(subdirs)
    others = sorted(others)

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
