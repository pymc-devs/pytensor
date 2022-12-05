"""
Invokes mypy and compare the reults with files in /pytensor
and a list of files that are known to fail.

Exit code 0 indicates that there are no unexpected results.

Usage
-----
python scripts/run_mypy.py [--verbose]
"""
import argparse
import importlib
import os
import pathlib
import subprocess
import sys
from typing import Iterator

import pandas


DP_ROOT = pathlib.Path(__file__).absolute().parent.parent
FAILING = """
pytensor/compile/builders.py
pytensor/compile/compilelock.py
pytensor/compile/debugmode.py
pytensor/compile/function/pfunc.py
pytensor/compile/function/types.py
pytensor/compile/mode.py
pytensor/compile/sharedvalue.py
pytensor/graph/rewriting/basic.py
pytensor/ifelse.py
pytensor/link/basic.py
pytensor/link/numba/dispatch/elemwise.py
pytensor/link/numba/dispatch/random.py
pytensor/link/numba/dispatch/scan.py
pytensor/printing.py
pytensor/raise_op.py
pytensor/sandbox/rng_mrg.py
pytensor/scalar/basic.py
pytensor/sparse/basic.py
pytensor/sparse/type.py
pytensor/tensor/basic.py
pytensor/tensor/blas.py
pytensor/tensor/blas_c.py
pytensor/tensor/blas_headers.py
pytensor/tensor/elemwise.py
pytensor/tensor/extra_ops.py
pytensor/tensor/math.py
pytensor/tensor/nnet/abstract_conv.py
pytensor/tensor/nnet/ctc.py
pytensor/tensor/nnet/neighbours.py
pytensor/tensor/random/basic.py
pytensor/tensor/random/op.py
pytensor/tensor/random/utils.py
pytensor/tensor/rewriting/basic.py
pytensor/tensor/rewriting/elemwise.py
pytensor/tensor/shape.py
pytensor/tensor/slinalg.py
pytensor/tensor/subtensor.py
pytensor/tensor/type.py
pytensor/tensor/type_other.py
pytensor/tensor/var.py
"""


def enforce_pep561(module_name):
    try:
        module = importlib.import_module(module_name)
        fp = pathlib.Path(module.__path__[0], "py.typed")
        if not fp.exists():
            fp.touch()
    except ModuleNotFoundError:
        print(f"Can't enforce PEP 561 for {module_name} because it is not installed.")
    return


def mypy_to_pandas(input_lines: Iterator[str]) -> pandas.DataFrame:
    """Reformats mypy output with error codes to a DataFrame.

    Adapted from: https://gist.github.com/michaelosthege/24d0703e5f37850c9e5679f69598930a
    """
    current_section = None
    data = {
        "file": [],
        "line": [],
        "type": [],
        "errorcode": [],
        "message": [],
    }
    for line in input_lines:
        line = line.strip()
        elems = line.split(":")
        if len(elems) < 3:
            continue
        try:
            file, lineno, message_type, *_ = elems[0:3]
            message_type = message_type.strip()
            if message_type == "error":
                current_section = line.split("  [")[-1][:-1]
            message = line.replace(f"{file}:{lineno}: {message_type}: ", "").replace(
                f"  [{current_section}]", ""
            )
            data["file"].append(file)
            data["line"].append(lineno)
            data["type"].append(message_type)
            data["errorcode"].append(current_section)
            data["message"].append(message)
        except Exception as ex:
            print(elems)
            print(ex)
    return pandas.DataFrame(data=data).set_index(["file", "line"])


def check_no_unexpected_results(mypy_lines: Iterator[str]):
    """Compares mypy results with list of known FAILING files.

    Exits the process with non-zero exit code upon unexpected results.
    """
    df = mypy_to_pandas(mypy_lines)

    all_files = {
        str(fp).replace(str(DP_ROOT), "").strip(os.sep).replace(os.sep, "/")
        for fp in DP_ROOT.glob("pytensor/**/*.py")
    }
    failing = set(df.reset_index().file.str.replace(os.sep, "/", regex=False))
    if not failing.issubset(all_files):
        raise Exception(
            "Mypy should have ignored these files:\n"
            + "\n".join(sorted(map(str, failing - all_files)))
        )
    passing = all_files - failing
    expected_failing = set(FAILING.strip().split("\n")) - {""}
    unexpected_failing = failing - expected_failing
    unexpected_passing = passing.intersection(expected_failing)

    if not unexpected_failing:
        print(f"{len(passing)}/{len(all_files)} files pass as expected.")
    else:
        print("!!!!!!!!!")
        print(f"{len(unexpected_failing)} files unexpectedly failed.")
        print("\n".join(sorted(map(str, unexpected_failing))))
        print(
            "These files did not fail before, so please check the above output"
            f" for errors in {unexpected_failing} and fix them."
        )
        print(
            "You can run `python scripts/run_mypy.py --verbose` to reproduce this test locally."
        )
        sys.exit(1)

    if unexpected_passing:
        print("!!!!!!!!!")
        print(f"{len(unexpected_passing)} files unexpectedly passed the type checks:")
        print("\n".join(sorted(map(str, unexpected_passing))))
        print(
            "This is good news! Go to scripts/run_mypy.py and remove them from the `FAILING` list."
        )
        if all_files.issubset(passing):
            print("WOW! All files are passing the mypy type checks!")
            print("scripts\\run_mypy.py may no longer be needed.")
        print("!!!!!!!!!")
        sys.exit(1)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run mypy type checks on PyTensor codebase."
    )
    parser.add_argument(
        "--verbose", action="count", default=0, help="Pass this to print mypy output."
    )
    parser.add_argument(
        "--groupby",
        default="file",
        help="How to group verbose output. One of {file|errorcode|message}.",
    )
    args, _ = parser.parse_known_args()

    cp = subprocess.run(
        ["mypy", "--show-error-codes", "pytensor"],
        capture_output=True,
    )
    output = cp.stdout.decode()
    if args.verbose:
        df = mypy_to_pandas(output.split("\n"))
        for section, sdf in df.reset_index().groupby(args.groupby):
            print(f"\n\n[{section}]")
            for row in sdf.itertuples():
                print(f"{row.file}:{row.line}: {row.type}: {row.message}")
        print()
    else:
        print(
            "Mypy output hidden."
            " Run `python run_mypy.py --verbose` to see the full output,"
            " or `python run_mypy.py --help` for other options."
        )

    check_no_unexpected_results(output.split("\n"))
    sys.exit(0)
