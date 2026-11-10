"""Compare identical benchmark cases against two source revisions in one environment."""

# ruff: noqa: T201
import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def fetch(repository, ref, directory):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("repository must be owner/name")
    if ref.startswith("-") or ":" in ref:
        raise ValueError("expected a revision, not a git option or refspec")
    git(
        "-C",
        directory,
        "fetch",
        "--quiet",
        "--no-tags",
        "--depth=1",
        f"https://github.com/{repository}.git",
        ref,
    )
    return git("-C", directory, "rev-parse", "FETCH_HEAD")


def worker(args):
    import psutil

    import pytensor

    source = Path(args.source).resolve()
    assert Path(pytensor.__file__).is_relative_to(source), pytensor.__file__
    process = psutil.Process()
    affinity = None
    if hasattr(process, "cpu_affinity"):
        affinity = process.cpu_affinity()[:1]
        process.cpu_affinity(affinity)
    spec = importlib.util.spec_from_file_location("benchmark_cases", args.case_file)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    results = {}
    for name, function, inputs in cases.cases(args.size):
        function(*inputs)  # Compile and warm up before measuring.
        samples = []
        for _ in range(args.batches):
            start = time.perf_counter_ns()
            for _ in range(args.calls):
                function(*inputs)
            samples.append((time.perf_counter_ns() - start) / args.calls / 1e9)
        results[name] = samples
        print(f"{name}: {statistics.median(samples) * 1e6:.2f} us", flush=True)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libc": platform.libc_ver(),
        "affinity": affinity,
        "source": str(pytensor.__file__),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "numba", "llvmlite", "psutil")
        },
    }
    if sys.platform == "linux":
        metadata["cpu"] = Path("/proc/cpuinfo").read_text()
    elif sys.platform == "darwin":
        metadata["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    Path(args.output).write_text(
        json.dumps({"metadata": metadata, "samples": results}, indent=2)
    )


def compare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_file = Path(args.case_file).resolve()
    with tempfile.TemporaryDirectory(prefix="pytensor-compare-") as temporary:
        temporary = Path(temporary)
        repository = temporary / "git"
        git("init", "--quiet", "--bare", repository)
        revisions = {
            label: fetch(args.repository, ref, repository)
            for label, ref in (("base", args.base), ("head", args.head))
        }
        if args.resolve_only:
            print(json.dumps(revisions))
            if destination := os.environ.get("GITHUB_OUTPUT"):
                with Path(destination).open("a") as stream:
                    stream.writelines(
                        f"{label}={sha}\n" for label, sha in revisions.items()
                    )
            return
        (output / "revisions.json").write_text(json.dumps(revisions, indent=2))
        sources = {}
        for label, revision in revisions.items():
            source = temporary / label
            source.mkdir()
            archive = temporary / f"{label}.tar"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "archive",
                    "--output",
                    str(archive),
                    revision,
                    "pytensor",
                ],
                check=True,
            )
            with tarfile.open(archive) as stream:
                stream.extractall(source, filter="data")
            sources[label] = source
        runs = {"base": [], "head": []}
        for round_index in range(args.rounds):
            # Reverse each round to balance temperature and run-order effects.
            order = ("base", "head") if round_index % 2 == 0 else ("head", "base")
            for label in order:
                result = output / f"{label}-{round_index}.json"
                env = dict(os.environ)
                env.update(
                    PYTHONPATH=str(sources[label]),
                    PYTENSOR_FLAGS=f"cxx=,blas__ldflags=,base_compiledir={temporary / ('cache-' + label)}",
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    NUMBA_NUM_THREADS="1",
                )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source",
                    str(sources[label]),
                    "--case-file",
                    str(case_file),
                    "--output",
                    str(result),
                    "--size",
                    str(args.size),
                    "--batches",
                    str(args.batches),
                    "--calls",
                    str(args.calls),
                ]
                print(
                    f"Running {label} {revisions[label]} round {round_index + 1}",
                    flush=True,
                )
                subprocess.run(command, env=env, cwd=sources[label], check=True)
                runs[label].append(json.loads(result.read_text()))
        rows = []
        for name in runs["base"][0]["samples"]:
            timings = {
                label: [statistics.median(run["samples"][name]) for run in arm]
                for label, arm in runs.items()
            }
            base, head = (
                statistics.median(timings[label]) for label in ("base", "head")
            )
            rows.append(
                {
                    "case": name,
                    "base_us": base * 1e6,
                    "head_us": head * 1e6,
                    "speedup": base / head,
                    "round_speedups": [
                        b / h
                        for b, h in zip(timings["base"], timings["head"], strict=True)
                    ],
                }
            )
        (output / "summary.json").write_text(json.dumps(rows, indent=2))
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = f"Base: `{revisions['base']}`\n\nHead: `{revisions['head']}`\n\n"
        summary += "Speedup = base / head; above 1 means faster. Times exclude compilation.\n\n"
        summary += "| Case | Base us | Head us | Speedup | Per round |\n|---|---:|---:|---:|---|\n"
        for row in rows:
            ratios = ", ".join(f"{ratio:.2f}" for ratio in row["round_speedups"])
            summary += f"| {row['case']} | {row['base_us']:.2f} | {row['head_us']:.2f} | {row['speedup']:.2f} | {ratios} |\n"
        (output / "summary.md").write_text(summary)
        print(summary)
        if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(destination).open("a") as stream:
                stream.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="pymc-devs/pytensor")
    parser.add_argument("--base", default="main")
    parser.add_argument("--head")
    parser.add_argument("--case-file", default="scripts/benchmarks/special.py")
    parser.add_argument("--output", default="benchmark-results")
    parser.add_argument("--size", type=int, default=200_000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if min(options.size, options.rounds, options.batches, options.calls) < 1:
        parser.error("size, rounds, batches, and calls must be positive")
    if options.worker:
        worker(options)
    elif options.head:
        compare(options)
    else:
        parser.error("--head is required")
