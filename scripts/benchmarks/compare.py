"""Compare identical benchmark cases against two source revisions in one environment."""

# ruff: noqa: T201
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
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
    import numba.core.event as event
    import psutil

    import pytensor

    source = Path(args.source).resolve()
    assert Path(pytensor.__file__).resolve().is_relative_to(source), pytensor.__file__
    cache_root = Path(os.environ["NUMBA_CACHE_DIR"]).parent.resolve()
    assert pytensor.config.base_compiledir.resolve().is_relative_to(cache_root)
    assert pytensor.config.compiledir.resolve().is_relative_to(cache_root)
    process = psutil.Process()
    affinity = None
    if hasattr(process, "cpu_affinity"):
        affinity = process.cpu_affinity()[:1]
        process.cpu_affinity(affinity)
    spec = importlib.util.spec_from_file_location("benchmark_cases", args.case_file)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    if args.case is None:
        names = [name for name, *_ in cases.cases(1)]
        assert len(names) == len(set(names)), "case names must be unique"
        Path(args.output).write_text(json.dumps(names))
        return

    # Preparing symbolic graphs/data must not call build() or evaluate the function.
    name, build, points, check = next(
        case for case in cases.cases(args.size) if case[0] == args.case
    )
    measurements = {}
    compilations = {}
    phase = None

    class CompileCounter(event.Listener):
        def on_start(self, evt):
            compilations.setdefault(phase, []).append(
                evt.data["dispatcher"].py_func.__qualname__
            )

        def on_end(self, evt):
            pass

    def measure(label, function):
        nonlocal phase
        phase = label
        print(
            "BENCH_PHASE "
            + json.dumps(
                {
                    "phase": label,
                    "event": "start",
                    "rss_bytes": process.memory_info().rss,
                }
            ),
            flush=True,
        )
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        result = function()
        measurements[label] = {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
        }
        print(
            "BENCH_PHASE "
            + json.dumps(
                {"phase": label, "event": "end", "rss_bytes": process.memory_info().rss}
            ),
            flush=True,
        )
        phase = None
        return result

    first_inputs = next(iter(points.values()))
    with event.install_listener("numba:compile", CompileCounter()):
        function = measure("build", build)
        signatures_before_first_call = list(map(str, function.vm.jit_fn.signatures))
        first_output = measure("first_call", lambda: function(*first_inputs))
        ready_rss = process.memory_info().rss
        # Validation follows the measured first call, so it cannot consume lazy JIT.
        check(first_output, first_inputs)
        del first_output

        def validate():
            for inputs in points.values():
                check(function(*inputs), inputs)

        measure("validation", validate)

        def warmup():
            for inputs in points.values():
                for _ in range(args.warmup_calls):
                    function(*inputs)

        measure("warmup", warmup)
        results = {}
        if args.cache_state == "reload":
            for point_name, inputs in points.items():

                def evaluate(inputs=inputs):
                    samples = []
                    for _ in range(args.batches):
                        start = time.perf_counter_ns()
                        for _ in range(args.calls):
                            function(*inputs)
                        samples.append(
                            (time.perf_counter_ns() - start) / args.calls / 1e9
                        )
                    return samples

                results[f"{name}/{point_name}"] = measure(
                    f"evaluate/{point_name}", evaluate
                )
    stats = function.vm.jit_fn.stats
    hits, misses = sum(stats.cache_hits.values()), sum(stats.cache_misses.values())
    native_compilations = sum(map(len, compilations.values()))
    cache = {
        "requested_state": args.cache_state,
        "top_level_hits": hits,
        "top_level_misses": misses,
        "native_compilations": native_compilations,
        "compilations_by_phase": compilations,
        "verified_reload": hits > 0 and misses == 0 and native_compilations == 0,
        "signatures_before_first_call": signatures_before_first_call,
        "cache_path": str(stats.cache_path),
        "pytensor_base_compiledir": str(pytensor.config.base_compiledir),
        "pytensor_compiledir": str(pytensor.config.compiledir),
        "numba_cache_dir": os.environ["NUMBA_CACHE_DIR"],
    }
    cache["status"] = (
        "full_hit"
        if cache["verified_reload"]
        else "partial_hit"
        if hits
        else "miss_or_uncacheable"
    )
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
        json.dumps(
            {
                "metadata": metadata,
                "case": name,
                "measurements": measurements,
                "cache": cache,
                "ready_rss_bytes": ready_rss,
                "samples": results,
            },
            indent=2,
        )
    )


def run_worker(command, env, source, result, interval):
    """Sample child RSS from the parent; no sampling thread runs in timed code."""
    import psutil

    child = subprocess.Popen(
        command,
        env=env,
        cwd=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process = psutil.Process(child.pid)
    stop, lock = threading.Event(), threading.Lock()
    phase = None
    memory = {
        "method": "parent-sampled process RSS, plus worker phase endpoints",
        "interval_seconds": interval,
        "phases": {},
    }

    def sample():
        while not stop.is_set():
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                return
            with lock:
                if phase is not None:
                    record = memory["phases"][phase]
                    record["sample_count"] += 1
                    record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss)
            stop.wait(interval)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with result.with_suffix(".log").open("w") as log:
            for line in child.stdout:
                log.write(line)
                if line.startswith("BENCH_PHASE "):
                    marker = json.loads(line.removeprefix("BENCH_PHASE "))
                    with lock:
                        rss = marker["rss_bytes"]
                        if marker["event"] == "start":
                            phase = marker["phase"]
                            memory["phases"][phase] = {
                                "baseline_rss_bytes": rss,
                                "peak_rss_bytes": rss,
                                "sample_count": 0,
                            }
                        else:
                            record = memory["phases"][marker["phase"]]
                            record["after_rss_bytes"] = rss
                            record["peak_rss_bytes"] = max(
                                record["peak_rss_bytes"], rss
                            )
                            record["peak_increase_bytes"] = (
                                record["peak_rss_bytes"] - record["baseline_rss_bytes"]
                            )
                            phase = None
                else:
                    print(line, end="", flush=True)
        if child.wait():
            raise subprocess.CalledProcessError(child.returncode, command)
    finally:
        stop.set()
        sampler.join()
        result.with_suffix(".memory.json").write_text(json.dumps(memory, indent=2))
    data = json.loads(result.read_text())
    data["memory"] = memory
    result.write_text(json.dumps(data, indent=2))
    return data


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
        configuration = vars(args) | {
            "case_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (output / "configuration.json").write_text(json.dumps(configuration, indent=2))
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

        def environment(label, cache):
            env = dict(os.environ)
            # PYTENSOR_FLAGS uses POSIX shlex even on Windows. Raw backslashes
            # would be consumed as escapes and silently change the cache path.
            base = shlex.quote((cache / "pytensor").as_posix())
            compiledir = shlex.quote((cache / "pytensor" / "c").as_posix())
            env.update(
                PYTHONPATH=str(sources[label]),
                PYTENSOR_FLAGS=f"cxx=,blas__ldflags=,base_compiledir={base},compiledir={compiledir},numba__cache=True",
                NUMBA_CACHE_DIR=str(cache / "numba"),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                NUMBA_NUM_THREADS="1",
            )
            return env

        def command(result):
            return [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
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
                "--warmup-calls",
                str(args.warmup_calls),
            ]

        names = None
        for label in sources:
            result = output / f"{label}-cases.json"
            subprocess.run(
                [*command(result), "--source", str(sources[label])],
                env=environment(label, temporary / f"discovery-{label}"),
                cwd=sources[label],
                check=True,
            )
            discovered = json.loads(result.read_text())
            if names is not None:
                assert names == discovered, "base/head must expose identical cases"
            names = discovered
        runs = {label: [{} for _ in range(args.rounds)] for label in sources}
        for round_index in range(args.rounds):
            # Reverse each round to balance temperature and run-order effects.
            order = ("base", "head") if round_index % 2 == 0 else ("head", "base")
            for label in order:
                print(
                    f"Running {label} {revisions[label]} round {round_index + 1}",
                    flush=True,
                )
                for case_index, name in enumerate(names):
                    pair = {}
                    # Every cold case/round has an empty cache. Only its reload
                    # process reuses it; neither arm can read the other's entries.
                    cache = temporary / f"cache-{label}-{round_index}-{case_index}"
                    assert not cache.exists()
                    for state in ("cold", "reload"):
                        result = (
                            output / f"{label}-{round_index}-{case_index}-{state}.json"
                        )
                        pair[state] = run_worker(
                            [
                                *command(result),
                                "--source",
                                str(sources[label]),
                                "--case",
                                name,
                                "--cache-state",
                                state,
                            ],
                            environment(label, cache),
                            sources[label],
                            result,
                            args.rss_interval,
                        )
                        print(
                            f"{label} {name} {state}: first call {pair[state]['measurements']['first_call']['wall_seconds']:.3f}s; {pair[state]['cache']['status']}, {pair[state]['cache']['native_compilations']} fresh native compilations",
                            flush=True,
                        )
                    runs[label][round_index][name] = pair
        rows = []
        for group in names:
            for name in runs["base"][0][group]["reload"]["samples"]:
                timings = {
                    label: [
                        statistics.median(run[group]["reload"]["samples"][name])
                        for run in arm
                    ]
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
                            for b, h in zip(
                                timings["base"], timings["head"], strict=True
                            )
                        ],
                    }
                )
        setup_rows = []
        for name in names:
            for state in ("cold", "reload"):
                row = {"case": name, "cache_state": state}
                for label in runs:
                    records = [run[name][state] for run in runs[label]]

                    for metric in ("build", "first_call", "warmup"):
                        row[f"{label}_{metric}_seconds"] = statistics.median(
                            record["measurements"][metric]["wall_seconds"]
                            for record in records
                        )
                    row[f"{label}_ready_seconds"] = statistics.median(
                        sum(
                            record["measurements"][phase]["wall_seconds"]
                            for phase in ("build", "first_call")
                        )
                        for record in records
                    )
                    row[f"{label}_compile_peak_rss_mib"] = statistics.median(
                        max(
                            record["memory"]["phases"][phase]["peak_rss_bytes"]
                            for phase in ("build", "first_call")
                        )
                        / 2**20
                        for record in records
                    )
                    row[f"{label}_ready_rss_mib"] = statistics.median(
                        record["ready_rss_bytes"] / 2**20 for record in records
                    )
                    row[f"{label}_verified_reload"] = all(
                        record["cache"]["verified_reload"] for record in records
                    )
                    row[f"{label}_cache_status"] = ",".join(
                        sorted({record["cache"]["status"] for record in records})
                    )
                    row[f"{label}_native_compilations"] = [
                        record["cache"]["native_compilations"] for record in records
                    ]
                    if state == "reload":
                        for metric in ("peak_rss_bytes", "peak_increase_bytes"):
                            row[f"{label}_eval_{metric.removesuffix('_bytes')}_mib"] = (
                                statistics.median(
                                    max(
                                        value[metric]
                                        for phase, value in record["memory"][
                                            "phases"
                                        ].items()
                                        if phase.startswith("evaluate/")
                                    )
                                    / 2**20
                                    for record in records
                                )
                            )
                setup_rows.append(row)
        (output / "setup-summary.json").write_text(json.dumps(setup_rows, indent=2))
        with (output / "setup-summary.csv").open("w", newline="") as stream:
            fields = list(dict.fromkeys(key for row in setup_rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(setup_rows)
        (output / "summary.json").write_text(json.dumps(rows, indent=2))
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = f"Base: `{revisions['base']}`\n\nHead: `{revisions['head']}`\n\n"
        summary += "Ready = build + first call (JIT/load + one evaluation); imports and graph/data preparation excluded.\n\n"
        summary += "A reused directory can give a partial hit: reload is verified only with a top-level hit, no top-level misses and zero fresh native compilations. Raw records name fresh compilations.\n\n"
        summary += "| Case / cache | Base build s | Head build s | Base first call s | Head first call s | Reload status B/H |\n|---|---:|---:|---:|---:|---|\n"
        for row in setup_rows:
            status = (
                f"{row['base_cache_status']}/{row['head_cache_status']}"
                if row["cache_state"] == "reload"
                else "—"
            )
            summary += f"| {row['case']} / {row['cache_state']} | {row['base_build_seconds']:.3f} | {row['head_build_seconds']:.3f} | {row['base_first_call_seconds']:.3f} | {row['head_first_call_seconds']:.3f} | {status} |\n"
        summary += "\nRSS is sampled from the parent; peaks are approximate, absolute RSS includes shared pages. Evaluation memory comes from the fresh reload process.\n\n"
        summary += "| Case | Cold compile peak MiB B/H | Reload ready MiB B/H | Eval peak MiB B/H | Eval peak increase MiB B/H |\n|---|---:|---:|---:|---:|\n"
        for cold, reload in zip(setup_rows[::2], setup_rows[1::2], strict=True):

            def pair(row, field):
                return f"{row[f'base_{field}']:.1f}/{row[f'head_{field}']:.1f}"

            summary += f"| {cold['case']} | {pair(cold, 'compile_peak_rss_mib')} | {pair(reload, 'ready_rss_mib')} | {pair(reload, 'eval_peak_rss_mib')} | {pair(reload, 'eval_peak_increase_mib')} |\n"
        summary += "\nSpeedup = base / head; above 1 means faster. Runtime excludes construction, first call, validation and warmup.\n\n"
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
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--rss-interval", type=float, default=0.01)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache-state", choices=("cold", "reload"), help=argparse.SUPPRESS
    )
    options = parser.parse_args()
    if (
        min(
            options.size,
            options.rounds,
            options.batches,
            options.calls,
            options.warmup_calls,
        )
        < 1
        or options.rss_interval <= 0
    ):
        parser.error(
            "size, rounds, batches, calls, warmup calls and RSS interval must be positive"
        )
    if options.worker:
        worker(options)
    elif options.head:
        compare(options)
    else:
        parser.error("--head is required")
