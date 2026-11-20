"""Native execution sampling for the Numba linker.

Numba's loaded object and cache metadata are private APIs. Keep their use here,
separate from graph conversion and the backend-independent profiling tables.
The default Linux x86-64 backend uses DWARF caller recovery. Other backends
provide their own object readers and sample decoders; attribution and reports
do not decode instructions or implement architecture-specific stack unwinding.
"""

import inspect
import json
import os
import platform
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numba
from numba.core.registry import CPUDispatcher

from pytensor.compile.debug.profiling import _print_native_summary
from pytensor.printing import _get_inner_graph_op, debugprint


_sampling_lock = threading.Lock()


class NativeRecorder(Protocol):
    """A backend recorder with acknowledged sampling boundaries."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class NativeSamplingBackend(Protocol):
    """Adapt native tooling to shared graph attribution and reporting.

    Backends own platform validation, object-file parsing, recording, sample
    decoding, and optional assembly annotation. ``decode`` exports normalized
    ``samples.txt`` and, when requested, ``callchains.txt`` in the output
    directory. See the developer documentation on native profiling backends
    for their format.
    """

    name: str
    default_event: str
    call_graph: str

    def validate(
        self, *, event: str, n_hot_functions: int, call_graph: bool
    ) -> None: ...

    def functions(self, blob: bytes) -> list[dict]: ...

    def external_symbols(self, blob: bytes) -> set[str]: ...

    def recorder(
        self, output: Path, frequency: int, event: str, call_graph: bool
    ) -> NativeRecorder: ...

    def decode(self, output: Path, *, call_graph: bool) -> dict | None: ...

    def annotate(self, report: dict, output: Path, n_hot_functions: int) -> None: ...


def _compiled_object(dispatcher, signature, compiled):
    try:
        return compiled.library._get_compiled_object()
    except (ValueError, RuntimeError) as error:
        # MCJIT consumes the retained buffer when loading a cached object.
        # Read that exact specialization back without re-emitting LLVM or
        # loading another library (either could change the instruction layout).
        cache = dispatcher._cache
        if hasattr(cache, "_cache_file"):
            key = cache._index_key(signature, compiled.codegen)
            data = cache._cache_file.load(key)
            if (
                data is not None
                and data[1].llvm_func_name == compiled.fndesc.llvm_func_name
                and data[0][1] == "object"
            ):
                return data[0][2][0]
        raise RuntimeError(
            "The loaded Numba object is unavailable. Enable numba__cache and "
            "retain its cache files until native sampling finishes."
        ) from error


def _elf_symbols(blob):
    """Read the symbols in a little-endian ELF64 object."""
    if len(blob) < 64 or blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Native sampling requires little-endian ELF64 objects")
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    size, count = struct.unpack_from("<HH", blob, 58)
    if size < 64 or not count or shoff + count * size > len(blob):
        raise ValueError("Invalid ELF section table")
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", blob, shoff + i * size) for i in range(count)
    ]
    functions = []
    for section in sections:
        if section[1] != 2:
            continue
        if (
            section[6] >= count
            or section[9] < 24
            or section[5] % section[9]
            or section[4] + section[5] > len(blob)
        ):
            raise ValueError("Invalid ELF symbol table")
        names = sections[section[6]]
        strings = blob[names[4] : names[4] + names[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, info, _, index, value, length = struct.unpack_from(
                "<IBBHQQ", blob, offset
            )
            if name:
                end = strings.index(b"\0", name)
                functions.append(
                    dict(
                        name=strings[name:end].decode(),
                        section=index,
                        offset=value,
                        size=length,
                        kind=info & 15,
                    )
                )
    return functions


def _elf_functions(blob):
    return [
        {key: value for key, value in symbol.items() if key != "kind"}
        for symbol in _elf_symbols(blob)
        if symbol["kind"] == 2 and symbol["section"] and symbol["size"]
    ]


def _external_symbols(blob):
    return {symbol["name"] for symbol in _elf_symbols(blob) if not symbol["section"]}


def _prefer_inner_locations(mapping):
    return {
        name: sorted(
            location
            for location in candidates
            if not any(other.startswith(location + "/") for other in candidates)
        )
        for name, candidates in mapping.items()
    }


class NumbaProfiler:
    """Retain per-compilation node associations without modifying JIT code."""

    def __init__(self, fgraph):
        self.fgraph = fgraph
        self.node_functions = {}
        self.dispatcher = None

    def capture(self, output, prefix, backend: NativeSamplingBackend | None = None):
        backend = _LINUX_PERF_BACKEND if backend is None else backend
        dispatcher = self.dispatcher
        if not dispatcher.signatures:
            raise ValueError("Call each Numba function before starting native sampling")

        locations = defaultdict(set)
        nodes = []

        def walk(graph, path):
            for i, node in enumerate(graph.toposort()):
                location = f"{path}/{i}"
                locations[node].add(location)
                nodes.append(
                    dict(
                        location=location,
                        op=str(node.op),
                        op_class=type(node.op).__name__,
                        inputs=[str(var.type) for var in node.inputs],
                        outputs=[str(var.type) for var in node.outputs],
                    )
                )
                if inner_op := _get_inner_graph_op(node.op):
                    for name, inner in inner_op.inner_graphs.items():
                        walk(inner, f"{location}/{name}" if name else location)

        walk(self.fgraph, prefix)
        self.node_locations = locations
        with (output / f"{prefix}.txt").open("w") as stream:
            debugprint(
                self.fgraph,
                print_type=True,
                print_inner_graphs=True,
                print_profile=False,
                file=stream,
            )
        sources = []
        seen = set()
        dispatchers = []

        def collect_source(function):
            if id(function) in seen:
                return
            seen.add(id(function))
            if isinstance(function, CPUDispatcher):
                dispatchers.append(function)
            function = getattr(function, "py_func", function)
            source = getattr(function, "__source__", None)
            if source is None:
                try:
                    source = inspect.getsource(function)
                except (OSError, TypeError):
                    source = "# Source unavailable"
            sources.append(source)
            # PyTensor's cache wrappers delegate to the generated graph here.
            inner = getattr(function, "__globals__", {}).get("jitable_func")
            if callable(inner):
                collect_source(inner)

        collect_source(dispatcher)
        for function in self.node_functions.values():
            collect_source(function)
        (output / f"{prefix}_generated.py").write_text("\n\n".join(sources) + "\n")

        symbols_by_name = {}
        definitions = set()
        object_index = 0
        for current in dispatchers:
            for signature, compiled in current.overloads.items():
                library = compiled.library
                try:
                    blob = _compiled_object(current, signature, compiled)
                except RuntimeError:
                    if current is dispatcher:
                        raise
                    continue
                object_name = f"{prefix}_{object_index}.o"
                object_index += 1
                (output / object_name).write_bytes(blob)
                for symbol in backend.functions(blob):
                    name = symbol["name"]
                    definition = name == compiled.fndesc.llvm_func_name
                    if name in definitions or (
                        name in symbols_by_name and not definition
                    ):
                        continue
                    try:
                        address = library.get_pointer_to_function(name)
                    except (NameError, RuntimeError):
                        continue
                    if address:
                        symbols_by_name[name] = dict(
                            symbol, address=address, object=object_name, graph=prefix
                        )
                        if definition:
                            # Linked helpers can be optimized differently in
                            # each object. Their own specialization defines
                            # the layout at the resolved execution address.
                            definitions.add(name)
        symbols = list(symbols_by_name.values())

        if not symbols:
            raise RuntimeError("No loaded Numba function addresses could be resolved")
        active_names = {symbol["name"] for symbol in symbols}
        mapping = defaultdict(set)
        errors = []
        cache_names = {}
        cache_externals = {}
        cache_objects = set()
        cache_failed = set()
        external_calls = defaultdict(set)
        missing_call_metadata = set()
        for node, function in self.node_functions.items():
            if node not in locations:
                continue
            if not isinstance(function, CPUDispatcher):
                missing_call_metadata.update(locations[node])
                continue
            names = set()
            externals = set()
            inspected = False
            for signature, compiled in function.overloads.items():
                names.add(compiled.fndesc.llvm_func_name)
                try:
                    blob = _compiled_object(function, signature, compiled)
                    names.update(row["name"] for row in backend.functions(blob))
                    externals.update(backend.external_symbols(blob))
                    inspected = True
                except (ValueError, RuntimeError):
                    pass

            # A cached graph can load without specializing its Op dispatchers.
            # Match their retained names against symbols in the *loaded* object.
            cachefile = getattr(getattr(function, "_cache", None), "_cache_file", None)
            if cachefile is not None:
                key = str(cachefile._index_path)
                if key not in cache_names:
                    cache_names[key] = set()
                    cache_externals[key] = set()
                    try:
                        for filename in cachefile._load_index().values():
                            data = cachefile._load_data(filename)
                            cache_names[key].add(data[1].llvm_func_name)
                            library = data[0]
                            if library[1] == "object":
                                cache_objects.add(key)
                                cache_externals[key].update(
                                    backend.external_symbols(library[2][0])
                                )
                                cache_names[key].update(
                                    row["name"]
                                    for row in backend.functions(library[2][0])
                                )
                    except Exception as error:
                        cache_failed.add(key)
                        errors.append(dict(cache=key, error=repr(error)))
                names.update(cache_names[key])
                externals.update(cache_externals[key])
                inspected = (
                    inspected or key in cache_objects
                ) and key not in cache_failed
            if not inspected:
                missing_call_metadata.update(locations[node])
            for name in externals:
                external_calls[name].update(locations[node])
            for name in names & active_names:
                # NRT functions are process-wide runtime helpers. Membership in
                # a node's library cannot identify their callers.
                if name.startswith("NRT_"):
                    continue
                mapping[name].update(locations[node])

        # Inner nodes are more precise than the enclosing OpFromGraph/Scan.
        mapping = _prefer_inner_locations(mapping)
        self.external_calls = _prefer_inner_locations(external_calls)
        if missing_call_metadata:
            # Empty is not an ELF symbol name: reserve it for opaque callers.
            self.external_calls[""] = sorted(missing_call_metadata)
        return symbols, mapping, nodes, errors


def _summarize(lines, symbols, mapping, nodes=()):
    # LLVM aliases can name the same range. Keep all candidate nodes and count
    # each sampled period once.
    aliases = defaultdict(list)
    objects = {}
    for symbol in symbols:
        aliases[symbol["address"], symbol["size"]].append(symbol["name"])
        objects.setdefault((symbol["address"], symbol["size"]), symbol)
    ranges = sorted(aliases)
    addresses = [address for address, _ in ranges]
    weights, counts = Counter(), Counter()
    descriptions = {}
    offsets = defaultdict(Counter)
    coverage = {
        name: dict(sample_weight=0, samples=0)
        for name in (
            "one node",
            "shared / ambiguous",
            "native only",
            "library / unmapped",
        )
    }
    ops = {node["location"]: node["op"] for node in nodes}
    total = 0
    malformed = 0
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^\s*(\d+)\s+([0-9a-fA-F]+)\s+(.*?)\s+\((.*)\)\s*$", line)
        if match is None:
            malformed += 1
            continue
        period, ip = int(match[1]), int(match[2], 16)
        total += period
        index = bisect_right(addresses, ip) - 1
        if index >= 0 and ip < ranges[index][0] + ranges[index][1]:
            key = ranges[index]
            names = aliases[key]
            locations = sorted({loc for name in names for loc in mapping.get(name, ())})
            attribution = (
                "one node"
                if len(locations) == 1
                else "shared / ambiguous"
                if locations
                else "native only"
            )
            description = dict(
                name=names[0],
                address=ranges[index][0],
                size=ranges[index][1],
                locations=locations,
                ops=[ops.get(loc, "") for loc in locations],
                attribution=attribution,
            )
            symbol = objects[key]
            if "object" in symbol:
                description.update(
                    object=symbol["object"],
                    object_offset=symbol["offset"],
                    section=symbol["section"],
                )
                offsets[key][symbol["offset"] + ip - symbol["address"]] += period
        else:
            key = (match[3], match[4])
            description = dict(
                name=match[3],
                library=match[4],
                locations=[],
                ops=[],
                attribution="library / unmapped",
            )
        descriptions[key] = description
        coverage[description["attribution"]]["sample_weight"] += period
        coverage[description["attribution"]]["samples"] += 1
        weights[key] += period
        counts[key] += 1
    if malformed:
        raise RuntimeError(
            f"Could not parse {malformed} native samples; see samples.txt"
        )
    if not total:
        raise RuntimeError("No usable samples; run a longer workload")
    for row in coverage.values():
        row["percent_self"] = 100 * row["sample_weight"] / total
    return dict(
        total_sample_weight=total,
        total_samples=sum(counts.values()),
        coverage=coverage,
        captured_jit_percent=100 - coverage["library / unmapped"]["percent_self"],
        functions=[
            dict(
                descriptions[key],
                sample_weight=weight,
                percent_self=100 * weight / total,
                samples=counts[key],
                sampled_offsets=[
                    dict(object_offset=offset, sample_weight=weight)
                    for offset, weight in sorted(offsets[key].items())
                ],
            )
            for key, weight in weights.most_common()
        ],
    )


def _callchains(lines):
    period = None
    frames = []
    for line in lines:
        if not line.strip():
            if period is not None:
                yield period, frames
                period, frames = None, []
        elif line.lstrip().startswith("#"):
            continue
        elif period is None and line.strip().isdigit():
            period = int(line.strip())
        elif period is not None and (
            match := re.match(r"^\s*([0-9a-fA-F]+)\s+(.*?)\s+\((.*)\)\s*$", line)
        ):
            frames.append((int(match[1], 16), match[2], match[3]))
        else:
            raise RuntimeError("Could not parse native call chains; see callchains.txt")
    if period is not None:
        yield period, frames


def _attribute_callchains(
    report,
    flat_lines,
    stack_lines,
    symbols,
    mapping,
    external_calls,
    nodes,
    call_sites=None,
):
    aliases = {}
    for symbol in symbols:
        entry = aliases.setdefault(
            (symbol["address"], symbol["size"]), dict(names=[], graphs=set())
        )
        entry["names"].append(symbol["name"])
        if "graph" in symbol:
            entry["graphs"].add(symbol["graph"])
    ranges = sorted(aliases)
    addresses = [address for address, _ in ranges]

    def locate(ip):
        i = bisect_right(addresses, ip) - 1
        return ranges[i] if i >= 0 and ip < ranges[i][0] + ranges[i][1] else None

    def candidates(key):
        return sorted(
            {loc for name in aliases[key]["names"] for loc in mapping.get(name, ())}
        )

    source_rows = {
        (row["address"], row["size"])
        if "address" in row
        else (row["name"], row["library"]): row
        for row in report["functions"]
    }
    weights, counts = Counter(), Counter()
    caller_locations = defaultdict(set)
    caller_weights = Counter()
    total, n_samples = 0, 0
    mismatched_stacks = 0
    flat_lines = (
        line
        for line in flat_lines
        if line.strip() and not line.lstrip().startswith("#")
    )
    for flat, (period, frames) in zip(
        flat_lines, _callchains(stack_lines), strict=True
    ):
        match = re.match(r"^\s*(\d+)\s+([0-9a-fA-F]+)\s+(.*?)\s+\((.*)\)\s*$", flat)
        if match is None or int(match[1]) != period:
            raise RuntimeError("Flat samples and call chains disagree")
        ip = int(match[2], 16)
        if frames and frames[0][0] != ip:
            # Keep the trustworthy flat sample; do not use a malformed unwind.
            frames = []
            mismatched_stacks += 1
        leaf = locate(ip)
        locations = candidates(leaf) if leaf else []
        via = "sampled function" if locations else "unassigned"
        if not locations:
            for i, (caller_ip, _, _) in enumerate(frames[1:], start=1):
                # Caller IPs are return addresses, potentially just past a range.
                key = locate(caller_ip - 1)
                if key is None:
                    continue
                locations = candidates(key)
                # An inlined Op may have no separate native range. Match the
                # library call to imports of nodes inside this caller before
                # falling back to the enclosing graph Op.
                target = (call_sites or {}).get(caller_ip) or frames[i - 1][1].split(
                    "@", 1
                )[0]
                graphs = aliases[key]["graphs"]

                def inside_caller(loc):
                    return loc.split("/", 1)[0] in graphs and (
                        not locations
                        or any(
                            loc == scope or loc.startswith(scope + "/")
                            for scope in locations
                        )
                    )

                references = {
                    loc for loc in external_calls.get(target, ()) if inside_caller(loc)
                }
                if references:
                    locations = sorted(
                        references
                        | {
                            loc
                            for loc in external_calls.get("", ())
                            if inside_caller(loc)
                        }
                    )
                    via = "external call references"
                    break
                if locations:
                    via = "caller stack"
                    break
        group = (tuple(locations), via)
        weights[group] += period
        counts[group] += 1
        total += period
        n_samples += 1
        source_key = leaf if leaf else (match[3], match[4])
        source = source_rows[source_key]
        if period and locations and not source["locations"]:
            caller_locations[source_key].update(locations)
            caller_weights[source_key] += period
    if total != report["total_sample_weight"] or n_samples != report["total_samples"]:
        raise RuntimeError("Call-chain totals disagree with the flat profile")

    ops = {node["location"]: node["op"] for node in nodes}
    for key, locations in caller_locations.items():
        source_rows[key].update(
            caller_locations=sorted(locations),
            caller_sample_weight=caller_weights[key],
            caller_ops=[ops[loc] for loc in sorted(locations)],
        )
    coverage = {
        name: dict(sample_weight=0, samples=0)
        for name in ("one node", "shared / ambiguous", "unassigned")
    }
    attributed = []
    for (locations, via), weight in weights.most_common():
        category = (
            "shared / ambiguous"
            if len(locations) > 1
            else "one node"
            if locations
            else "unassigned"
        )
        count = counts[locations, via]
        coverage[category]["sample_weight"] += weight
        coverage[category]["samples"] += count
        attributed.append(
            dict(
                locations=list(locations),
                ops=[ops[loc] for loc in locations],
                via=via,
                attribution=category,
                sample_weight=weight,
                samples=count,
                percent_self=100 * weight / total,
            )
        )
    for row in coverage.values():
        row["percent_self"] = 100 * row["sample_weight"] / total
    report.update(
        node_samples=attributed,
        node_coverage=coverage,
        node_attributed_percent=100 - coverage["unassigned"]["percent_self"],
        callchain_mismatches=mismatched_stacks,
    )


def _annotate_disassembly(report, output, n_hot_functions):
    selected = [
        row for row in report["functions"] if "object" in row and row["sample_weight"]
    ][:n_hot_functions]
    for i, row in enumerate(selected, start=1):
        result = subprocess.run(
            [
                "objdump",
                "-d",
                "--no-show-raw-insn",
                "--disassemble=" + row["name"],
                str(output / row["object"]),
            ],
            capture_output=True,
            text=True,
        )
        (output / f"hot_{i}.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"objdump failed; see {output / f'hot_{i}.log'}")
        instructions = [
            (int(match[1], 16), match[2])
            for line in result.stdout.splitlines()
            if (match := re.match(r"^\s*([0-9a-fA-F]+):\s+(.*)", line))
        ]
        if not instructions:
            raise RuntimeError(f"objdump returned no instructions for {row['name']}")
        starts = [offset for offset, _ in instructions]
        weights = Counter()
        for sample in row["sampled_offsets"]:
            index = bisect_right(starts, sample["object_offset"]) - 1
            if index < 0:
                raise RuntimeError("Sample precedes the disassembled function")
            weights[index] += sample["sample_weight"]
        row["hot_instructions"] = [
            dict(
                object_offset=hex(instructions[index][0]),
                instruction=instructions[index][1],
                sample_weight=weight,
                percent_self=100 * weight / report["total_sample_weight"],
                percent_function=100 * weight / row["sample_weight"],
            )
            for index, weight in weights.most_common(8)
        ]
        row["disassembly_file"] = f"hot_{i}.asm"
        header = [
            f"# {row['name']}",
            "# Candidate nodes: " + (", ".join(row["locations"]) or "unmapped"),
            "# Hot instruction locations (sampling can skid; not exact instruction timings):",
        ]
        header.extend(
            f"# {hot['percent_self']:6.2f}% total, {hot['percent_function']:6.2f}% function "
            f"at {hot['object_offset']}: {hot['instruction']}"
            for hot in row["hot_instructions"]
        )
        (output / row["disassembly_file"]).write_text(
            "\n".join(header) + "\n\n" + result.stdout
        )


def _save_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "report.json")


def _report_backend(
    report, backend: NativeSamplingBackend | None
) -> NativeSamplingBackend:
    recorded = report.get("backend", _LINUX_PERF_BACKEND.name)
    if backend is None:
        backend = {
            "linux-perf": _LINUX_PERF_BACKEND,
            "linux-perf-dwarf": LinuxDwarfBackend(),
        }.get(recorded)
        if backend is None:
            raise ValueError(
                f"This report uses backend {recorded!r}; pass its backend instance to rebuild_report(..., backend=...)."
            )
    if recorded != backend.name:
        raise ValueError(
            f"This report uses backend {recorded!r}; pass its backend instance "
            "to rebuild_report(..., backend=...)."
        )
    return backend


def _build_report(output, report, n_hot_functions, backend=None):
    backend = _report_backend(report, backend)
    symbols = json.loads((output / "symbols.json").read_text())
    mapping = json.loads((output / "native_to_nodes.json").read_text())
    nodes = json.loads((output / "nodes.json").read_text())
    with (output / "samples.txt").open() as stream:
        report.update(_summarize(stream, symbols, mapping, nodes))
    if (output / "callchains.txt").exists():
        external_calls = json.loads((output / "external_calls.json").read_text())
        call_sites = (
            {
                int(pc): symbol
                for pc, symbol in json.loads(
                    (output / "call_sites.json").read_text()
                ).items()
            }
            if (output / "call_sites.json").exists()
            else {}
        )
        with (
            (output / "samples.txt").open() as flat,
            (output / "callchains.txt").open() as stacks,
        ):
            _attribute_callchains(
                report,
                flat,
                stacks,
                symbols,
                mapping,
                external_calls,
                nodes,
                call_sites,
            )
    backend.annotate(report, output, n_hot_functions)
    report.update(status="complete", n_hot_functions=n_hot_functions)
    with (output / "report.txt").open("w") as stream:
        _print_native_summary(report, stream)
    _save_report(output, report)
    return report


def rebuild_report(
    output, *, n_hot_functions=None, backend: NativeSamplingBackend | None = None
):
    """Rebuild saved reports, supplying the original backend for custom tooling."""
    output = Path(output).resolve()
    report = json.loads((output / "report.json").read_text())
    report["output"] = str(output)
    if n_hot_functions is None:
        n_hot_functions = report.get("n_hot_functions", 5)
    _check_hot_functions(n_hot_functions)
    return _build_report(output, report, n_hot_functions, backend=backend)


def _check_hot_functions(n_hot_functions):
    if (
        not isinstance(n_hot_functions, int)
        or isinstance(n_hot_functions, bool)
        or n_hot_functions < 0
    ):
        raise ValueError("n_hot_functions must be a nonnegative integer")


class _PerfRecorder:
    """Start disabled and acknowledge each sampling boundary via perf control."""

    def __init__(
        self, output, frequency, event, call_graph=True, *, call_graph_mode="fp"
    ):
        self.output = output
        self.frequency = frequency
        self.event = event
        self.call_graph = call_graph
        self.call_graph_mode = call_graph_mode
        self.process = None
        self.fds = []
        self.log = None

    def start(self):
        control_read, self.control_write = os.pipe()
        self.fds = [control_read, self.control_write]
        self.ack_read, ack_write = os.pipe()
        self.fds.extend((self.ack_read, ack_write))
        self.log = (self.output / "perf.log").open("w")
        command = [
            "perf",
            "record",
            "--delay=-1",
            f"--control=fd:{control_read},{ack_write}",
            "--event",
            self.event,
            "--freq",
            str(self.frequency),
            "--pid",
            str(os.getpid()),
            "--output",
            str(self.output / "perf.data"),
        ]
        if self.call_graph:
            command.extend(
                (
                    "--call-graph",
                    self.call_graph_mode,
                    "--user-callchains",
                    "--all-user",
                )
            )
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            pass_fds=(control_read, ack_write),
        )
        for fd in (control_read, ack_write):
            os.close(fd)
            self.fds.remove(fd)
        self.control("enable")

    def control(self, command):
        try:
            os.write(self.control_write, (command + "\n").encode())
        except BrokenPipeError as error:
            raise RuntimeError(
                f"perf control closed; see {self.output / 'perf.log'}"
            ) from error
        deadline = time.monotonic() + 10
        response = b""
        while b"\n" not in response:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"perf record failed; see {self.output / 'perf.log'}"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for perf to {command}")
            if select.select([self.ack_read], [], [], 0.1)[0]:
                chunk = os.read(self.ack_read, 64)
                if not chunk:
                    raise RuntimeError(
                        f"perf control closed; see {self.output / 'perf.log'}"
                    )
                response += chunk
        if response.strip(b"\0") != b"ack\n":
            raise RuntimeError(f"Unexpected perf acknowledgement: {response!r}")

    def stop(self):
        self.control("disable")
        self.process.send_signal(signal.SIGINT)
        self.process.wait(timeout=10)
        if self.process.returncode not in (0, -signal.SIGINT):
            raise RuntimeError(f"perf record failed; see {self.output / 'perf.log'}")

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            for fd in self.fds:
                os.close(fd)
            if self.log is not None:
                self.log.close()


class LinuxPerfBackend:
    """Linux perf with frame-pointer callers; no native unwinder helper required."""

    name = "linux-perf"
    default_event = "cycles:u"
    call_graph = "fp"

    functions = staticmethod(_elf_functions)
    external_symbols = staticmethod(_external_symbols)

    @staticmethod
    def _check_assembly(n_hot_functions):
        if n_hot_functions and shutil.which("objdump") is None:
            raise RuntimeError(
                "Annotated assembly requires objdump; set n_hot_functions=0 to disable it"
            )

    def validate(self, *, event, n_hot_functions, call_graph):
        if sys.platform != "linux":
            raise RuntimeError("Native sampling requires Linux perf")
        if sys.byteorder != "little" or struct.calcsize("P") != 8:
            raise RuntimeError(
                "Native sampling requires a little-endian 64-bit process"
            )
        if any(char in event for char in ",{}"):
            raise ValueError(
                "event must name a single perf event, without groups or commas"
            )
        if shutil.which("perf") is None:
            raise RuntimeError("Native sampling requires the perf executable")
        self._check_assembly(n_hot_functions)

    def recorder(self, output, frequency, event, call_graph):
        return _PerfRecorder(output, frequency, event, call_graph)

    def decode(self, output, *, call_graph):
        with (output / "samples.txt").open("w") as stream:
            result = subprocess.run(
                [
                    "perf",
                    "script",
                    "--hide-call-graph",
                    "--input",
                    str(output / "perf.data"),
                    "--fields",
                    "ip,period,sym,dso",
                ],
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
            )
        (output / "perf_script.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"perf script failed; see {output / 'perf_script.log'}")
        if call_graph:
            with (output / "callchains.txt").open("w") as stream:
                result = subprocess.run(
                    [
                        "perf",
                        "script",
                        "--input",
                        str(output / "perf.data"),
                        "--fields",
                        "ip,period,sym,dso",
                        "--max-stack",
                        "64",
                    ],
                    stdout=stream,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            (output / "callchains.log").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(
                    f"perf call-chain decoding failed; see {output / 'callchains.log'}"
                )
        with (output / "perf.txt").open("w") as stream:
            result = subprocess.run(
                [
                    "perf",
                    "report",
                    "--stdio",
                    "--no-children",
                    "--percent-limit",
                    "0.1",
                    "--sort",
                    "dso,symbol",
                    "--input",
                    str(output / "perf.data"),
                ],
                stdout=stream,
                stderr=subprocess.PIPE,
                text=True,
            )
        (output / "perf_report.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"perf report failed; see {output / 'perf_report.log'}")

    def annotate(self, report, output, n_hot_functions):
        self._check_assembly(n_hot_functions)
        _annotate_disassembly(report, output, n_hot_functions)


class LinuxDwarfBackend(LinuxPerfBackend):
    """Linux x86-64 DWARF recovery through a cached libdwfl helper."""

    name = "linux-perf-dwarf"
    call_graph = "dwarf"

    def __init__(self, *, stack_bytes: int = 8192, keep_raw: bool = False):
        self.stack_bytes = stack_bytes
        self.keep_raw = keep_raw

    def validate(self, *, event, n_hot_functions, call_graph):
        super().validate(
            event=event,
            n_hot_functions=0 if call_graph else n_hot_functions,
            call_graph=call_graph,
        )
        if platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError("This DWARF recovery backend supports Linux x86-64")
        if (
            not isinstance(self.stack_bytes, int)
            or not 1024 <= self.stack_bytes <= 32768
            or self.stack_bytes % 8
        ):
            raise ValueError(
                "stack_bytes must be a multiple of 8 between 1024 and 32768"
            )
        if call_graph:
            if shutil.which("objdump") is None:
                raise RuntimeError("DWARF caller recovery requires objdump")
            from pytensor.link.numba._profiling_dwarf import load_unwinder

            load_unwinder()

    def recorder(self, output, frequency, event, call_graph):
        return _PerfRecorder(
            output,
            frequency,
            event,
            call_graph,
            call_graph_mode=f"dwarf,{self.stack_bytes}",
        )

    def decode(self, output, *, call_graph):
        if not call_graph:
            return super().decode(output, call_graph=False)
        from pytensor.link.numba._profiling_dwarf import decode

        metadata = decode(output)
        metadata.update(stack_bytes=self.stack_bytes, raw_data_retained=self.keep_raw)
        if not self.keep_raw:
            (output / "perf.data").unlink()
        return metadata


_LINUX_PERF_BACKEND = LinuxPerfBackend()


def _default_backend(call_graph):
    if (
        call_graph
        and sys.platform == "linux"
        and platform.machine().lower() in ("x86_64", "amd64")
    ):
        return LinuxDwarfBackend()
    return _LINUX_PERF_BACKEND


@contextmanager
def sample(
    profile,
    *,
    output,
    frequency,
    event,
    n_hot_functions,
    call_graph,
    backend: NativeSamplingBackend | None = None,
):
    backend = _default_backend(call_graph) if backend is None else backend
    event = backend.default_event if event is None else event
    if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    if not isinstance(event, str) or not event or "\n" in event:
        raise ValueError("event must be a nonempty single-line string")
    _check_hot_functions(n_hot_functions)
    if not isinstance(call_graph, bool):
        raise TypeError("call_graph must be a bool")
    backend.validate(
        event=event, n_hot_functions=n_hot_functions, call_graph=call_graph
    )
    if not _sampling_lock.acquire(blocking=False):
        raise RuntimeError("A native sampling window is already active in this process")
    recorder = None
    report = dict(
        schema_version=2,
        status="capturing",
        event=event,
        frequency=frequency,
        numba_version=numba.__version__,
        pid=os.getpid(),
        n_hot_functions=n_hot_functions,
        backend=backend.name,
        call_graph=backend.call_graph if call_graph else None,
    )
    try:
        if output is None:
            output = Path(tempfile.mkdtemp(prefix="pytensor-native-profile-"))
        else:
            output = Path(output).resolve()
            output.mkdir(parents=True, exist_ok=False)
        report["output"] = str(output)

        def save():
            _save_report(output, report)

        save()
        try:
            symbols, mapping, nodes, errors = [], {}, [], []
            external_calls = {}
            profilers = tuple(profile._native_profilers)
            signatures = [tuple(p.dispatcher.signatures) for p in profilers]
            for i, profiler in enumerate(profilers):
                current_symbols, current_mapping, current_nodes, current_errors = (
                    profiler.capture(output, f"graph{i}", backend=backend)
                )
                symbols.extend(current_symbols)
                nodes.extend(current_nodes)
                errors.extend(current_errors)
                for name, locations in current_mapping.items():
                    mapping[name] = sorted(set(mapping.get(name, ())) | set(locations))
                for name, locations in profiler.external_calls.items():
                    external_calls[name] = sorted(
                        set(external_calls.get(name, ())) | set(locations)
                    )
            for filename, value in (
                ("symbols.json", symbols),
                ("nodes.json", nodes),
                ("native_to_nodes.json", mapping),
                ("external_calls.json", external_calls),
            ):
                (output / filename).write_text(json.dumps(value, indent=2) + "\n")
            report["mapping_errors"] = errors
            report["native_functions"] = len(symbols)
            report["associated_functions"] = len(mapping)
            recorder = backend.recorder(output, frequency, event, call_graph)
            report["status"] = "recording"
            save()
            recorder.start()
            start = time.perf_counter()
            try:
                yield report
            finally:
                report["wall_seconds"] = time.perf_counter() - start
                # Do not obscure a workload exception with a second perf error.
                workload_error = sys.exc_info()[1]
                try:
                    recorder.stop()
                except BaseException as error:
                    if workload_error is None:
                        raise
                    workload_error.add_note(f"Stopping perf also failed: {error!r}")
            if tuple(profile._native_profilers) != profilers or any(
                tuple(p.dispatcher.signatures) != before
                for p, before in zip(profilers, signatures, strict=True)
            ):
                raise RuntimeError(
                    "Numba compiled new functions during sampling. Warm all input "
                    "signatures before starting a new sampling window."
                )
            metadata = backend.decode(output, call_graph=call_graph)
            if metadata:
                report.update(metadata)
            _build_report(output, report, n_hot_functions, backend=backend)
            profile.native_samples.append(report)
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            save()
            raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            _sampling_lock.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=rebuild_report.__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hot-functions", type=int, default=None)
    args = parser.parse_args()
    rebuild_report(args.output, n_hot_functions=args.hot_functions)
    sys.stdout.write((args.output / "report.txt").read_text())
