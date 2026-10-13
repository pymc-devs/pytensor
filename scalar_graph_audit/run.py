"""Interleaved native-call benchmarks and LLVM evidence for PR #2310.

Run with single-threaded BLAS/Numba, numba__cache=False and a pinned CPU.
--count measures allocations only; never use an instrumented run for timings.
The upstream arm loads upstream's actual fusion pass and dispatch from an archive.
"""

# ruff: noqa: T201

import argparse
import ast
import ctypes
import json
import os
import platform
import re
import statistics
import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

import llvmlite
import numba
import numpy as np
from numba.core.runtime import nrt

import pytensor
from pytensor.compile.mode import get_mode
from pytensor.link.numba.dispatch import basic, elemwise, vectorize_codegen
from pytensor.tensor.rewriting import fused_elemwise, scalarize


REBASED_FUNCTIONS = None


def source_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def upstream_functions(root):
    rewrites = source_functions(
        root / "pytensor/tensor/rewriting/fused_elemwise.py",
        ["apply"],
        dict(vars(fused_elemwise)),
        "FuseElemwise",
    )
    dispatch = source_functions(
        root / "pytensor/link/numba/dispatch/elemwise.py",
        ["_build_reduce_impl_src", "numba_funcify_FusedElemwise"],
        dict(vars(elemwise)),
    )
    return rewrites["apply"], dispatch["numba_funcify_FusedElemwise"]


def fusion_without_scalar_outputs():
    import inspect
    import textwrap

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(fused_elemwise.FuseElemwise.apply))
    )
    matches = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "reduced.type.ndim == 0"
        ):
            node.test = ast.Constant(False)
            matches += 1
    assert matches == 1
    namespace = dict(vars(fused_elemwise))
    exec(
        compile(ast.fix_missing_locations(tree), "<no-scalar-outputs>", "exec"),
        namespace,
    )
    return namespace["apply"]


def variant_context(arm, upstream):
    stack = ExitStack()
    mode = get_mode("NUMBA")
    if arm == "rebased":
        for name, fn in REBASED_FUNCTIONS.items():
            value = fn if name == "apply" else staticmethod(fn)
            stack.enter_context(patch.object(scalarize.Scalarize, name, value))
    elif arm == "no_unit_inputs":
        for name in ("_scalarize_elemwise", "_rescalarize"):
            fn = REBASED_FUNCTIONS[name]
            namespace = fn.__globals__ | {"_scalar_behind": scalarize._scalar_behind}
            original = FunctionType(fn.__code__, namespace, name, fn.__defaults__)
            stack.enter_context(
                patch.object(scalarize.Scalarize, name, staticmethod(original))
            )
    elif arm == "single_view_boundary":
        original = REBASED_FUNCTIONS["apply"].__globals__["_scalar_behind"]
        stack.enter_context(patch.object(scalarize, "_scalar_behind", original))
    elif arm == "upstream":
        apply, dispatch = upstream
        stack.enter_context(patch.object(fused_elemwise.FuseElemwise, "apply", apply))
        # Config disables cache, so the legacy dispatcher is the active one.
        previous = basic.numba_funcify.dispatch(fused_elemwise.FusedElemwise)
        basic.numba_funcify.register(fused_elemwise.FusedElemwise)(
            lambda *a, **kw: dispatch(*a, **kw)[0]
        )
        stack.callback(
            basic.numba_funcify.register(fused_elemwise.FusedElemwise), previous
        )
        mode = mode.excluding("scalarize", "wrap_scalar_careduce")
    elif arm == "no_subtensor":
        stack.enter_context(
            patch.object(
                scalarize.Scalarize, "_emit_producer", staticmethod(lambda *a: None)
            )
        )
    elif arm == "no_join":
        stack.enter_context(
            patch.object(
                scalarize.Scalarize, "_swallow_join", staticmethod(lambda *a: None)
            )
        )
    elif arm == "no_mixed":
        stack.enter_context(
            patch.object(
                scalarize.Scalarize,
                "_scalarize_elemwise",
                staticmethod(lambda *a: None),
            )
        )
        stack.enter_context(
            patch.object(
                scalarize.Scalarize, "_rescalarize", staticmethod(lambda *a: None)
            )
        )
    elif arm == "no_post":
        mode = mode.excluding("scalarize_post_fusion")
    elif arm == "no_careduce":
        mode = mode.excluding("wrap_scalar_careduce")
    elif arm == "no_lift":
        mode = mode.excluding("local_elemwise_lift_scalars")
    elif arm == "scalar_clients":
        stack.enter_context(
            patch.object(
                scalarize,
                "_SCALAR_CONSUMERS",
                (*scalarize._SCALAR_CONSUMERS, scalarize.ScalarFromTensor),
            )
        )
    elif arm == "wide_multi":
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(scalarize.Scalarize.apply))
        assert "elif single or scalar_computation:" in src
        src = src.replace("elif single or scalar_computation:", "else:")
        namespace = dict(vars(scalarize))
        exec(compile(src, "<wide-multi-output>", "exec"), namespace)
        stack.enter_context(
            patch.object(scalarize.Scalarize, "apply", namespace["apply"])
        )
    elif arm == "heap":
        make_outputs = vectorize_codegen.make_outputs

        def heap(*a, **kw):
            kw["stack_outputs"] = frozenset()
            return make_outputs(*a, **kw)

        stack.enter_context(patch.object(vectorize_codegen, "make_outputs", heap))
    elif arm == "no_scalar_outputs":
        stack.enter_context(
            patch.object(
                fused_elemwise.FuseElemwise, "apply", fusion_without_scalar_outputs()
            )
        )
    elif arm in ("unit_inputs", "unit_cleanup"):
        mixed = scalarize.Scalarize._scalarize_elemwise
        rescalarize = scalarize.Scalarize._rescalarize

        def lift_unit(node, fallback, scalars=None):
            if not all(all(out.type.broadcastable) for out in node.outputs):
                return fallback(node) if scalars is None else fallback(node, scalars)
            if not all(
                isinstance(inp.type, scalarize.ScalarType)
                or all(inp.type.broadcastable)
                for inp in node.inputs
            ):
                return fallback(node) if scalars is None else fallback(node, scalars)
            inner = node
            if isinstance(node.op, scalarize.ScalarizedElemwise):
                [inner] = [
                    n
                    for n in node.op.fgraph.apply_nodes
                    if isinstance(n.op, scalarize.Elemwise)
                ]
            values = [
                inp
                if isinstance(inp.type, scalarize.ScalarType)
                else (
                    scalarize.scalar_from_tensor(inp.squeeze())
                    if scalarize._scalar_behind(inp) is None
                    else scalarize._scalar_behind(inp)
                )
                for inp in node.inputs
            ]
            return scalarize.Scalarize._emit_scalar_elemwise(
                inner.op.scalar_op, node.outputs, values
            )

        stack.enter_context(
            patch.object(
                scalarize.Scalarize,
                "_scalarize_elemwise",
                staticmethod(lambda node, scalars: lift_unit(node, mixed, scalars)),
            )
        )
        stack.enter_context(
            patch.object(
                scalarize.Scalarize,
                "_rescalarize",
                staticmethod(lambda node: lift_unit(node, rescalarize)),
            )
        )
        if arm == "unit_cleanup":
            original_apply = scalarize.Scalarize.apply

            def cleanup(self, fgraph):
                original_apply(self, fgraph)
                for node in fgraph.toposort():
                    if not isinstance(node.op, fused_elemwise.ScalarFromTensor):
                        continue
                    value = node.inputs[0]
                    while value.owner and isinstance(
                        value.owner.op, scalarize.DimShuffle
                    ):
                        value = value.owner.inputs[0]
                    if value.owner and isinstance(
                        value.owner.op, scalarize.TensorFromScalar
                    ):
                        fgraph.replace(
                            node.outputs[0],
                            value.owner.inputs[0],
                            reason="scalar_view_cleanup",
                        )

            stack.enter_context(patch.object(scalarize.Scalarize, "apply", cleanup))
    elif arm == "singleton":
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(fused_elemwise.FuseElemwise.apply))
        src = src.replace(
            "if reduced.type.ndim == 0:", "if all(reduced.type.broadcastable):"
        )
        src = src.replace(
            "reduced = scalar_from_tensor(reduced)",
            "reduced = scalar_from_tensor(reduced.squeeze())",
        )
        src = src.replace(
            "new_out = tensor_from_scalar(new_out)",
            "new_out = tensor_from_scalar(new_out).dimshuffle(['x'] * reduced_outputs[out_idx].outputs[0].ndim)",
        )
        namespace = dict(vars(fused_elemwise))
        exec(compile(src, "<singleton-fusion>", "exec"), namespace)
        stack.enter_context(
            patch.object(fused_elemwise.FuseElemwise, "apply", namespace["apply"])
        )
        src = textwrap.dedent(inspect.getsource(elemwise.numba_funcify_FusedElemwise))
        tree = ast.parse(src)
        tree.body[0].decorator_list = []
        src = ast.unparse(tree).replace(
            "on_stack = scalar_out and not kept_axes", "on_stack = scalar_out"
        )
        dispatch_globals = dict(vars(elemwise))
        build = elemwise._build_reduce_impl_src
        dispatch_globals["_build_reduce_impl_src"] = lambda nout, specs, *a: build(
            nout,
            [None if s is None else ((), *s[1:]) if s[3] else s for s in specs],
            *a,
        )
        exec(compile(src, "<singleton-dispatch>", "exec"), dispatch_globals)
        previous = basic.numba_funcify.dispatch(fused_elemwise.FusedElemwise)
        dispatch = dispatch_globals["numba_funcify_FusedElemwise"]
        basic.numba_funcify.register(fused_elemwise.FusedElemwise)(
            lambda *a, **kw: dispatch(*a, **kw)[0]
        )
        stack.callback(
            basic.numba_funcify.register(fused_elemwise.FusedElemwise), previous
        )
    elif arm == "reshape":
        build = elemwise._build_reduce_impl_src

        def reshape(*a):
            src = build(*a)
            return re.sub(
                r"as_strided\((raw(?:\[\d+\])?), shape=(\([^\n]*?\)), strides=\([^\n]*?\)\)",
                r"\1.reshape(\2)",
                src,
            )

        stack.enter_context(patch.object(elemwise, "_build_reduce_impl_src", reshape))
    elif arm != "full":
        raise ValueError(arm)
    return stack, mode


def build_model(name, mode):
    if name == "unit_tensor":
        import pytensor.tensor as pt

        x = pt.vector("parameters", shape=(2,))
        reduced = pt.exp(np.arange(7.0) * x[0]).sum()
        unit = x[1:]
        joined = pt.join(0, reduced[None] * unit, reduced[None] + unit)
        fn = pytensor.function([x], [reduced, joined], mode=mode)
        return fn, np.array([0.2, 1.5])
    if name == "shared_reduce":
        import pytensor.tensor as pt

        x = pt.vector("parameters", shape=(10_000,))
        reduced = pt.sum(x[2:])
        scale = pt.tensor_from_scalar(
            pt.scalar_from_tensor(reduced) + pt.scalar_from_tensor(x[0])
        )
        fn = pytensor.function([x], [reduced, x * scale], mode=mode)
        return fn, np.random.default_rng(42).standard_normal(10_000)
    if name == "wide_multi":
        import pytensor.tensor as pt

        x = pt.vector("parameters", shape=(16,))
        scalar = pt.exp(x[0])
        first, second = x[1:] - scalar, x[1:] + scalar
        fn = pytensor.function([x], [first, pt.join(0, x[0][None], second)], mode=mode)
        return fn, np.random.default_rng(42).standard_normal(16)
    if name == "bare_reduce":
        import pytensor.tensor as pt

        x = pt.vector("parameters", shape=(20,))
        logp = pt.tensor_from_scalar(
            pt.scalar_from_tensor(pt.sum(x[2:])) + pt.scalar_from_tensor(x[0])
        )
        fn = pytensor.function([x], [logp, x * 2], mode=mode)
        return fn, np.random.default_rng(42).standard_normal(20)
    if name.startswith("synthetic"):
        import importlib.util

        source = (
            Path(__file__).resolve().parents[1] / "tests/benchmarks/test_scalar_ops.py"
        )
        spec = importlib.util.spec_from_file_location("scalar_benchmark_graph", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        size = (100, 5) if name == "synthetic_small" else (10_000, 100)
        fn, x, *_ = module._build_fn(mode, *size)
        return fn, x

    import sys

    from pymc.model.transform.optimization import freeze_dims_and_data

    alchemize = Path("/home/ricardo/Documents/alchemize")
    sys.path.insert(0, str(alchemize))
    sys.path.insert(0, str(alchemize / "examples"))
    import bench_logp

    old_cwd = Path.cwd()
    os.chdir(alchemize)
    try:
        model, _ = getattr(bench_logp, f"make_{name}_model")()
    finally:
        os.chdir(old_cwd)
    frozen = freeze_dims_and_data(model)
    wrapper = frozen.logp_dlogp_function(ravel_inputs=True, mode=mode)
    fn = wrapper._pytensor_function
    size = sum(
        np.asarray(frozen.initial_point()[v.name]).size for v in wrapper._grad_vars
    )
    x = np.random.default_rng(42).standard_normal(size)
    fn.trust_input = True
    return fn, x


def llvm_evidence(jit):
    ir = next(iter(jit.inspect_llvm().values()))
    functions = dict(
        re.findall(r"^define[^@]*@([^ (]+)\([^\n]*\n(.*?)^}", ir, re.M | re.S)
    )
    native = [
        name
        for name in functions
        if name.startswith("_ZN8pytensor4link5utils")
        and "numba_funcified_fgraph" in name
    ]
    assert len(native) == 1, native
    visited, pending = set(), native.copy()
    calls = []
    allocation_sites = []
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        body = functions[name]
        found = re.findall(r"\b(?:call|invoke)\b[^\n@]*@([^ (]+)\(", body)
        calls.extend(found)
        pending.extend(c for c in found if c in functions and c not in visited)
        allocation_sites.extend(
            {"function": name, "line": line.strip()}
            for line in body.splitlines()
            if re.search(r"\bcall\b.*@NRT_MemInfo_alloc", line)
        )
    reachable = "\n".join(functions[name] for name in visited)
    counts = {name: calls.count(name) for name in sorted(set(calls))}
    return ir, {
        "reachable_functions": len(visited),
        "alloc_sites": len(allocation_sites),
        "allocation_sites": allocation_sites,
        "nrt_calls": {k: v for k, v in counts.items() if k.startswith("NRT_")},
        "reshape_calls": counts.get("numba_attempt_nocopy_reshape", 0),
        "math_calls": {
            k: v
            for k, v in counts.items()
            if k.startswith("llvm.")
            and any(op in k for op in ("exp", "log", "sqrt", "pow"))
        },
        "vector_arithmetic": len(
            re.findall(r"\b(?:fadd|fmul|fsub|fdiv)\b[^\n]*<\d+ x", reachable)
        ),
    }


def native_runner(output):
    source = output / "runner.c"
    source.write_text("""#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <time.h>
typedef int64_t (*kernel)(uint64_t, double*, double*, double*);
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
double run(kernel fn, uint64_t dim, double* x, double* grad, double* logp, uint64_t count) {
    double start = now();
    for (uint64_t i = 0; i < count; i++) fn(dim, x, grad, logp);
    return (now() - start) * 1e9 / count;
}
""")
    shared = output / "runner.so"
    subprocess.run(
        ["cc", "-O3", "-shared", "-fPIC", str(source), "-o", str(shared)], check=True
    )
    lib = ctypes.CDLL(str(shared))
    lib.run.restype = ctypes.c_double
    lib.run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    return lib


def c_wrapper(jit, vector_first=False):
    sig = numba.types.int64(
        numba.types.uint64, *([numba.types.CPointer(numba.types.float64)] * 3)
    )

    @numba.cfunc(sig)
    def call(dim, x_ptr, grad_ptr, lp_ptr):
        lp, grad = jit(numba.carray(x_ptr, (dim,)))
        numba.carray(lp_ptr, (1,))[0] = lp[0] if vector_first else lp
        numba.carray(grad_ptr, (dim,))[:] = grad
        return 0

    return call


def main():
    global REBASED_FUNCTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream", type=Path, default=Path("/tmp/scalar-upstream-main")
    )
    parser.add_argument("--rebased-ref", default="436776b51")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", default="normal,linreg,hierarchical,synthetic_small,synthetic_large"
    )
    parser.add_argument(
        "--arms",
        default="upstream,full,no_subtensor,no_join,no_mixed,no_post,no_careduce,heap,no_scalar_outputs,reshape",
    )
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--count", action="store_true")
    args = parser.parse_args()
    assert not pytensor.config.numba__cache
    assert bool(numba.config.NRT_STATS) == args.count
    args.output.mkdir(parents=True, exist_ok=True)
    namespace = dict(vars(scalarize))
    snapshot = args.output / "rebased_scalarize.py"
    snapshot.write_text(
        subprocess.check_output(
            [
                "git",
                "show",
                f"{args.rebased_ref}:pytensor/tensor/rewriting/scalarize.py",
            ],
            text=True,
        )
    )
    source_functions(snapshot, ["_scalar_behind"], namespace)
    REBASED_FUNCTIONS = source_functions(
        snapshot,
        ["apply", "_scalarize_elemwise", "_rescalarize"],
        namespace,
        "Scalarize",
    )
    upstream = upstream_functions(args.upstream)
    runner = native_runner(args.output)
    result = {
        "environment": {
            "python": platform.python_version(),
            "numba": numba.__version__,
            "llvmlite": llvmlite.__version__,
            "llvm": llvmlite.binding.llvm_version_info,
            "numpy": np.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "fastmath": pytensor.config.numba__fastmath,
            "allocation_instrumentation": args.count,
            "rounds": args.rounds,
            "rebased_ref": args.rebased_ref,
            "upstream_archive": str(args.upstream),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "load_start": os.getloadavg(),
        },
        "models": {},
    }
    arms = args.arms.split(",")
    for name in args.models.split(","):
        built, rows = {}, {}
        for arm in arms:
            context, mode = variant_context(arm, upstream)
            with context:
                fn, x = build_model(name, mode)
                fn(x)
                jit = fn.vm.jit_fn
                # Compile all variants before restoring the dispatch monkeypatch.
                wrapper = c_wrapper(jit, vector_first=name == "wide_multi")
            ir, evidence = llvm_evidence(jit)
            (args.output / f"{name}-{arm}.ll").write_text(ir)
            (args.output / f"{name}-{arm}.caller.ll").write_text(wrapper.inspect_llvm())
            (args.output / f"{name}-{arm}.graph.txt").write_text(
                pytensor.printing.debugprint(
                    fn, file="str", print_type=True, print_memory_map=True
                )
            )
            built[arm] = (jit, wrapper, x)
            rows[arm] = {"nodes": len(fn.maker.fgraph.apply_nodes), "llvm": evidence}
            print(
                f"built {name}/{arm}: nodes={rows[arm]['nodes']} LLVM_alloc_sites={evidence['alloc_sites']}",
                flush=True,
            )
        # Compare several fresh inputs; copy outputs before the next call can reuse buffers.
        rng = np.random.default_rng(614)
        for delta in [
            np.zeros_like(x),
            *[rng.normal(0, 0.1, x.shape) for _ in range(3)],
        ]:
            ref = None
            for arm in arms:
                value = tuple(
                    np.array(o, copy=True) for o in built[arm][0](built[arm][2] + delta)
                )
                if ref is None:
                    ref = value
                else:
                    for a, b in zip(ref, value, strict=True):
                        np.testing.assert_allclose(
                            a, b, rtol=1e-9, atol=1e-9, err_msg=f"{name}/{arm}"
                        )
        if args.count:
            for arm in arms:
                _, wrapper, x = built[arm]
                gradient, logp = np.zeros_like(x), np.zeros(1)
                runner.run(
                    wrapper.address,
                    x.size,
                    x.ctypes.data,
                    gradient.ctypes.data,
                    logp.ctypes.data,
                    10,
                )
                before = nrt.rtsys.get_allocation_stats()
                runner.run(
                    wrapper.address,
                    x.size,
                    x.ctypes.data,
                    gradient.ctypes.data,
                    logp.ctypes.data,
                    100,
                )
                after = nrt.rtsys.get_allocation_stats()
                rows[arm]["allocs_per_call"] = (after.alloc - before.alloc) / 100
                rows[arm]["frees_per_call"] = (after.free - before.free) / 100
        else:
            buffers = {
                arm: (np.zeros_like(x), np.zeros(1)) for arm, (_, _, x) in built.items()
            }
            chunk = 3000 if name in ("synthetic_large", "shared_reduce") else 30_000

            def run(arm):
                _, wrapper, x = built[arm]
                gradient, logp = buffers[arm]
                return runner.run(
                    wrapper.address,
                    x.size,
                    x.ctypes.data,
                    gradient.ctypes.data,
                    logp.ctypes.data,
                    chunk,
                )

            for _ in range(2):
                for arm in arms:
                    run(arm)
            samples = {arm: [] for arm in arms}
            for i in range(args.rounds):
                for arm in arms if i % 2 == 0 else arms[::-1]:
                    samples[arm].append(run(arm))
            for arm in arms:
                rows[arm]["ns_median"] = statistics.median(samples[arm])
                rows[arm]["ns_samples"] = samples[arm]
                if "full" in arms:
                    rows[arm]["paired_ratio_to_full"] = statistics.median(
                        [
                            a / b
                            for a, b in zip(samples[arm], samples["full"], strict=True)
                        ]
                    )
        result["models"][name] = rows
        import sys

        if "pymc" in sys.modules:
            result["environment"]["pymc"] = sys.modules["pymc"].__version__
        result["environment"]["load_end"] = os.getloadavg()
        (args.output / "results.json").write_text(json.dumps(result, indent=2))
        print(
            name,
            {
                arm: {k: v for k, v in row.items() if k not in {"llvm", "ns_samples"}}
                for arm, row in rows.items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
