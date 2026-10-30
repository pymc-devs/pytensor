from hashlib import sha256
from textwrap import dedent, indent

import numpy as np
from numba import types
from numba.extending import overload

from pytensor.compile.aliasing import alias_root
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.basic import dfs_rewriter
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.link.numba.dispatch.string_codegen import create_tuple_string
from pytensor.scalar.basic import ScalarType
from pytensor.scan.op import Scan
from pytensor.tensor.basic import scalar_from_tensor, tensor_from_scalar
from pytensor.tensor.rewriting.basic import (
    local_scalar_tensor_scalar,
    local_tensor_scalar_tensor,
)
from pytensor.tensor.rewriting.scalarize import ScalarizeRewriter
from pytensor.tensor.type import TensorType


def scalarize_scan_inner_graph(fgraph):
    """Expose native scalar step inputs and outputs in a separate lowering graph."""
    if not any(
        isinstance(var.type, TensorType) and var.ndim == 0
        for var in (*fgraph.inputs, *fgraph.outputs)
    ):
        return fgraph
    fgraph = fgraph.unfreeze()
    inputs = [
        ScalarType(inp.dtype)(inp.name)
        if isinstance(inp.type, TensorType) and inp.ndim == 0
        else inp
        for inp in fgraph.inputs
    ]
    replacements = {
        original: tensor_from_scalar(scalar)
        for original, scalar in zip(fgraph.inputs, inputs, strict=True)
        if original is not scalar
    }
    outputs = graph_replace(fgraph.outputs, replacements, strict=False)
    for i, out in enumerate(outputs):
        if isinstance(out.type, TensorType) and out.ndim == 0:
            # Native scalar outputs cannot alias a buffer, so boundary copies
            # and views have no role in the step's scalar return convention.
            while isinstance(out.owner_op, DeepCopyOp | ViewOp):
                out = out.owner.inputs[0]
            outputs[i] = scalar_from_tensor(out)
    inner = FunctionGraph(inputs, outputs, features=[DestroyHandler()])
    ScalarizeRewriter().rewrite(inner)
    dfs_rewriter(
        local_scalar_tensor_scalar, local_tensor_scalar_tensor, ignore_newtrees=True
    ).rewrite(inner)
    return inner.freeze()


def idx_to_str(
    array_name: str,
    offset: int,
    size: str | None = None,
    idx_symbol: str = "i",
    allow_scalar=False,
    indexed_buffers: dict[str, set[int]] | None = None,
) -> str:
    assert offset >= 0
    if offset > 0:
        indices = f"{idx_symbol} + {offset}"
    else:
        indices = idx_symbol

    if size and indexed_buffers is not None:
        indexed_buffers.setdefault(size, set()).add(offset)
        indices = f"{size}_index_{offset}"
    elif size:
        # TODO FIXME: The `Scan` `Op` should tell us which outputs are computed
        # in this way.  We shouldn't have to waste run-time efforts in order to
        # compensate for this poor `Op`/rewrite design and implementation.
        indices = f"({indices}) % {size}"

    if allow_scalar:
        return f"{array_name}[{indices}]"
    else:
        return f"np.asarray({array_name}[{indices}])"


@overload(range)
def array0d_range(x):
    if isinstance(x, types.Array) and x.ndim == 0:

        def range_arr(x):
            return range(x[()])

        return range_arr


@register_funcify_and_cache_key(Scan)
def numba_funcify_Scan(op: Scan, node, **kwargs):
    """Generate a Numba implementation of a `Scan` loop."""
    # Outer untraced_sit_sot outputs whose inner input the baked step fn destroys;
    # when not owned, the codegen copies the outer input on the first iteration.
    destroyed_roots = {
        alias_root(inner_node.inputs[pos])
        for inner_node in op.fgraph.apply_nodes
        for positions in inner_node.op.destroy_map.values()
        for pos in positions
    }
    untraced_inputs_destroyed_by_inner_function = {
        outer_out_idx
        for inner_inp, (outer_out_idx, _) in zip(
            op.inner_untraced_sit_sot(op.fgraph.inputs),
            op.outer_untraced_sit_sot_outs(node.outputs, with_idx=True),
            strict=True,
        )
        if inner_inp in destroyed_roots
    }

    inner_fgraph = scalarize_scan_inner_graph(op.fgraph)
    scalar_inner_io = all(
        isinstance(var.type, ScalarType)
        for var in (*inner_fgraph.inputs, *inner_fgraph.outputs)
    )
    scan_inner_func, inner_func_cache_key = numba_funcify_and_cache_key(
        inner_fgraph, fgraph_name="numba_scan", ofg_memo=kwargs.get("ofg_memo")
    )

    outer_in_names_to_vars = {
        (f"outer_in_{i}" if i > 0 else "n_steps"): v for i, v in enumerate(node.inputs)
    }
    outer_in_names = list(outer_in_names_to_vars)
    outer_in_seqs_names = op.outer_seqs(outer_in_names)
    outer_in_mit_mot_names = op.outer_mitmot(outer_in_names)
    outer_in_mit_sot_names = op.outer_mitsot(outer_in_names)
    outer_in_sit_sot_names = op.outer_sitsot(outer_in_names)
    outer_in_nit_sot_names = op.outer_nitsot(outer_in_names)
    outer_in_untraced_sit_sot_names = op.outer_untraced_sit_sot(outer_in_names)
    outer_in_non_seqs_names = op.outer_non_seqs(outer_in_names)

    # These are all the outer-input names that have produce outputs/have output
    # taps (i.e. they have inner-outputs and corresponding outer-outputs).
    # Outer-outputs are ordered as follows:
    # mit-mot-outputs + mit-sot-outputs + sit-sot-outputs + nit-sots + untraced-sit-sot-outputs
    outer_in_outtap_names = (
        outer_in_mit_mot_names
        + outer_in_mit_sot_names
        + outer_in_sit_sot_names
        + outer_in_nit_sot_names
        + outer_in_untraced_sit_sot_names
    )

    # We create distinct variables for/references to the storage arrays for
    # each output.
    outer_in_to_storage_name: dict[str, str] = {}
    for outer_in_name in outer_in_mit_mot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_mitmot_storage"

    for outer_in_name in outer_in_mit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_mitsot_storage"

    for outer_in_name in outer_in_sit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_sitsot_storage"

    for outer_in_name in outer_in_nit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_nitsot_storage"

    for outer_in_name in outer_in_untraced_sit_sot_names:
        outer_in_to_storage_name[outer_in_name] = (
            f"{outer_in_name}_untraced_sit_sot_storage"
        )

    outer_output_names = list(outer_in_to_storage_name.values())
    assert len(outer_output_names) == len(node.outputs)

    # Construct the inner-input expressions (e.g. indexed storage expressions)
    # Inner-inputs are ordered as follows:
    # sequences + mit-mot-inputs + mit-sot-inputs + sit-sot-inputs +
    # untraced-sit-sot-inputs + non-sequences.
    inner_in_exprs: list[str] = []
    scalar_input_stmts: list[str] = []
    indexed_buffers: dict[str, set[int]] = {}

    def add_inner_in_expr(
        outer_in_name: str,
        tap_offset: int | None,
        storage_size_var: str | None,
    ):
        """Construct an inner-input expression."""
        outer_in_var = outer_in_names_to_vars[outer_in_name]
        storage_name = outer_in_to_storage_name.get(outer_in_name, outer_in_name)
        inner_input = inner_fgraph.inputs[len(inner_in_exprs)]
        scalar_input = isinstance(inner_input.type, ScalarType)
        indexed_inner_in_str = (
            storage_name
            if tap_offset is None
            else idx_to_str(
                storage_name,
                tap_offset,
                size=storage_size_var,
                allow_scalar=True,
                indexed_buffers=indexed_buffers,
            )
        )
        if (
            scalar_input
            and tap_offset is None
            and outer_in_name not in outer_in_untraced_sit_sot_names
            and isinstance(outer_in_var.type, TensorType)
        ):
            indexed_inner_in_str = f"{outer_in_name}_scalar"
            scalar_input_stmts.append(f"{indexed_inner_in_str} = {outer_in_name}[()]")
        inner_in_exprs.append(indexed_inner_in_str)

    for outer_in_name in outer_in_seqs_names:
        # These outer-inputs are indexed without offsets or storage wrap-around
        add_inner_in_expr(outer_in_name, 0, None)

    inner_in_names_to_input_taps: dict[str, tuple[int, ...]] = dict(
        zip(
            outer_in_mit_mot_names + outer_in_mit_sot_names + outer_in_sit_sot_names,
            op.info.mit_mot_in_slices
            + op.info.mit_sot_in_slices
            + op.info.sit_sot_in_slices,
            strict=True,
        )
    )
    inner_in_names_to_output_taps: dict[str, tuple[int, ...] | None] = dict(
        zip(outer_in_mit_mot_names, op.info.mit_mot_out_slices, strict=True)
    )

    # Inner-outputs consist of:
    # mit-mot-outputs + mit-sot-outputs + sit-sot-outputs + nit-sots +
    # untraced-sit-sot-outputs [+ while-condition]
    inner_output_names = [f"inner_out_{i}" for i in range(len(op.inner_outputs))]

    # The assignment statements that copy inner-outputs into the outer-outputs
    # storage
    inner_out_to_outer_in_stmts: list[str] = []

    # Special statements that perform storage truncation for `while`-loops and
    # rotation for initially truncated storage.
    output_storage_post_proc_stmts: list[str] = []

    # In truncated storage situations (e.g. created by `scan_reduce_trace`),
    # the taps and output storage overlap, instead of the standard situation in
    # which the output storage is large enough to contain both the initial taps
    # values and the output storage.  In this truncated case, we use the
    # storage array like a circular buffer, and that's why we need to track the
    # storage size along with the taps length/indexing offset.
    def add_output_storage_post_proc_stmt(
        outer_in_name: str, max_offset: int, storage_size: str
    ):
        # Rotate the storage so that the last computed value is at the end of the storage array.
        # This is needed when the output storage array does not have a length
        # equal to the number of taps plus `n_steps`.
        output_storage_post_proc_stmts.append(
            dedent(
                f"""
                if 1 < {storage_size} < (i + {max_offset}):
                    {outer_in_name}_shift = (i + {max_offset}) % ({storage_size})
                    if {outer_in_name}_shift > 0:
                        {outer_in_name}_left = {outer_in_name}[:{outer_in_name}_shift]
                        {outer_in_name}_right = {outer_in_name}[{outer_in_name}_shift:]
                        {outer_in_name} = np.concatenate(({outer_in_name}_right, {outer_in_name}_left))
                """
            ).strip()
        )

        if op.info.as_while:
            # While loops need to truncate the output storage to a length given
            # by the number of iterations performed.
            output_storage_post_proc_stmts.append(
                dedent(
                    f"""
                    elif {storage_size} > (i + {max_offset}):
                        {outer_in_name} = {outer_in_name}[:i + {max_offset}]
                    """
                ).strip()
            )
        else:
            # And regular loops should zero out unused entries of the output buffer
            # These show up with truncated gradients of while loops
            output_storage_post_proc_stmts.append(
                dedent(
                    f"""
                    elif {storage_size} > (i + {max_offset}):
                        {outer_in_name}[i + {max_offset}:] = 0
                    """
                ).strip()
            )

    # Special in-loop statements that create (nit-sot) storage arrays after a
    # single iteration is performed.  This is necessary because we don't know
    # the exact shapes of the storage arrays that need to be allocated until
    # after an iteration is performed.
    inner_out_post_processing_stmts: list[str] = []

    # Storage allocation statements
    # For output storage allocated/provided by the inputs, these statements
    # will either construct aliases between the input names and the entries in
    # `outer_in_to_storage_name` or assign the latter to expressions that
    # create copies of those storage inputs.
    # In the nit-sot case, empty dummy arrays are assigned to the storage
    # variables and updated later by the statements in
    # `inner_out_post_processing_stmts`.
    storage_alloc_stmts: list[str] = []

    for outer_in_name in outer_in_outtap_names:
        outer_in_var = outer_in_names_to_vars[outer_in_name]

        if outer_in_name not in outer_in_nit_sot_names:
            storage_name = outer_in_to_storage_name[outer_in_name]

            is_tapped = outer_in_name not in outer_in_untraced_sit_sot_names
            if is_tapped:
                storage_size_name = f"{outer_in_name}_len"
                storage_size_stmt = f"{storage_size_name} = {outer_in_name}.shape[0]"
                input_taps = inner_in_names_to_input_taps[outer_in_name]
                max_lookback_inp_tap = -min(0, min(input_taps))
                assert max_lookback_inp_tap >= 0

                for in_tap in input_taps:
                    tap_offset = max_lookback_inp_tap + in_tap
                    add_inner_in_expr(
                        outer_in_name,
                        tap_offset,
                        storage_size_name,
                    )

                output_taps = inner_in_names_to_output_taps.get(outer_in_name, [0])
                for out_tap in output_taps:
                    tap_offset = max_lookback_inp_tap + out_tap
                    assert tap_offset >= 0
                    inner_out_to_outer_in_stmts.append(
                        idx_to_str(
                            storage_name,
                            tap_offset,
                            size=storage_size_name,
                            allow_scalar=True,
                            indexed_buffers=indexed_buffers,
                        )
                    )

                if outer_in_name not in outer_in_mit_mot_names:
                    # MIT-SOT and SIT-SOT may require buffer rolling/truncation after the main loop
                    max_offset_out_tap = max(output_taps) + max_lookback_inp_tap
                    add_output_storage_post_proc_stmt(
                        storage_name, max_offset_out_tap, storage_size_name
                    )

            else:
                storage_size_stmt = ""
                add_inner_in_expr(outer_in_name, None, None)
                inner_out_to_outer_in_stmts.append(storage_name)

            output_idx = outer_output_names.index(storage_name)
            # Copy the outer inputs when the loop mutates them and the destroy_map doesn't already grant permission
            needs_copy = output_idx not in node.op.destroy_map and (
                # Traced buffers are always mutated by the loop write-back procedure
                is_tapped
                # Untraced inputs are only mutated by the inner function,
                # so we make the copy conditional on that actually happening
                or output_idx in untraced_inputs_destroyed_by_inner_function
            )
            if (
                not is_tapped
                and isinstance(outer_in_var.type, TensorType)
                and outer_in_var.ndim == 0
            ):
                storage_alloc_stmt = f"{storage_name} = {outer_in_name}[()]"
                outer_output_names[output_idx] = (
                    f"np.asarray({storage_name}, dtype=np.{outer_in_var.dtype})"
                )
            elif needs_copy:
                storage_alloc_stmt = f"{storage_name} = numba_deepcopy({outer_in_name})"
            else:
                storage_alloc_stmt = f"{storage_name} = {outer_in_name}"

            storage_alloc_stmt = dedent(
                f"""
                {storage_size_stmt}
                {storage_alloc_stmt}
                """
            ).strip()

            storage_alloc_stmts.append(storage_alloc_stmt)

        else:
            assert outer_in_name in outer_in_nit_sot_names

            # This is a special case in which there are no outer-inputs used
            # for outer-output storage, so we need to create our own storage
            # from scratch.
            storage_name = outer_in_to_storage_name[outer_in_name]
            storage_size_name = f"{outer_in_name}_len"

            inner_out_to_outer_in_stmts.append(
                idx_to_str(
                    storage_name,
                    0,
                    size=storage_size_name,
                    allow_scalar=True,
                    indexed_buffers=indexed_buffers,
                )
            )
            add_output_storage_post_proc_stmt(storage_name, 0, storage_size_name)

            # In case of nit-sots we are provided the length of the array in
            # the iteration dimension instead of actual arrays, hence we
            # allocate space for the results accordingly.
            curr_nit_sot_position = outer_in_nit_sot_names.index(outer_in_name)
            curr_nit_sot = op.inner_nitsot_outs(op.inner_outputs)[curr_nit_sot_position]

            known_static_shape = all(dim is not None for dim in curr_nit_sot.type.shape)
            if known_static_shape:
                storage_shape = create_tuple_string(
                    (storage_size_name, *(map(str, curr_nit_sot.type.shape)))
                )
            else:
                storage_shape = create_tuple_string(
                    (storage_size_name, *(["0"] * curr_nit_sot.ndim))
                )
            storage_dtype = curr_nit_sot.type.numpy_dtype.name

            storage_alloc_stmts.append(
                dedent(
                    f"""
                {storage_size_name} = {outer_in_name}[()]
                {storage_name} = np.empty({storage_shape}, dtype=np.{storage_dtype})
                """
                ).strip()
            )

            if not known_static_shape:
                # In this case, we don't know the shape of the output storage
                # array until we get some output from the inner-function.
                # With the following we add delayed output storage initialization:
                inner_out_name = op.inner_nitsot_outs(inner_output_names)[
                    curr_nit_sot_position
                ]
                inner_out_post_processing_stmts.append(
                    dedent(
                        f"""
                    if i == 0:
                        {storage_name} = np.empty(({storage_size_name},) + np.shape({inner_out_name}), dtype=np.{storage_dtype})
                    """
                    ).strip()
                )

    for name in outer_in_non_seqs_names:
        add_inner_in_expr(name, None, None)

    if op.info.as_while:
        # The inner function will return a boolean as the last value
        inner_out_to_outer_in_stmts.append("cond")

    assert len(inner_in_exprs) == len(op.fgraph.inputs)

    inner_outputs = create_tuple_string(inner_output_names)
    input_storage_block = "\n".join(storage_alloc_stmts)
    output_storage_post_processing_block = "\n".join(output_storage_post_proc_stmts)
    inner_out_post_processing_block = "\n".join(inner_out_post_processing_stmts)

    inner_out_to_outer_out_stmts = "\n".join(
        f"{s} = {d}"
        for s, d in zip(inner_out_to_outer_in_stmts, inner_output_names, strict=True)
    )

    def loop_source(index_stmts, in_exprs=inner_in_exprs):
        inner_in_args = f",\n{' ' * 12}".join(in_exprs)
        return f"""\
    i = 0
    cond = False
    while i < n_steps and not cond:
{indent(index_stmts, " " * 8)}
        {inner_outputs} = scan_inner_func(
            {inner_in_args}
        )
{indent(inner_out_post_processing_block, " " * 8)}
{indent(inner_out_to_outer_out_stmts, " " * 8)}
        i += 1
"""

    if indexed_buffers:
        conditions = []
        indices = []
        for size, offsets in indexed_buffers.items():
            conditions.append(f"n_steps <= {size} - {max(offsets)}")
            indices.extend(
                f"{size}_index_{offset} = i + {offset} if no_wrap else (i + {offset}) % {size}"
                for offset in sorted(offsets)
            )
        # LLVM unswitches this invariant choice. Known single-slot recurrences
        # have already become untraced states; other buffers keep modulo indexing.
        loop_block = f"    no_wrap = {' and '.join(conditions)}\n" + loop_source(
            "\n".join(indices)
        )
    elif op.info.n_seqs and scalar_inner_io:
        # The guards prove nonnegative indices and fixed strides for LLVM.
        unsigned_inputs = [f"{seq}[np.uintp(i)]" for seq in outer_in_seqs_names]
        unsigned_loop = loop_source(
            "", unsigned_inputs + inner_in_exprs[op.info.n_seqs :]
        )
        branches = []
        for keyword, direction in (("if", -1), ("elif", 1)):
            conditions = [
                f"(0 <= n_steps <= {seq}.shape[0] and {seq}.strides[0] == "
                f"{direction * outer_in_names_to_vars[seq].type.numpy_dtype.itemsize})"
                for seq in outer_in_seqs_names
            ]
            branches.append(
                f"    {keyword} {' and '.join(conditions)}:\n"
                + indent(unsigned_loop, " " * 4)
            )
        loop_block = (
            "".join(branches) + "    else:\n" + indent(loop_source(""), " " * 4)
        )
    else:
        loop_block = loop_source("")

    scalar_input_block = "\n".join(scalar_input_stmts)
    scan_op_src = f"""
def scan({", ".join(outer_in_names)}):

{indent(input_storage_block, " " * 4)}
{indent(scalar_input_block, " " * 4)}
    n_steps = n_steps[()]

{loop_block}

{indent(output_storage_post_processing_block, " " * 4)}

    return {", ".join(outer_output_names)}
    """

    scan_op_fn = compile_numba_function_src(
        scan_op_src,
        "scan",
        globals()
        | {
            "np": np,
            "scan_inner_func": scan_inner_func,
            "numba_deepcopy": numba_deepcopy,
        },
    )

    if inner_func_cache_key is None:
        # If we can't cache the inner function, we can't cache the Scan either
        scan_cache_key = None
    else:
        scan_cache_version = 1
        scan_cache_key = sha256(
            f"({scan_op_src}, {inner_func_cache_key}, {scan_cache_version})".encode()
        ).hexdigest()

    return numba_basic.numba_njit(scan_op_fn, boundscheck=False), scan_cache_key
