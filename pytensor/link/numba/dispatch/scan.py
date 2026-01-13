from hashlib import sha256
from textwrap import dedent, indent

import numpy as np
from numba import types
from numba.extending import overload

from pytensor.compile.function.types import add_supervisor_to_fgraph, insert_deepcopy
from pytensor.compile.io import In, Out
from pytensor.compile.mode import NUMBA, get_mode
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.string_codegen import create_tuple_string
from pytensor.scan.op import Scan
from pytensor.tensor.type import TensorType


def idx_to_str(
    array_name: str,
    offset: int,
    size: str | None = None,
    idx_symbol: str = "i",
    allow_scalar=False,
) -> str:
    assert offset >= 0
    if offset > 0:
        indices = f"{idx_symbol} + {offset}"
    else:
        indices = idx_symbol

    if size:
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
            return range(x.item())

        return range_arr


@register_funcify_and_cache_key(Scan)
def numba_funcify_Scan(op: Scan, node, **kwargs):
    # Apply inner rewrites
    # TODO: Not sure this is the right place to do this, should we have a rewrite that
    #  explicitly triggers the optimization of the inner graphs of Scan?
    #  The C-code defers it to the make_thunk phase
    rewriter = (
        get_mode(op.mode)
        .including("numba")
        .excluding(*NUMBA._optimizer.exclude)
        .optimizer
    )
    fgraph = op.fgraph
    # When the buffer can only hold one SITSOT or as as many MITSOT as there are taps,
    # We must always discard the oldest tap, so it's safe to destroy it in the inner function.
    # TODO: Allow inplace for MITMOT
    destroyable_sitsot = [
        inner_sitsot
        for outer_sitsot, inner_sitsot in zip(
            op.outer_sitsot(node.inputs), op.inner_sitsot(fgraph.inputs), strict=True
        )
        if outer_sitsot.type.shape[0] == 1
    ]
    destroyable_mitsot = [
        oldest_inner_mitmot
        for outer_mitsot, oldest_inner_mitmot, taps in zip(
            op.outer_mitsot(node.inputs),
            op.oldest_inner_mitsot(fgraph.inputs),
            op.info.mit_sot_in_slices,
            strict=True,
        )
        if outer_mitsot.type.shape[0] == abs(min(taps))
    ]
    destroyable = {*destroyable_sitsot, *destroyable_mitsot}
    input_specs = [In(x, borrow=True, mutable=x in destroyable) for x in fgraph.inputs]
    add_supervisor_to_fgraph(
        fgraph=fgraph,
        input_specs=input_specs,
        accept_inplace=True,
    )
    rewriter(fgraph)
    output_specs = [Out(x, borrow=False) for x in fgraph.outputs]
    insert_deepcopy(fgraph, wrapped_inputs=input_specs, wrapped_outputs=output_specs)

    scan_inner_func, inner_func_cache_key = numba_funcify_and_cache_key(
        op.fgraph, fgraph_name="numba_scan"
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
    temp_0d_storage_alloc_stmts: list[str] = []
    inner_in_exprs_scalar: list[str] = []
    inner_in_exprs: list[str] = []

    def add_inner_in_expr(
        outer_in_name: str,
        tap_offset: int | None,
        storage_size_var: str | None,
        vector_slice_opt: bool,
    ):
        """Construct an inner-input expression."""
        storage_name = outer_in_to_storage_name.get(outer_in_name, outer_in_name)
        if vector_slice_opt:
            indexed_inner_in_str_scalar = idx_to_str(
                storage_name, tap_offset, size=storage_size_var, allow_scalar=True
            )
            temp_storage = f"{storage_name}_temp_scalar_{tap_offset}"
            storage_dtype = outer_in_var.type.numpy_dtype.name
            temp_0d_storage_alloc_stmts.append(
                f"{temp_storage} = np.empty((), dtype=np.{storage_dtype})"
            )
            inner_in_exprs_scalar.append(
                f"{temp_storage}[()] = {indexed_inner_in_str_scalar}"
            )
            indexed_inner_in_str = temp_storage
        else:
            indexed_inner_in_str = (
                storage_name
                if tap_offset is None
                else idx_to_str(
                    storage_name, tap_offset, size=storage_size_var, allow_scalar=True
                )
            )
        inner_in_exprs.append(indexed_inner_in_str)

    for outer_in_name in outer_in_seqs_names:
        # These outer-inputs are indexed without offsets or storage wrap-around
        outer_in_var = outer_in_names_to_vars[outer_in_name]
        is_vector = outer_in_var.ndim == 1
        add_inner_in_expr(outer_in_name, 0, None, vector_slice_opt=is_vector)

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

    # In truncated storage situations (e.g. created by `scan_save_mem`),
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

            is_tensor_type = isinstance(outer_in_var.type, TensorType)
            if is_tensor_type:
                storage_size_name = f"{outer_in_name}_len"
                storage_size_stmt = f"{storage_size_name} = {outer_in_name}.shape[0]"
                input_taps = inner_in_names_to_input_taps[outer_in_name]
                max_lookback_inp_tap = -min(0, min(input_taps))
                assert max_lookback_inp_tap >= 0

                for in_tap in input_taps:
                    tap_offset = max_lookback_inp_tap + in_tap
                    is_vector = outer_in_var.ndim == 1
                    add_inner_in_expr(
                        outer_in_name,
                        tap_offset,
                        storage_size_name,
                        vector_slice_opt=is_vector,
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
                add_inner_in_expr(outer_in_name, None, None, vector_slice_opt=False)
                inner_out_to_outer_in_stmts.append(storage_name)

            output_idx = outer_output_names.index(storage_name)
            if output_idx in node.op.destroy_map or not is_tensor_type:
                storage_alloc_stmt = f"{storage_name} = {outer_in_name}"
            else:
                storage_alloc_stmt = f"{storage_name} = np.copy({outer_in_name})"

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
                idx_to_str(storage_name, 0, size=storage_size_name, allow_scalar=True)
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
                {storage_size_name} = {outer_in_name}.item()
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
        add_inner_in_expr(name, None, None, vector_slice_opt=False)

    if op.info.as_while:
        # The inner function will return a boolean as the last value
        inner_out_to_outer_in_stmts.append("cond")

    assert len(inner_in_exprs) == len(op.fgraph.inputs)

    inner_scalar_in_args_to_temp_storage = "\n".join(inner_in_exprs_scalar)
    # Break inputs in new lines, just for readability of the source code
    inner_in_args = f",\n{' ' * 12}".join(inner_in_exprs)
    inner_outputs = create_tuple_string(inner_output_names)
    input_storage_block = "\n".join(storage_alloc_stmts)
    input_temp_0d_storage_block = "\n".join(temp_0d_storage_alloc_stmts)
    output_storage_post_processing_block = "\n".join(output_storage_post_proc_stmts)
    inner_out_post_processing_block = "\n".join(inner_out_post_processing_stmts)

    inner_out_to_outer_out_stmts = "\n".join(
        f"{s} = {d}"
        for s, d in zip(inner_out_to_outer_in_stmts, inner_output_names, strict=True)
    )

    scan_op_src = f"""
def scan({", ".join(outer_in_names)}):

{indent(input_storage_block, " " * 4)}

{indent(input_temp_0d_storage_block, " " * 4)}

    i = 0
    cond = np.array(False)
    while i < n_steps and not cond.item():
{indent(inner_scalar_in_args_to_temp_storage, " " * 8)}

        {inner_outputs} = scan_inner_func(
            {inner_in_args}
        )
{indent(inner_out_post_processing_block, " " * 8)}
{indent(inner_out_to_outer_out_stmts, " " * 8)}
        i += 1

{indent(output_storage_post_processing_block, " " * 4)}

    return {", ".join(outer_output_names)}
    """

    scan_op_fn = compile_numba_function_src(
        scan_op_src,
        "scan",
        globals() | {"np": np, "scan_inner_func": scan_inner_func},
    )

    if inner_func_cache_key is None:
        # If we can't cache the inner function, we can't cache the Scan either
        scan_cache_key = None
    else:
        scan_cache_key = sha256(
            f"({scan_op_src}, {inner_func_cache_key})".encode()
        ).hexdigest()

    return numba_basic.numba_njit(scan_op_fn, boundscheck=False), scan_cache_key
