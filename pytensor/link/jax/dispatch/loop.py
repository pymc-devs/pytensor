import jax
from jax.tree_util import tree_flatten, tree_unflatten

from pytensor.compile.mode import get_mode
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.loop.op import Scan
from pytensor.typed_list import TypedListType


@jax_funcify.register(Scan)
def jax_funcify_Scan(op, node, global_fgraph, **kwargs):
    # TODO: Rewrite as a while loop if only last states are used
    if op.has_while_condition:
        raise NotImplementedError(
            "Scan ops with while condition cannot be transpiled JAX"
        )

    # Apply inner rewrites
    # TODO: Not sure this is the right place to do this, should we have a rewrite that
    #  explicitly triggers the optimization of the inner graphs of Scan?
    update_fg = op.update_fg.clone()
    rewriter = get_mode("JAX").optimizer
    rewriter(update_fg)

    jaxified_scan_inner_fn = jax_funcify(update_fg, **kwargs)

    # Only include the intermediate states that are used elsewhere
    used_traces_idxs = [
        i
        for i, trace in enumerate(node.outputs[op.n_states :])
        if global_fgraph.clients[trace]
    ]

    def scan(max_iters, *outer_inputs):
        states = outer_inputs[: op.n_states]
        constants = outer_inputs[op.n_states :]

        def scan_fn(carry, _):
            resume, *carry = jaxified_scan_inner_fn(*carry, *constants)
            assert resume
            carry = list(carry)
            # Return states as both carry and output to be appended
            return carry, [c for i, c in enumerate(carry) if i in used_traces_idxs]

        states, traces = jax.lax.scan(
            scan_fn, init=list(states), xs=None, length=max_iters
        )
        final_traces = [None] * len(states)
        for idx, trace in zip(used_traces_idxs, traces):
            if isinstance(op.trace_types[idx], TypedListType):
                flattened_trace, treedef = tree_flatten(trace)
                transposed_trace = [
                    tree_unflatten(treedef, l) for l in zip(*flattened_trace)
                ]
                final_traces[idx] = transposed_trace
            else:
                final_traces[idx] = trace

        return *states, *final_traces

    return scan
