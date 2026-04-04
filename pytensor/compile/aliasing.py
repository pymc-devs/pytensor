"""View, alias, and destroy handling for compiled functions."""

from collections.abc import Sequence
from itertools import chain

from pytensor.compile.io import SymbolicInput
from pytensor.compile.ops import deep_copy_op, view_op
from pytensor.configdefaults import config
from pytensor.graph.destroyhandler import DestroyHandler
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.traversal import graph_inputs
from pytensor.graph.utils import InconsistencyError


def alias_root(v):
    """
    Return the variable to which v is aliased by view_maps and destroy_maps.

    """
    if v.owner is None:
        return v
    vmap = v.owner.op.view_map
    dmap = v.owner.op.destroy_map
    outpos = v.owner.outputs.index(v)
    v_views = vmap.get(outpos, []) + dmap.get(outpos, [])
    if len(v_views) > 1:
        raise NotImplementedError(
            f"{v} is a view/destroyed version of more then one inputs. "
            "Currently, we only support the case where an output is a view or "
            "a destroyed version of one input."
        )
    elif v_views:
        return alias_root(v.owner.inputs[v_views[0]])
    else:
        return v


def view_tree_set(fgraph, v, treeset):
    """
    Add to `treeset` all variables that are views of v, given that v is
    not a view.

    """
    treeset.add(v)
    for cl, v_input_pos_to_cl in fgraph.clients[v]:
        vmap = cl.op.view_map
        dmap = cl.op.destroy_map
        for opos, iposlist in chain(vmap.items(), dmap.items()):
            if v_input_pos_to_cl in iposlist:
                if cl.outputs[opos] not in treeset:
                    view_tree_set(fgraph, cl.outputs[opos], treeset)


def infer_reuse_pattern(fgraph, outputs_to_disown):
    """
    Given an fgraph and a list of variables, returns the list or set
    of all variables which may share the same underlying data storage
    as any of the specified variables. Used internally by function,
    FunctionMaker.

    This list (or set) is also referred to as no_recycling sometimes,
    especially by linker code.

    """
    rval = set()
    for o in outputs_to_disown:
        view_tree_set(fgraph, alias_root(o), rval)
    # remove from rval all of the inputs, constants, values.
    rval = {r for r in rval if r.owner is not None}

    return rval


class Supervisor(Feature):
    """
    Listener for FunctionGraph events which makes sure that no
    operation overwrites the contents of protected Variables. The
    outputs of the FunctionGraph are protected by default.

    """

    def __init__(self, protected):
        self.fgraph = None
        self.protected = list(protected)

    def clone(self):
        return type(self)(self.protected)

    def on_attach(self, fgraph):
        if hasattr(fgraph, "_supervisor"):
            raise AlreadyThere(f"A Supervisor is already attached to {fgraph}.")

        if self.fgraph is not None and self.fgraph != fgraph:
            raise Exception("This Feature is already associated with a FunctionGraph")

        fgraph._supervisor = self
        self.fgraph = fgraph

    def validate(self, fgraph):
        if config.cycle_detection == "fast" and hasattr(fgraph, "has_destroyers"):
            if fgraph.has_destroyers(self.protected):
                raise InconsistencyError("Trying to destroy protected variables.")
            return True
        if not hasattr(fgraph, "destroyers"):
            return True
        for r in self.protected + list(fgraph.outputs):
            if fgraph.destroyers(r):
                raise InconsistencyError(f"Trying to destroy a protected variable: {r}")


def add_supervisor_to_fgraph(
    fgraph: FunctionGraph,
    input_specs: Sequence[SymbolicInput],
    accept_inplace: bool = False,
) -> None:
    """Setup Supervisor Feature in a FunctionGraph, so that inplace rewrites can be used.

    Parameters
    ----------
    fgraph: FunctionGraph
        The FunctionGraph to setup the Supervisor Feature in.
    input_specs: Sequence of SymbolicInput
        The input specifications for the FunctionGraph.
        Inputs with the attribute `mutable=False` and which are not already destroyed by an inplace operation
        (if `accept_inplace` is True) will be protected from inplace operations.
        Otherwise, they will be allowed to be destroyed.
    accept_inplace: bool
        Whether to allow inplace operations to already be present in the graph.

    Raises
    ------
    TypeError
        If inplace operations are not allowed and the graph already contains inplace operations.

    """

    has_destroy_handler = hasattr(fgraph, "destroyers")
    if not (has_destroy_handler and accept_inplace):
        # Check if fgraph already contains destructive operations,
        # in which case we need to add a DestroyHandler or raise an error
        for node in fgraph.apply_nodes:
            if node.op.destroy_map:
                if not accept_inplace:
                    raise TypeError(
                        f"Graph must not contain inplace operations: {node}"
                    )
                else:
                    has_destroy_handler = True
                    fgraph.attach_feature(DestroyHandler())
                    break

    # Protect all immutable inputs from inplace operations.
    fgraph.attach_feature(
        Supervisor(
            input
            for spec, input in zip(input_specs, fgraph.inputs, strict=True)
            if not (
                spec.mutable or (has_destroy_handler and fgraph.has_destroyers([input]))
            )
        )
    )


def insert_deepcopy(fgraph, wrapped_inputs, wrapped_outputs):
    """Insert deepcopy in the fgraph to break aliasing of outputs.

    This loop was inserted to remove aliasing between outputs when they all
    evaluate to the same value. Originally it was OK for outputs to be aliased,
    but some of the outputs can be shared variables, and is not good for shared
    variables to be aliased. It might be possible to rewrite this by making
    sure there is no aliasing only between shared variables.

    If some outputs are constant, we add deep copy to respect the memory
    contract

    We don't insert deep copy when :attr:`SymbolicOutput.borrow` is ``True``
    for all concerned outputs.
    """

    assert len(wrapped_inputs) == len(fgraph.inputs)
    assert len(wrapped_outputs) == len(fgraph.outputs)
    reason = "insert_deepcopy"
    updated_fgraph_inputs = {
        fgraph_i
        for i, fgraph_i in zip(wrapped_inputs, fgraph.inputs, strict=True)
        if getattr(i, "update", None) is not None
    }

    # We can't use fgraph.inputs as this don't include Constant Value.
    all_graph_inputs = list(graph_inputs(fgraph.outputs))
    has_destroyers_attr = hasattr(fgraph, "has_destroyers")

    for i in range(len(fgraph.outputs)):
        original_out = fgraph.outputs[i]
        output_client = fgraph.get_output_client(i)

        views_of_output_i = set()
        view_tree_set(fgraph, alias_root(original_out), views_of_output_i)
        copied = False
        # do not allow outputs to be aliased
        for j in range(i + 1, len(fgraph.outputs)):
            # We could don't put deep copy if both outputs have borrow==True
            # and not(wrapped_outputs[i].borrow and wrapped_outputs[j].borrow):
            if fgraph.outputs[j] in views_of_output_i:
                if wrapped_outputs[i].borrow and wrapped_outputs[j].borrow:
                    fgraph.change_node_input(
                        *output_client, view_op(original_out), reason=reason
                    )
                else:
                    fgraph.change_node_input(
                        *output_client, deep_copy_op(original_out), reason=reason
                    )
                copied = True
                break

        if not copied:  # no-break
            for input_j in all_graph_inputs:
                # do not allow outputs to be aliased to an inputs (j), unless
                # a) that j'th input has been 'destroyed' by
                #    e.g. in-place computations
                # b) that j'th input is a shared variable that is also
                #    being updated
                if input_j in updated_fgraph_inputs:
                    continue
                if input_j in views_of_output_i and not (
                    has_destroyers_attr and fgraph.has_destroyers([input_j])
                ):
                    # We don't put deep_copy_op if the input and the
                    # output have borrow==True
                    if input_j in fgraph.inputs:
                        j = fgraph.inputs.index(input_j)
                        if wrapped_outputs[i].borrow and wrapped_inputs[j].borrow:
                            fgraph.change_node_input(
                                *output_client,
                                view_op(original_out),
                                reason=reason,
                            )
                            break
                        else:
                            fgraph.change_node_input(
                                *output_client,
                                deep_copy_op(original_out),
                                reason=reason,
                            )
                            break
                    elif wrapped_outputs[i].borrow:
                        fgraph.change_node_input(
                            *output_client,
                            view_op(original_out),
                            reason=reason,
                        )
                        break
                    else:
                        fgraph.change_node_input(
                            *output_client,
                            deep_copy_op(original_out),
                            reason=reason,
                        )
                        break
