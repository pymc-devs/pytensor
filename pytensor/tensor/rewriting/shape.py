import traceback
from io import StringIO
from typing import cast as type_cast
from warnings import warn

import numpy as np

import pytensor
from pytensor.configdefaults import config
from pytensor.graph.basic import Constant, Variable, equal_computations
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    copy_stack_trace,
    node_rewriter,
)
from pytensor.graph.traversal import ancestors
from pytensor.graph.utils import InconsistencyError, get_variable_trace_string
from pytensor.scalar import ScalarType
from pytensor.tensor.basic import (
    MakeVector,
    as_tensor_variable,
    cast,
    constant,
    expand_dims,
    get_scalar_constant_value,
    register_infer_shape,
    stack,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError, ShapeError
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
    register_useless,
    topo_constant_folding,
)
from pytensor.tensor.rewriting.elemwise import apply_local_dimshuffle_lift
from pytensor.tensor.shape import (
    Reshape,
    Shape,
    Shape_i,
    SpecifyShape,
    specify_shape,
)
from pytensor.tensor.subtensor import Subtensor, get_idx_list
from pytensor.tensor.type import TensorType, discrete_dtypes, integer_dtypes
from pytensor.tensor.type_other import NoneTypeT
from pytensor.tensor.variable import TensorVariable


class ShapeFeature(Feature):
    r"""A `Feature` that tracks shape information in a graph.

    This `Feature` aids in the replacement of all `Shape`\s and `Subtensor`\s of `Shape`\s with
    `Shape_i` and `MakeVector` `Op`\s.

    This `Feature` and its associated rewrites have several goals:

    1. to "lift" `Shape`\s to as close to the inputs as possible,
    2. to infer the shape of every node in the graph in terms of the
       input shapes, and
    3. remove fill `Op`\s (e.g. `Second`) from the graph.

    Lifting shapes as close to the inputs as possible is important for
    canonicalization because it is very bad form to have to compute
    something just to know how big it will be.  Firstly, it is a waste
    of time to compute such outputs.  But it is important to get rid
    of these outputs as early as possible in the compilation process
    because the extra computations make it appear as if many internal
    graph nodes have multiple clients.  Many rewrites refuse to
    work on nodes with multiple clients.

    Lifting is done by using an `<Op>.infer_shape` function if one is
    present, or else using a conservative default.  An Op that
    supports shape-lifting should define a infer_shape(self, fgraph, node,
    input_shapes) function.  The argument input_shapes is a tuple of
    tuples... there is an interior tuple for each input to the node.
    The tuple has as many elements as dimensions.  The element in
    position i of tuple j represents the i'th shape component of the
    j'th input.  The function should return a tuple of tuples.  One
    output tuple for each node.output.  Again, the i'th element of the
    j'th output tuple represents the output[j].shape[i] of the
    function.  If an output is not a TensorType, then None should be
    returned instead of a tuple for that output.

    For example the infer_shape for a matrix-matrix product would accept
    input_shapes=((x0,x1), (y0,y1)) and return ((x0, y1),).

    Inferring the shape of internal nodes in the graph is important
    for doing size-driven rewrites.  If we know how big various
    intermediate results will be, we can estimate the cost of many Ops
    accurately, and generate c-code that is specific [e.g. unrolled]
    to particular sizes.

    In cases where you cannot figure out the shape, raise a ShapeError.

    Notes
    -----
    To use this shape information in rewrites, use the
    ``shape_of`` dictionary.

    For example:

    .. code-block:: python

        try:
            shape_of = fgraph.shape_feature.shape_of
        except AttributeError:
            # This can happen when the mode doesn't include the ShapeFeature.
            return

        shape_of_output_zero = shape_of[node.output[0]]

    The ``shape_of_output_zero`` symbol will contain a tuple, whose
    elements are either integers or symbolic integers.

    TODO: check to see if the symbols are necessarily
    non-constant... or are integer literals sometimes PyTensor
    constants?? That would be confusing.

    """

    def get_node_infer_shape(self, node):
        try:
            shape_infer = node.op.infer_shape
        except AttributeError:
            shape_infer = self.default_infer_shape

        try:
            o_shapes = shape_infer(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )
        except ShapeError:
            o_shapes = self.default_infer_shape(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                "Code called by infer_shape failed raising a "
                "NotImplementedError. Raising NotImplementedError to "
                "indicate that a shape cannot be computed is no longer "
                "supported, and one should now use ShapeError "
                f"instead. The original exception message is: {e}"
            ).with_traceback(e.__traceback__)
        except Exception as e:
            msg = (
                f"Failed to infer_shape from Op {node.op}.\nInput shapes: "
                f"{[self.shape_of[r] for r in node.inputs]}\nException encountered during infer_shape: "
                f"{type(e)}\nException message: {e!s}\nTraceback: {traceback.format_exc()}"
            )
            if config.on_shape_error == "raise":
                raise Exception(msg).with_traceback(e.__traceback__)
            else:
                warn(msg)
            o_shapes = self.default_infer_shape(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )

        return o_shapes

    def get_shape(self, var, idx):
        """Rewrites can call this to get a `Shape_i`.

        It is better to call this then use directly ``shape_of[var][idx]``
        as this method should update `shape_of` if needed.

        TODO: Up to now, we don't update it in all cases. Update in all cases.
        """
        r = self.shape_of[var][idx]
        if (
            r.owner
            and isinstance(r.owner.op, Shape_i)
            and r.owner.inputs[0] not in self.fgraph.variables
        ):
            assert var.owner
            node = var.owner
            # recur on inputs
            for i in node.inputs:
                if getattr(i.type, "ndim", None) > 0:
                    self.get_shape(i, 0)
            o_shapes = self.get_node_infer_shape(node)
            assert len(o_shapes) == len(node.outputs)

            # Only change the variables and dimensions that would introduce
            # extra computation
            for new_shps, out in zip(o_shapes, node.outputs, strict=True):
                if not hasattr(out.type, "ndim"):
                    continue

                merged_shps = list(self.shape_of[out])
                changed = False
                for i in range(out.type.ndim):
                    n_r = merged_shps[i]
                    if (
                        n_r.owner
                        and isinstance(n_r.owner.op, Shape_i)
                        and n_r.owner.inputs[0] not in self.fgraph.variables
                    ):
                        changed = True
                        merged_shps[i] = new_shps[i]
                if changed:
                    self.set_shape(out, merged_shps, override=True)
            r = self.shape_of[var][idx]
        return r

    def shape_ir(self, i, r):
        """Return symbolic r.shape[i] for tensor variable r, int i."""
        if hasattr(r.type, "shape") and r.type.shape[i] is not None:
            return constant(r.type.shape[i], dtype="int64")
        else:
            # Do not call make_node for test_value
            s = Shape_i(i)(r)
            try:
                s = get_scalar_constant_value(s)
            except NotScalarConstantError:
                pass
            return s

    def shape_tuple(self, r):
        """Return a tuple of symbolic shape vars for tensor variable r."""
        if not hasattr(r.type, "ndim"):
            # This happen for NoneConst.
            return None
        return tuple(self.shape_ir(i, r) for i in range(r.type.ndim))

    def default_infer_shape(self, fgraph, node, i_shapes):
        """Return a list of shape tuple or None for the outputs of node.

        This function is used for Ops that don't implement infer_shape.
        Ops that do implement infer_shape should use the i_shapes parameter,
        but this default implementation ignores it.

        """
        rval = []
        for r in node.outputs:
            try:
                rval.append(self.shape_tuple(r))
            except AttributeError:
                rval.append(None)
        return rval

    def unpack(self, s_i, var):
        """Return a symbolic integer scalar for the shape element s_i.

        The s_i argument was produced by the infer_shape() of an Op subclass.

        var: the variable that correspond to s_i. This is just for
        error reporting.

        """
        assert s_i is not None

        if s_i == 1:
            return self.lscalar_one
        if isinstance(s_i, float) and int(s_i) == s_i:
            s_i = int(s_i)
        if isinstance(s_i, np.integer | int) or (
            isinstance(s_i, np.ndarray) and s_i.ndim == 0
        ):
            # this shape is a constant
            if s_i < 0:
                msg = "There is a negative shape in the graph!"
                msg += get_variable_trace_string(var)
                # The rest of the pipeline don't handle correctly this
                # case.  So we have 2 choices, stop compilation or
                # consider the shape as unknown.  As we have more
                # chance to give the stack trace here then later, I
                # choose that options as it would give better error
                # message.
                raise AssertionError(msg)
            return constant(s_i, dtype="int64")
        if isinstance(s_i, tuple | list):
            # this dimension is the same as many of the inputs
            # which tells us that if one of the inputs is known,
            # the others all become known.
            # TODO: should be implemented in Elemwise, and Dot
            #
            # worst case, we loop over shape_of and replace things
            raise NotImplementedError(s_i)

        # s_i is x.shape[i] for some x, we change it to shape_of[x][i]
        if (
            s_i.owner
            and isinstance(s_i.owner.op, Subtensor)
            and s_i.owner.inputs[0].owner
            and isinstance(s_i.owner.inputs[0].owner.op, Shape)
        ):
            assert s_i.type.ndim == 0
            assert len(s_i.owner.op.idx_list) == 1

            # The current Subtensor always put constant index in the graph.
            # This was not True in the past. So call the Subtensor function
            # that will return the right index.
            idx = get_idx_list(s_i.owner.inputs, s_i.owner.op.idx_list)
            assert len(idx) == 1
            idx = idx[0]
            try:
                i = get_scalar_constant_value(idx)
            except NotScalarConstantError:
                pass
            else:
                # Executed only if no exception was raised
                x = s_i.owner.inputs[0].owner.inputs[0]
                # x should already have been imported, and should be in shape_of.
                s_i = self.shape_of[x][i]

        if s_i.type.dtype in integer_dtypes:
            if getattr(s_i.type, "ndim", 0):
                raise TypeError("Shape element must be scalar", s_i)
            return s_i
        else:
            raise TypeError(
                "Unsupported shape element", s_i, type(s_i), getattr(s_i, "type", None)
            )

    def set_shape(self, r, s, override=False):
        """Assign the shape `s` to previously un-shaped variable `r`.

        Parameters
        ----------
        r : a variable
        s : None or a tuple of symbolic integers
        override : If False, it mean r is a new object in the fgraph.
            If True, it mean r is already in the fgraph and we want to
            override its shape.

        """
        if not override:
            assert r not in self.shape_of, "r already in shape_of"
        if s is None:
            self.shape_of[r] = s
        else:
            if not isinstance(s, tuple | list):
                raise TypeError("shapes must be tuple/list", (r, s))

            if r.type.ndim != len(s):
                sio = StringIO()
                pytensor.printing.debugprint(r, file=sio, print_type=True)
                raise AssertionError(
                    f"Something inferred a shape with {len(s)} dimensions "
                    f"for a variable with {int(r.type.ndim)} dimensions"
                    f" for the variable:\n{sio.getvalue()}"
                )

            shape_vars = []
            for i in range(r.type.ndim):
                if hasattr(r.type, "shape") and r.type.shape[i] is not None:
                    shape_vars.append(constant(r.type.shape[i], dtype="int64"))
                else:
                    shape_vars.append(self.unpack(s[i], r))
            assert all(
                not hasattr(r.type, "shape")
                or r.type.shape[i] != 1
                or self.lscalar_one.equals(shape_vars[i])
                or self.lscalar_one.equals(
                    get_scalar_constant_value(shape_vars[i], raise_not_constant=False)
                )
                for i in range(r.type.ndim)
            )
            self.shape_of[r] = tuple(shape_vars)
            for sv in shape_vars:
                self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def update_shape(self, r, other_r):
        """Replace shape of r by shape of other_r.

        If, on some dimensions, the shape of other_r is not informative,
        keep the shape of r on those dimensions.

        """
        # other_r should already have a shape
        assert other_r in self.shape_of, ("other_r not in shape_of", other_r)
        other_shape = self.shape_of[other_r]

        # If other_shape has no information, call is pointless.
        if other_shape is None:
            return

        if r in self.shape_of:
            r_shape = self.shape_of[r]
        else:
            # If no info is known on r's shape, use other_shape
            self.set_shape(r, other_shape)
            return
        if (
            other_r.owner
            and r.owner
            and other_r.owner.inputs == r.owner.inputs
            and other_r.owner.op == r.owner.op
        ):
            # We are doing a merge, so the two shape graphs will be the
            # same.  This is only done so that we call `ancestors` less
            # frequently.
            return

        # Merge other_shape with r_shape, giving the priority to other_shape
        merged_shape = []
        for i, ps in enumerate(other_shape):
            if r_shape is None and other_shape:
                merged_shape.append(other_shape[i])
            elif (
                ps.owner
                and isinstance(ps.owner.op, Shape_i)
                and ps.owner.op.i == i
                and ps.owner.inputs[0] in (r, other_r)
            ):
                # If other_shape[i] is uninformative, use r_shape[i].
                # For now, we consider 2 cases of uninformative other_shape[i]:
                #  - Shape_i(i)(other_r);
                #  - Shape_i(i)(r).
                merged_shape.append(r_shape[i])
            elif isinstance(r_shape[i], Constant | int):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(r_shape[i])
            elif isinstance(other_shape[i], Constant | int):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(other_shape[i])
            elif other_shape[i] == r_shape[i]:
                # This mean the shape is equivalent
                # We do not want to do the ancestor check in those cases
                merged_shape.append(r_shape[i])
            elif any(
                (
                    r_shape[i] == anc
                    or (
                        anc.owner
                        and isinstance(anc.owner.op, Shape)
                        and anc.owner.inputs[0] == r
                    )
                )
                for anc in ancestors([other_shape[i]])
            ):
                # Another case where we want to use r_shape[i] is when
                # other_shape[i] actually depends on r_shape[i]. In that case,
                # we do not want to substitute an expression with another that
                # is strictly more complex. Such a substitution could also lead
                # to cycles: if (in the future) r_shape[i] gets replaced by an
                # expression of other_shape[i], other_shape[i] may end up
                # depending on itself.
                merged_shape.append(r_shape[i])
            else:
                merged_shape.append(other_shape[i])
        assert all(
            (
                not hasattr(r.type, "shape")
                or (r.type.shape[i] != 1 and other_r.type.shape[i] != 1)
            )
            or self.lscalar_one.equals(merged_shape[i])
            or self.lscalar_one.equals(
                get_scalar_constant_value(
                    merged_shape[i],
                    only_process_constants=True,
                    raise_not_constant=False,
                )
            )
            for i in range(r.type.ndim)
        )
        self.shape_of[r] = tuple(merged_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def set_shape_i(self, r, i, s_i):
        """Replace element i of shape_of[r] by s_i"""
        assert r in self.shape_of
        prev_shape = self.shape_of[r]
        # prev_shape is a tuple, so we cannot change it inplace,
        # so we build another one.
        new_shape = []
        for j, s_j in enumerate(prev_shape):
            if j == i:
                new_shape.append(self.unpack(s_i, r))
            else:
                new_shape.append(s_j)
        assert all(
            not hasattr(r.type, "shape")
            or r.type.shape[idx] != 1
            or self.lscalar_one.equals(new_shape[idx])
            or self.lscalar_one.equals(
                get_scalar_constant_value(new_shape[idx], raise_not_constant=False)
            )
            for idx in range(r.type.ndim)
        )
        self.shape_of[r] = tuple(new_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def init_r(self, r):
        """Register r's shape in the shape_of dictionary."""
        if r not in self.shape_of:
            self.set_shape(r, self.shape_tuple(r))

    def make_vector_shape(self, r):
        return as_tensor_variable(self.shape_of[r], ndim=1, dtype="int64")

    def on_attach(self, fgraph):
        if hasattr(fgraph, "shape_feature"):
            raise AlreadyThere("This FunctionGraph already has a ShapeFeature")

        if hasattr(self, "fgraph") and self.fgraph != fgraph:
            raise Exception("This ShapeFeature is already attached to a graph")

        self.fgraph = fgraph

        fgraph.shape_feature = self
        # Must be local to the object as otherwise we reuse the same
        # variable for multiple fgraph!
        self.lscalar_one = constant(1, dtype="int64")
        assert self.lscalar_one.type.dtype == "int64"

        self.fgraph = fgraph
        # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
        self.shape_of = {}
        # Variable ->
        self.scheduled = {}
        # shape var -> graph v
        self.shape_of_reverse_index = {}

        for node in fgraph.toposort():
            self.on_import(fgraph, node, reason="on_attach")

    def on_detach(self, fgraph):
        self.shape_of = {}
        self.scheduled = {}
        self.shape_of_reverse_index = {}
        self.fgraph = None
        del fgraph.shape_feature

    def on_import(self, fgraph, node, reason):
        if node.outputs[0] in self.shape_of:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.shape_of
            return

        for i, r in enumerate(node.inputs):
            # make sure we have shapes for the inputs
            self.init_r(r)

        o_shapes = self.get_node_infer_shape(node)

        # this is packed information
        # an element of o_shapes is either None or a tuple
        #   elements of the tuple can be either strings, or ints
        if len(o_shapes) != len(node.outputs):
            raise Exception(
                f'The infer_shape method for the Op "{node.op}" returned a list '
                f"with the wrong number of element: len(o_shapes) = {len(o_shapes)} "
                f" != len(node.outputs) = {len(node.outputs)}"
            )

        # Ensure shapes are in 'int64'. This is to make sure the assert
        # found in the `local_useless_subtensor` rewrite does not fail.
        for sh_idx, sh in enumerate(o_shapes):
            if sh is None:
                continue
            if not isinstance(sh, list | tuple):
                raise ValueError(
                    f"infer_shape of {node} didn't return a list of"
                    f" list. It returned '{o_shapes}'"
                )
            new_shape = []
            for i, d in enumerate(sh):
                # Note: we ignore any shape element that is not typed (i.e.,
                # does not have a 'dtype' attribute). This means there may
                # still remain int elements that are int32 on 32-bit platforms,
                # but this works with `local_useless_subtensor`, so for now we
                # keep it this way. See #266 for a better long-term fix.
                if getattr(d, "dtype", "int64") != "int64":
                    assert d.dtype in discrete_dtypes, (node, d.dtype)
                    assert str(d.dtype) != "uint64", node
                    new_shape += sh[len(new_shape) : i + 1]
                    if isinstance(d, Constant):
                        casted_d = constant(d.data, dtype="int64")
                    else:
                        casted_d = cast(d, "int64")
                    new_shape[i] = casted_d
            if new_shape:
                # We replace the shape with wrong dtype by the one with
                # 'int64'.
                new_shape += sh[len(new_shape) :]
                o_shapes[sh_idx] = tuple(new_shape)

        for r, s in zip(node.outputs, o_shapes, strict=True):
            self.set_shape(r, s)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if new_r not in self.shape_of:
            # It happen that the fgraph didn't called on_import for some
            # new_r.  This happen when new_r don't have an
            # owner(i.e. it is a constant or an input of the graph)
            # update_shape suppose that r and new_r are in shape_of.
            self.init_r(new_r)

        # This tells us that r and new_r must have the same shape if
        # we didn't know that the shapes are related, now we do.
        self.update_shape(new_r, r)

        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.

        # In case 1, if r has a shape_i client, we will want to
        # replace the shape_i of r with the shape of new_r.  Say that
        # r is *scheduled*.
        # At that point, node is no longer a client of r, but of new_r
        for shpnode, idx in fgraph.clients[r] + [(node, i)]:
            if isinstance(shpnode.op, Shape_i):
                idx = shpnode.op.i
                repl = self.shape_of[new_r][idx]
                if repl.owner is shpnode:
                    # This mean the replacement shape object is
                    # exactly the same as the current shape object. So
                    # no need for replacement.
                    continue
                if (
                    repl.owner
                    and repl.owner.inputs[0] is shpnode.inputs[0]
                    and isinstance(repl.owner.op, Shape_i)
                    and repl.owner.op.i == shpnode.op.i
                ):
                    # The replacement is a shape_i of the same
                    # input. So no need to do this equivalent
                    # replacement.
                    continue

                if shpnode.outputs[0] in ancestors([repl]):
                    raise InconsistencyError(
                        "This substitution would insert a cycle in the graph:"
                        f"node: {node}, i: {i}, r: {r}, new_r: {new_r}"
                    )

                self.scheduled[shpnode] = new_r
        # In case 2, if r is a variable that we've scheduled for shape update,
        # then we should cancel it.
        unscheduled = [k for k, v in self.scheduled.items() if v == r]
        for k in unscheduled:
            del self.scheduled[k]

        # In either case, r could be in shape_of.values(), that is, r itself
        # is the shape of  something. In that case, we want to update
        # the value in shape_of, to keep it up-to-date.
        for v in self.shape_of_reverse_index.get(r, []):
            # The reverse index is only approximate. It is not updated on
            # deletion of variables, or on change_input so it might be the
            # case that there are a few extra `v`'s in it that no longer have
            # a shape of r or possibly have been deleted from shape_of
            # entirely. The important thing is that it permits to recall
            # all variables with r in their shape.
            for ii, svi in enumerate(self.shape_of.get(v, [])):
                if svi == r:
                    self.set_shape_i(v, ii, new_r)
        self.shape_of_reverse_index[r] = set()

    def same_shape(
        self,
        x: Variable,
        y: Variable,
        dim_x: int | None = None,
        dim_y: int | None = None,
    ) -> bool:
        """Return ``True`` if `x` and `y` have the same shape.

        Parameters
        ==========
        x
            The `Variable` for which its shape is to be compared with `y`'s shape.
        y
            The `Variable` for which its shape is to be compared with `x`'s shape.
        dim_x
            If non ``None``, compare only the dimension of `x` equal to
            `dim_x`.
        dim_y
            If non ``None``, compare only the dimension of `y` equal to
            `dim_y`.

        """
        sx = self.shape_of[x]
        sy = self.shape_of[y]

        if sx is None or sy is None:
            return False

        if dim_x is not None:
            sx = [sx[dim_x]]

        if dim_y is not None:
            sy = [sy[dim_y]]

        if len(sx) != len(sy):
            return False

        # Canonicalize the graphs so that comparisons are reasonable
        # TODO FIXME: This should *not* need to be performed manually here.
        # Instead, the shape information in `self.shape_of` should be operated
        # upon alongside all the other elements in a `FunctionGraph` (e.g. as
        # if `self.shape_of.values()` were additional outputs).
        shapes_fg = FunctionGraph(
            outputs=sx + sy,
            # features=[self],
            clone=True,
            # copy_inputs=False,
        )
        from pytensor.graph.rewriting.utils import rewrite_graph

        canon_shapes_fg = type_cast(
            FunctionGraph,
            rewrite_graph(shapes_fg, custom_rewrite=topo_constant_folding),
        )
        canon_shapes = canon_shapes_fg.outputs

        sx = canon_shapes[: len(sx)]
        sy = canon_shapes[len(sx) :]

        for dx, dy in zip(sx, sy, strict=True):
            if not equal_computations([dx], [dy]):
                return False

        return True

    def clone(self):
        return type(self)()


class ShapeOptimizer(GraphRewriter):
    """Rewriter that adds `ShapeFeature` as a feature."""

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ShapeFeature())

    def apply(self, fgraph):
        pass


class UnShapeOptimizer(GraphRewriter):
    """Rewriter that removes `ShapeFeature` as a feature."""

    def apply(self, fgraph):
        for feature in fgraph._features:
            if isinstance(feature, ShapeFeature):
                fgraph.remove_feature(feature)


# Register it after merge1 optimization at 0. We don't want to track
# the shape of merged node.
pytensor.compile.mode.optdb.register(
    "ShapeOpt", ShapeOptimizer(), "fast_run", "fast_compile", position=0.1
)
# Not enabled by default for now. Some crossentropy opt use the
# shape_feature.  They are at step 2.01. uncanonicalize is at step
# 3. After it goes to 48.5 that move to the gpu. So 10 seems reasonable.
pytensor.compile.mode.optdb.register("UnShapeOpt", UnShapeOptimizer(), position=10)


@register_useless
@register_canonicalize
@node_rewriter([Reshape])
def local_useless_expand_dims_in_reshape(fgraph, node):
    """
    Removes useless expand_dims `DimShuffle` operations inside Reshape:
      reshape(expand_dims(vector, axis=0), shp) => reshape(vector, shp)
      reshape(expand_dims(matrix, axis=(0, 2), shp) => reshape(matrix, shp)

    Implicit (and useless) squeezes are kept in the graph, as they are
    part of the canonical form of the graph.
    """
    expanded_x, new_shape = node.inputs

    if not (
        expanded_x.owner is not None
        and isinstance(expanded_x.owner.op, DimShuffle)
        and expanded_x.owner.op.augment
    ):
        return False

    [x] = expanded_x.owner.inputs

    new_order = tuple(o for o in expanded_x.owner.op.new_order if o != "x")
    if new_order != tuple(range(x.type.ndim)):
        x = x.dimshuffle(new_order)

    new_reshaped_x = x.reshape(new_shape)
    copy_stack_trace(node.outputs[0], new_reshaped_x)
    return [new_reshaped_x]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Reshape])
def local_reshape_chain(fgraph, node):
    """
    Reshape(Reshape(x, shape1),shape2) -> Reshape(x, shape2)

    """
    inner_reshape, final_shape = node.inputs

    if not (inner_reshape.owner and isinstance(inner_reshape.owner.op, Reshape)):
        return None

    x, _ = inner_reshape.owner.inputs
    new_reshape = node.op(x, final_shape)

    copy_stack_trace(node.outputs, new_reshape)
    return [new_reshape]


def _is_shape_i_of_x(
    var: TensorVariable,
    x: TensorVariable,
    i: int,
    shape_feature: ShapeFeature | None = None,
) -> bool:
    if var.type.ndim != 0:
        return False

    constant_var = get_scalar_constant_value(
        var,
        only_process_constants=False,
        # Don't go through Elemwise to keep things fast
        elemwise=False,
        raise_not_constant=False,
    )

    # Check var is a constant expression with the same value as x.type.shape[i]
    if constant_var == x.type.shape[i]:
        return True

    # Match shape_of[x][i] or its constant equivalent
    if shape_feature is not None:
        i_shape_of_x = shape_feature.get_shape(x, i)
        if i_shape_of_x == var or (
            isinstance(i_shape_of_x, Constant) and (i_shape_of_x.data == constant_var)
        ):
            return True

    if var.owner is None:
        # No more constant possibilities
        return False

    # Match Shape_i{i}(x)
    if isinstance(var.owner.op, Shape_i):
        return (var.owner.op.i == i) and (var.owner.inputs[0] == x)  # type: ignore

    # Match Subtensor((ScalarType,))(Shape(input), i)
    if isinstance(var.owner.op, Subtensor):
        return (
            # Check we have integer indexing operation
            # (and not slice or multiple indexing)
            len(var.owner.op.idx_list) == 1
            and isinstance(var.owner.op.idx_list[0], ScalarType)
            # Check we are indexing on the shape of x
            and var.owner.inputs[0].owner is not None
            and isinstance(var.owner.inputs[0].owner.op, Shape)
            and var.owner.inputs[0].owner.inputs[0] == x
            # Check that index == i
            and (
                get_scalar_constant_value(var.owner.inputs[1], raise_not_constant=False)
                == i
            )
        )

    return False


def _unpack_shape_vector(shape: TensorVariable) -> tuple[TensorVariable, ...]:
    """Return the elements of a symbolic vector representing a shape.

    Handles the most common constant vector or make_vector cases.

    Returns tuple(shape) as fallback.
    """
    if isinstance(shape, Constant):
        return tuple(as_tensor_variable(dim, ndim=0) for dim in shape.data)
    elif shape.owner and isinstance(shape.owner.op, MakeVector):
        return tuple(shape.owner.inputs)
    else:
        return tuple(shape)


@register_useless("shape_unsafe")
@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Reshape])
def local_useless_reshape(fgraph, node):
    """Remove two kinds of useless `Reshape`.

    - Remove `Reshape` when both the input and output have a single dimension.
    - Remove `Reshape` when reshaping to the shape of the input.

    """
    inp, output_shape = node.inputs
    [output] = node.outputs

    if inp.type.ndim != output.type.ndim:
        return False

    # Simple case: both input and output have a single dimension.
    if (
        inp.type.ndim == 1
        and output.type.ndim == 1
        and inp.type.broadcastable == output.type.broadcastable
    ):
        return [inp]

    # Second case: all the shapes match the input shape
    # Match Reshape(x, x.shape)
    if output_shape.owner and isinstance(output_shape.owner.op, Shape):
        shape_input = output_shape.owner.inputs[0]
        if shape_input == inp:
            return [inp]

    shape_feature = getattr(fgraph, "shape_feature", None)

    # Match case where at least (n-1) entries correspond to the original shape:
    # Reshape(x, [x.shape[0], ..., x.shape[-1]]), or Reshape(x, [x.shape[0], y, x.shape[2], ... x.shape[-1]])
    # Where y can be -1 or anything with an unknown value, since the only valid reshape is still a no reshape.
    output_shape_is = _unpack_shape_vector(output_shape)
    nb_m1 = 0
    shape_match = [False] * inp.type.ndim
    for dim in range(inp.type.ndim):
        outshp_i = output_shape_is[dim]
        if _is_shape_i_of_x(outshp_i, inp, dim, shape_feature=shape_feature):
            shape_match[dim] = True
        elif isinstance(outshp_i, Constant) and outshp_i.data == -1:
            shape_match[dim] = True
            nb_m1 += 1

    if nb_m1 <= 1 and all(shape_match):
        return [inp]  # This is provably correct

    # There is one missing match, but all other dimensions match
    # Such as x.type.shape == (3, 5, None) and output_shape == (3, 5, y)
    if (nb_m1 == 0) and (shape_match.count(False) == 1):
        return [inp]  # This could mask a shape error

    return False


@register_canonicalize("shape_unsafe")
@node_rewriter([Reshape])
def local_reshape_to_dimshuffle(fgraph, node):
    r"""Remove `Reshape` operations over length-1 (broadcastable) dimensions.

    It's always valid to squeeze an input before doing the same reshape operation.
    Equivalently, it's always valid to remove `1` entries from the reshape shape
    and replace them by an expand_dims after the rewritten reshape operation.

    We chose to canonicalize the graph in this way as it allows isolating
    operations that are unique to the reshaping operation (mixing dimensions)
    from those that can be more legibly encoded by DimShuffle (squeeze and expand_dims).
    This can allow further simplifications by other rewrites that target
    DimShuffle but not Reshape, as well as facilitate the removal of useless reshape operations.

    For example:
        - reshape(col, (m, n)) -> reshape(squeeze(col, axis=1), (m, n))
        - reshape(col, (1, m, n)) -> expand_dims(reshape(squeeze(col, axis=1), (m, n)), axis=0)
        - reshape(x, (1, m, 1, n, 1, 1)) -> expand_dims(reshape(x, (m, n)), axis=(0, 2, 4, 5))

    """
    inp, output_shape = node.inputs
    [output] = node.outputs

    # Trivial case, all dimensions of input/output are known to be broadcastable:
    # there's nothing to reshape
    if all(inp.type.broadcastable) or all(output.type.broadcastable):
        squeeze_axes = tuple(range(inp.type.ndim))
        new_output_shape = []
        expand_axes = tuple(range(output.type.ndim))

    else:
        squeeze_axes = [i for i, bcast in enumerate(inp.type.broadcastable) if bcast]
        unpacked_shape = _unpack_shape_vector(output_shape)
        new_output_shape = []
        expand_axes = []
        for i, dim_length in enumerate(unpacked_shape):
            if isinstance(dim_length, Constant) and (
                dim_length.data == 1
                # -1 can be an implicit expand_dims, but it's tricky to prove
                # as we would need to check whether all other dimensions
                # already explain the full size of the array.
                # Example: np.zeros((2, 2, 2)).reshape((8, -1))
                # We rely on the output static shape which will already have figured
                # it out for some (but not all) cases
                or (dim_length.data == -1 and output.type.shape[i] == 1)
            ):
                expand_axes.append(i)
            else:
                new_output_shape.append(dim_length)

    if squeeze_axes or expand_axes:
        new_out = inp.squeeze(squeeze_axes)

        if new_output_shape:
            new_out = new_out.reshape(new_output_shape)
            copy_stack_trace(output, new_out)

        new_out = expand_dims(new_out, expand_axes)

        if not new_output_shape:
            # Eagerly merge consecutive squeeze and expand_dims
            new_out = apply_local_dimshuffle_lift(fgraph, new_out)

        copy_stack_trace(output, new_out)
        return [new_out]


@register_specialize
@node_rewriter([Reshape])
def local_fuse_squeeze_reshape(fgraph, node):
    r"""If there is a squeeze right before a reshape, merge them.

    This undoes the effect of `local_reshape_to_dimshuffle` that is applied during canonicalization.
    """
    x, new_shape = node.inputs

    if (
        x.owner is not None
        and isinstance(x.owner.op, DimShuffle)
        and x.owner.op.is_squeeze
    ):
        # A reshape can always subsume a squeeze.
        x = x.owner.inputs[0]
        return [x.reshape(new_shape)]


@register_specialize
@node_rewriter([DimShuffle])
def local_fuse_expand_dims_reshape(fgraph, node):
    r"""If there is an expand_dims right after a reshape, merge them.

    This undoes the effect of `local_reshape_to_dimshuffle` that is applied during canonicalization.
    """
    if not node.op.is_expand_dims:
        return None

    reshaped_x = node.inputs[0]

    if not (reshaped_x.owner and isinstance(reshaped_x.owner.op, Reshape)):
        return None

    if len(fgraph.clients[reshaped_x]) > 1:
        # The reshape is used elsewhere, don't fuse as it can sometimes require a copy.
        # Example: `x = pt.matrix(); y = x.T.reshape(-1); out = y[: None] * y[None, :]`
        return None

    x, new_shape = reshaped_x.owner.inputs

    # Add expand_dims to shape
    new_shape = list(_unpack_shape_vector(new_shape))
    for i in node.op.augment:
        new_shape.insert(i, 1)

    new_reshaped_x = x.reshape(new_shape)
    copy_stack_trace(node.outputs[0], new_reshaped_x)
    return [new_reshaped_x]


@register_canonicalize
@register_specialize
@node_rewriter([Reshape])
def local_reshape_lift(fgraph, node):
    """
        Reshape(UnaryElemwise(x)) -> UnaryElemwise(Reshape(x))

    Notes
    -----
    This rewrite is needed by `log1msigm_to_softplus` in order to get applied
    when there is a reshape.

    """
    if (
        isinstance(node.op, Reshape)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and len(node.inputs[0].owner.inputs) == 1
    ):
        r = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])
        # Copy stacktrace from previous Reshape op, as an error in new
        # Reshape op could only have been caused by old one.
        copy_stack_trace(node.outputs, r)

        e = node.inputs[0].owner.op(r)
        # Copy stacktrace from both previous Reshape and UnaryElemwise op
        # because an error in new cg could have been caused by either ops.
        copy_stack_trace(node.outputs + node.inputs, e)
        return [e]


@register_useless
@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([SpecifyShape])
def local_useless_specify_shape(fgraph, node):
    """Remove SpecifyShape when the asserted shapes are already encoded in the static type of the input."""
    x, *shape = node.inputs
    for static_dim, specified_dim in zip(x.type.shape, shape, strict=True):
        if isinstance(specified_dim.type, NoneTypeT):
            continue
        if static_dim is None:
            # There is an unknown static dimension that is being specified
            return None
        if not (
            isinstance(specified_dim, Constant) and specified_dim.data == static_dim
        ):
            # The specified dim is either:
            # 1. Not constant or
            # 2. Constant that does not match the static dim
            # Either way, we must keep the SpecifyShape
            return None

    # If we arrived here, it means SpecifyShape was already encoded in the static shape
    # We don't need it
    copy_stack_trace(node.outputs[0], x)
    return [x]


@register_infer_shape
@register_useless
@register_canonicalize
@node_rewriter([SpecifyShape])
def local_merge_consecutive_specify_shape(fgraph, node):
    """Replace ``specify_shape(specify_shape(x, s1), s2)`` with ``specify_shape(x, s3)``,
    where s3 is the union of specified dimensions in s1 and s2, with preference given to s2.
    """

    if not isinstance(node.op, SpecifyShape):
        return False

    obj = node.inputs[0]
    if not (obj.owner and isinstance(obj.owner.op, SpecifyShape)):
        return False

    inner_obj, *shape = obj.owner.inputs
    for dim, sh in enumerate(node.inputs[1:]):
        if not isinstance(sh.type, NoneTypeT):
            shape[dim] = sh

    # TODO: We could make sure that the overlapping shapes of the two `SpecifyShape`s are
    # the same.

    return [specify_shape(inner_obj, shape)]


_empty_shape = constant([], dtype="int64")


@register_infer_shape
@node_rewriter([Shape])
def local_shape_ground(fgraph, node):
    """Rewrite shape(x) -> make_vector(x.type.shape) when this is constant."""
    [x] = node.inputs
    static_shape = x.type.shape
    if len(static_shape) == 0:
        return [_empty_shape]
    if not any(dim is None for dim in static_shape):
        return [stack([constant(dim, dtype="int64") for dim in static_shape])]


@register_infer_shape
@register_useless
@register_canonicalize
@node_rewriter([Shape])
def local_Shape_of_SpecifyShape(fgraph, node):
    """Replace ``specify_shape(x, s).shape`` with ``s``."""

    if not isinstance(node.op, Shape):
        return False

    specified_shape = node.inputs[0]

    if not (
        specified_shape.owner is not None
        and isinstance(specified_shape.owner.op, SpecifyShape)
    ):
        return False

    x, *shape = specified_shape.owner.inputs

    # Replace `NoneConst` by `shape_i`
    for i, sh in enumerate(shape):
        if isinstance(sh.type, NoneTypeT):
            shape[i] = x.shape[i]

    return [stack(shape).astype(np.int64)]


@register_infer_shape
@register_canonicalize
@register_specialize
@node_rewriter([SpecifyShape])
def local_specify_shape_lift(fgraph, node):
    """Lift SpecifyShape of Elemwise towards the inputs."""
    inp, *shape = node.inputs
    if inp.owner and isinstance(inp.owner.op, Elemwise):
        if len(inp.owner.outputs) != 1:
            return None

        elem_inps = inp.owner.inputs
        if len(elem_inps) == 1:
            new_elem_inps = [specify_shape(elem_inps[0], shape)]
        else:
            # Rewrite does not support case where specify_shape provides new broadcastable information,
            # As that may require a specify_shape for each input
            out_broadcastable = node.outputs[0].type.broadcastable
            if out_broadcastable != inp.type.broadcastable:
                return None

            # All non-broadcastable dimensions of inputs must match the non-broadcastbale specify_shape dims
            # We look for a sufficient input to assign all the specify_shape dims
            # We could consider distributing the SpecifyShape across multiple inputs, when none is sufficient

            nonbcast_dims = {
                i
                for i, (dim, bcast) in enumerate(
                    zip(shape, out_broadcastable, strict=True)
                )
                if (not bcast and not isinstance(dim.type, NoneTypeT))
            }
            new_elem_inps = elem_inps.copy()
            for i, elem_inp in enumerate(elem_inps):
                if all(
                    bcast_dim is False
                    for dim, bcast_dim in enumerate(elem_inp.type.broadcastable)
                    if dim in nonbcast_dims
                ):
                    new_elem_inps[i] = specify_shape(elem_inp, shape)
                    break
            else:  # no-break, no sufficient candidate found
                return None

        new_out = inp.owner.op.make_node(*new_elem_inps).outputs
        copy_stack_trace(node.outputs, new_out)
        return new_out


@register_infer_shape
@register_useless
@register_canonicalize
@node_rewriter([Shape_i])
def local_Shape_i_ground(fgraph, node):
    """Replace ``shape_i(x, i)`` with ``s`` when ``x.type.shape[i] == s``."""

    if not isinstance(node.op, Shape_i):
        return False

    shape_arg = node.inputs[0]

    if not isinstance(shape_arg.type, TensorType):
        return False

    s_val = shape_arg.type.shape[node.op.i]
    if s_val is not None:
        return [as_tensor_variable(s_val, dtype=np.int64)]


@register_infer_shape
@register_specialize
@register_canonicalize
@node_rewriter([Shape])
def local_shape_to_shape_i(fgraph, node):
    if isinstance(node.op, Shape):
        if not hasattr(fgraph, "shape_feature"):
            return
        shape_feature = fgraph.shape_feature
        ret = shape_feature.make_vector_shape(node.inputs[0])

        # We need to copy over stack trace from input to output
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_specialize
@register_canonicalize
@node_rewriter([Shape_i])
def local_track_shape_i(fgraph, node):
    if not isinstance(node.op, Shape_i):
        return False

    try:
        shape_feature = fgraph.shape_feature
    except AttributeError:
        return False

    if node not in shape_feature.scheduled:
        return False

    # Don't unschedule node as it could be reinserted in the
    # fgraph as we don't change it in the shapefeature internal
    # structure.
    replacement = shape_feature.scheduled[node]
    return [shape_feature.shape_of[replacement][node.op.i]]
