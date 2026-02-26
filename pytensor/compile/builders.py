"""Define new Ops from existing Ops"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from copy import copy
from functools import partial
from itertools import chain
from typing import cast

from pytensor.compile.function import function
from pytensor.compile.function.pfunc import rebuild_collect_shared
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.configdefaults import config
from pytensor.gradient import DisconnectedType, Rop, disconnected_type, grad
from pytensor.graph.basic import (
    Apply,
    Constant,
    NominalVariable,
    Variable,
)
from pytensor.graph.fg import FrozenFunctionGraph, FunctionGraph
from pytensor.graph.null_type import NullType
from pytensor.graph.op import HasInnerGraph, Op, io_connection_pattern
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import graph_inputs
from pytensor.graph.utils import MissingInputError


def infer_shape(outs, inputs, input_shapes):
    """
    Compute the shape of the outputs given the shape of the inputs of an PyTensor
    graph.

    We do it this way to avoid compiling the inner function just to get
    the shape. Changes to ShapeFeature could require changes in this function.

    """
    # We use a ShapeFeature because it has all the necessary logic
    # inside.  We don't use the full ShapeFeature interface, but we
    # let it initialize itself with an empty fgraph, otherwise we will
    # need to do it manually

    # TODO: ShapeFeature should live elsewhere
    from pytensor.tensor.rewriting.shape import ShapeFeature

    for inp, inp_shp in zip(inputs, input_shapes, strict=True):
        if inp_shp is not None and len(inp_shp) != inp.type.ndim:
            assert len(inp_shp) == inp.type.ndim

    shape_feature = ShapeFeature()
    fgraph = FunctionGraph([], [], features=[shape_feature])
    for v in chain.from_iterable(s for s in input_shapes if s is not None):
        # Import input_shape nodes, as for some graphs ShapeFeature assumes these were seen before
        if (node := v.owner) is not None:
            fgraph.import_node(node, import_missing=True)

    # Initialize shape_of with the input shapes
    for inp, inp_shp in zip(inputs, input_shapes, strict=True):
        shape_feature.set_shape(inp, inp_shp, override=True)

    def local_traverse(out):
        """
        Go back in the graph, from out, adding computable shapes to shape_of.

        """
        if out in shape_feature.shape_of:
            # Its shape is already known
            return
        elif out.owner is None:
            # This is an input of the graph
            shape_feature.init_r(out)
        else:
            # Recurse over inputs
            for inp in out.owner.inputs:
                if inp not in shape_feature.shape_of:
                    local_traverse(inp)

            # shape_feature.on_import does not actually use an fgraph
            # It will call infer_shape and set_shape appropriately
            dummy_fgraph = None
            shape_feature.on_import(dummy_fgraph, out.owner, reason="dummy")

    ret = []
    for o in outs:
        local_traverse(o)
        ret.append(shape_feature.shape_of[o])
    return ret


def construct_nominal_fgraph(
    inputs: Sequence[Variable], outputs: Sequence[Variable]
) -> tuple[
    FunctionGraph,
    Sequence[Variable],
    dict[Variable, Variable],
    dict[Variable, Variable],
]:
    """Construct an inner-`FunctionGraph` with ordered nominal inputs."""
    implicit_shared_inputs = []

    dummy_inputs = [inp.type() for inp in inputs]
    dummy_implicit_shared_inputs = []
    for var in graph_inputs(outputs, inputs):
        if var in inputs:
            continue
        if isinstance(var, SharedVariable):
            # We allow shared inputs to be added automatically to the graph
            implicit_shared_inputs.append(var)
            dummy_implicit_shared_inputs.append(var.type())
        elif not isinstance(var, Constant):
            raise MissingInputError(f"NominalGraph is missing an input: {var}")

    replacements = dict(
        zip(
            inputs + implicit_shared_inputs,
            dummy_inputs + dummy_implicit_shared_inputs,
            strict=True,
        )
    )

    new = rebuild_collect_shared(
        cast(Sequence[Variable], outputs),
        inputs=inputs + implicit_shared_inputs,
        replace=replacements,
        copy_inputs_over=False,
    )
    (
        local_inputs,
        local_outputs,
        (_clone_d, update_d, update_expr, new_shared_inputs),
    ) = new

    assert len(local_inputs) == len(inputs) + len(implicit_shared_inputs)
    assert len(local_outputs) == len(outputs)
    assert not update_d
    assert not update_expr
    assert not new_shared_inputs

    fgraph = FunctionGraph(local_inputs, local_outputs, clone=False)

    # The inputs need to be `NominalVariable`s so that we can merge
    # inner-graphs
    nominal_local_inputs = tuple(
        NominalVariable(n, var.type) for n, var in enumerate(local_inputs)
    )

    fgraph.replace_all(zip(local_inputs, nominal_local_inputs, strict=True))

    for i, inp in enumerate(fgraph.inputs):
        nom_inp = nominal_local_inputs[i]
        fgraph.inputs[i] = nom_inp
        fgraph.clients.pop(inp, None)
        fgraph.add_input(nom_inp)

    return fgraph, implicit_shared_inputs, update_d, update_expr


class OpFromGraph(Op, HasInnerGraph):
    r"""Create an Op from inputs and outputs lists of variables.

    The signature is similar to :func:`pytensor.function` and the resulting Op's perform will do
    the same operation as ``orig_function(inputs, outputs, **kwargs)``.

    Does not support ``updates`` or ``givens``.

    .. TODO::
        - Add support for NullType and DisconnectedType when R_op supports them
        - Add optimization to removing unused inputs/outputs
        - Add optimization to work inplace on inputs when not inline

    Notes
    -----
    - Shared variables in the inner graph are supported. They are detected automatically and added
      as implicit inputs.
    - Unused inputs are supported (needed for gradient overrides).
    - Nested OpFromGraph is supported.
    - ``inline=True`` causes the Op's inner graph to be inlined during compilation, which gives
      better runtime optimization at the cost of compilation time. Currently only works with
      ``fast_compile`` or ``fast_run`` mode.
    - Override callables should be pure functions (no side effects). They are called once at the
      first call to L_op/R_op and converted to OpFromGraph instances. They are also called once at
      construction time with dummy inputs to build a frozen representation for equality comparison.
    - Two OpFromGraph instances with the same inner graph, overrides, shared variables, and settings
      are considered equal. This allows the MergeOptimizer to deduplicate identical OpFromGraph
      nodes.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        from pytensor import function, tensor as pt
        from pytensor.compile.builders import OpFromGraph

        x, y, z = pt.scalars("xyz")
        e = x + y * z
        op = OpFromGraph([x, y, z], [e])
        # op behaves like a normal pytensor op
        e2 = op(x, y, z) + op(z, y, x)
        fn = function([x, y, z], [e2])

    With a shared variable:

    .. code-block:: python

        import numpy as np
        import pytensor
        from pytensor import config, function, tensor as pt
        from pytensor.compile.builders import OpFromGraph

        x, y, z = pt.scalars("xyz")
        s = pytensor.shared(np.random.random((2, 2)).astype(config.floatX))
        e = x + y * z + s
        op = OpFromGraph([x, y, z], [e])
        e2 = op(x, y, z) + op(z, y, x)
        fn = function([x, y, z], [e2])

    Per-input L_op override:

    .. code-block:: python

        from pytensor import function, tensor as pt, grad
        from pytensor.compile.builders import OpFromGraph

        x, y, z = pt.scalars("xyz")
        e = x + y * z


        def rescale_dy(inps, outputs, out_grads):
            x, y, z = inps
            (g,) = out_grads
            return z * 2


        op = OpFromGraph([x, y, z], [e], lop_overrides=[None, rescale_dy, None])
        e2 = op(x, y, z)
        dx, dy, dz = grad(e2, [x, y, z])
        fn = function([x, y, z], [dx, dy, dz])
        # the gradient wrt y is now doubled
        fn(2.0, 3.0, 4.0)  # [1., 8., 3.]
    """

    def __init__(
        self,
        inputs: list[Variable],
        outputs: list[Variable],
        *,
        inline: bool = False,
        lop_overrides: Callable | list | OpFromGraph | None = None,
        grad_overrides: Callable | list | OpFromGraph | None = None,
        rop_overrides: Callable | list | OpFromGraph | None = None,
        connection_pattern: list[list[bool]] | None = None,
        strict: bool = False,
        name: str | None = None,
        destroy_map: dict[int, tuple[int, ...]] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        inputs : list of Variable
            The inputs to the graph.
        outputs : list of Variable
            The outputs to the graph.
        inline : bool, optional
            If True, the Op's inner graph is inlined during compilation. If False (default), a
            pre-compiled function is used instead.
        lop_overrides : callable or OpFromGraph or list or None, optional
            Override for the L_op method. Mutually exclusive with ``grad_overrides``.

            - None: use the default L_op result.
            - OpFromGraph: should accept ``(inputs, outputs, output_grads)`` with the same types
              as the inner graph.
            - callable: should take three args ``(inputs, outputs, output_grads)``, each a list of
              Variable, and return a list of Variable.
            - list: one entry per input. Each entry is None (use default), a DisconnectedType or
              NullType Variable, or a callable returning a single Variable.
        grad_overrides : callable or OpFromGraph or list or None, optional
            Deprecated in favor of ``lop_overrides``. Same as ``lop_overrides`` but the callable
            signature is ``(inputs, output_grads)`` (no ``outputs`` argument). Mutually exclusive
            with ``lop_overrides``.
        rop_overrides : callable or OpFromGraph or list or None, optional
            Override for the R_op method.

            - None: use the default R_op result.
            - OpFromGraph: should accept ``(inputs, eval_points)`` with the same types as the
              inner graph inputs.
            - callable: should take two args ``(inputs, eval_points)``, each a list of Variable,
              and return a list of Variable.
            - list: one entry per output. Each entry is None (use default), a DisconnectedType or
              NullType Variable, or a callable returning a single Variable.

            .. warning::

                R_op overrides are ignored when ``pytensor.gradient.Rop`` is called with
                ``use_op_rop_implementation=False`` (the default). In that case the L_op is used
                twice to obtain a mathematically equivalent R_op.
        connection_pattern : list of list of bool, optional
            If provided, used as the connection pattern for this Op. Each inner list has one bool
            per output, and the outer list has one entry per input.
        strict : bool, optional
            If True, raises when any variables needed to compute the inner graph are not provided
            as explicit inputs. Only relevant for graphs with shared variables. Default False.
        name : str, optional
            A name for debugging purposes.
        **kwargs
            Additional arguments passed to :func:`pytensor.function`. Only used when
            ``inline=False``.
        """
        ignore_unused_inputs = kwargs.get("on_unused_input", False) == "ignore"
        if not ignore_unused_inputs and len(inputs) != len(set(inputs)):
            var_counts = {var: inputs.count(var) for var in inputs}
            duplicated_inputs = [var for var, count in var_counts.items() if count > 1]
            raise ValueError(
                f"The following variables were provided more than once as inputs to the OpFromGraph, resulting in an "
                f"invalid graph: {duplicated_inputs}. Use dummy variables or var.copy() to distinguish "
                f"variables when creating the OpFromGraph graph."
            )

        if not (isinstance(inputs, list) and isinstance(outputs, list)):
            raise TypeError("Inputs and outputs must be lists")

        for out in outputs:
            if not isinstance(out, Variable):
                raise TypeError(
                    f"Inputs and outputs must be Variable instances; got {out}"
                )

        if "updates" in kwargs or "givens" in kwargs:
            raise NotImplementedError("Updates and givens are not supported")

        self.is_inline = inline

        self.fgraph, self.shared_inputs, _, _ = construct_nominal_fgraph(
            inputs, outputs
        )
        self._frozen_fgraph = self.fgraph.freeze()

        if strict and self.shared_inputs:
            raise ValueError(
                "All variables needed to compute inner-graph must be provided as inputs under strict=True. "
                f"The inner-graph implicitly depends on the following shared variables {self.shared_inputs}"
            )

        self.kwargs = kwargs
        self.input_types = [inp.type for inp in inputs]
        self.output_types = [out.type for out in outputs]

        for override in (lop_overrides, grad_overrides, rop_overrides):
            if override == "default":
                raise ValueError(
                    "'default' is no longer a valid value for overrides. Use None instead."
                )
            if isinstance(override, Variable):
                raise TypeError(
                    "Variables are no longer valid types for overrides. Return them in a list for each output instead"
                )

        self.lop_overrides = lop_overrides
        self.grad_overrides = grad_overrides
        self.rop_overrides = rop_overrides

        self._lop_op_interface = True
        if grad_overrides is not None:
            if lop_overrides is not None:
                raise ValueError(
                    "lop_overrides and grad_overrides are mutually exclusive"
                )
            warnings.warn(
                "grad_overrides is deprecated in favor of lop_overrides. Using it will lead to an error in the future.",
                FutureWarning,
            )
            self._lop_op_interface = False
        # Dictionary where we cache OpFromGraph that represent the L_op
        # A distinct OpFromGraph is needed to represent each pattern of output_grads connection
        # It also returns a tuple that indicates which input_gradients are disconnected
        self._lop_op_cache: dict[tuple[bool, ...], Callable] = {}
        self._rop_op_cache: Callable | None = None

        self._connection_pattern = connection_pattern

        if name is not None:
            assert isinstance(name, str), "name must be None or string object"
        self.name = name
        self.destroy_map = destroy_map if destroy_map is not None else {}

        self._frozen_lop = self._freeze_lop_override()
        self._frozen_rop = self._freeze_rop_override()

    @staticmethod
    def _freeze_override_to_fgraph(
        all_inputs: list[Variable], results: list[Variable]
    ) -> tuple[tuple[bool, ...], FrozenFunctionGraph | None]:
        """Build a FrozenFunctionGraph from override results, filtering out disconnected/null types.

        A structural connection pattern is also constructed from the results, indicating where filtering occured.
        This pattern, together with the FrozenFunctionGraph, participate in the equality comparison of OpFromGraph
        instances to distinguish different override implementations.
        """
        pattern = tuple(
            isinstance(r.type, DisconnectedType | NullType) for r in results
        )
        connected = [
            r for r, is_disc in zip(results, pattern, strict=True) if not is_disc
        ]
        if not connected:
            return pattern, None
        return pattern, FunctionGraph(all_inputs, connected).freeze()

    def _freeze_lop_override(self):
        """Freeze lop/grad override into a FrozenFunctionGraph for equality comparison."""
        lop = self.lop_overrides if self._lop_op_interface else self.grad_overrides
        if lop is None:
            return None

        if isinstance(lop, OpFromGraph):
            return lop._frozen_fgraph

        dummy_inputs = [t() for t in self.input_types]
        dummy_outputs = [t() for t in self.output_types]
        dummy_output_grads = [t() for t in self.output_types]

        if self._lop_op_interface:
            all_inputs = dummy_inputs + dummy_outputs + dummy_output_grads
            callable_args = (dummy_inputs, dummy_outputs, dummy_output_grads)
        else:
            all_inputs = dummy_inputs + dummy_output_grads
            callable_args = (dummy_inputs, dummy_output_grads)

        if isinstance(lop, list):
            results = []
            for entry in lop:
                if entry is None:
                    results.append(disconnected_type())
                elif isinstance(entry, Variable):
                    results.append(entry)
                elif callable(entry):
                    results.append(entry(*callable_args))
            return self._freeze_override_to_fgraph(all_inputs, results)

        result = lop(*callable_args)
        return self._freeze_override_to_fgraph(all_inputs, result)

    def _freeze_rop_override(self):
        """Freeze rop override into a FrozenFunctionGraph for equality comparison."""
        rop = self.rop_overrides
        if rop is None:
            return None

        if isinstance(rop, OpFromGraph):
            return rop._frozen_fgraph

        dummy_inputs = [t() for t in self.input_types]
        dummy_eval_points = [t() for t in self.input_types]
        all_inputs = dummy_inputs + dummy_eval_points
        callable_args = (dummy_inputs, dummy_eval_points)

        if isinstance(rop, list):
            results = []
            for entry in rop:
                if entry is None:
                    results.append(disconnected_type())
                elif isinstance(entry, Variable):
                    results.append(entry)
                elif callable(entry):
                    results.append(entry(*callable_args))
            return self._freeze_override_to_fgraph(all_inputs, results)

        result = rop(*callable_args)
        return self._freeze_override_to_fgraph(all_inputs, result)

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        if self._frozen_fgraph != other._frozen_fgraph:
            return False
        if self.is_inline != other.is_inline:
            return False
        if self.destroy_map != other.destroy_map:
            return False
        if len(self.shared_inputs) != len(other.shared_inputs):
            return False
        if any(
            a is not b
            for a, b in zip(self.shared_inputs, other.shared_inputs, strict=True)
        ):
            return False
        if self._frozen_lop != other._frozen_lop:
            return False
        if self._frozen_rop != other._frozen_rop:
            return False
        return True

    def __hash__(self):
        return hash((type(self), self._frozen_fgraph, self.is_inline))

    def __str__(self):
        name = self.__class__.__name__ if self.name is None else self.name
        is_inline = self.is_inline
        return f"{name}{{inline={is_inline}}}"

    def _combine_list_overrides(self, default_outs, custom_outs, callable_args):
        """Combines default and custom overrides into a single list of outputs."""
        default_out_iter = iter(default_outs)
        combined_outs = []
        for custom_out in custom_outs:
            if custom_out is None:
                combined_outs.append(next(default_out_iter))
            elif isinstance(custom_out, Variable):
                if not isinstance(custom_out.type, NullType | DisconnectedType):
                    raise ValueError(
                        f"Override list can only contain NullType or DisconnectedType Variable instances, got {custom_out.type}"
                    )
                combined_outs.append(custom_out)
            elif callable(custom_out):
                combined_outs.append(custom_out(*callable_args))
            else:
                raise ValueError(
                    f"Override list should contain None, Variable or callable, got {type(custom_out)}"
                )
        return combined_outs

    def _call_custom_override(self, op_overrides, callable_args, nout):
        """Calls custom override function and provides informative error messages."""
        if not callable(op_overrides):
            raise TypeError(
                f"L_op/R_op override should be None, a list or a Callable, got {type(op_overrides)}"
            )
        outputs = op_overrides(*callable_args)
        if not isinstance(outputs, list):
            raise TypeError(
                f"Lop/Rop overriding function should return a list, got {type(outputs)}"
            )
        if len(outputs) != nout:
            raise ValueError(
                f"Lop/Rop overriding function {self.rop_overrides} should return "
                f"a list of {nout} outputs, got {len(outputs)}"
            )
        return outputs

    @config.change_flags(compute_test_value="off")
    def _build_and_cache_lop_op(
        self, disconnected_output_grads: tuple[bool, ...]
    ) -> Callable:
        """converts lop_overrides (or grad_overrides) from user supplied form to type(self) instance,
        specialized for the pattern of disconnected_output_grads

        Results are cached in self._lop_op_cache
        """
        try:
            return self._lop_op_cache[disconnected_output_grads]
        except KeyError:
            pass

        inner_inputs = self.inner_inputs
        inner_outputs = self.inner_outputs
        nin = len(inner_inputs)
        nout = len(inner_outputs)
        lop_overrides = (
            self.lop_overrides if self._lop_op_interface else self.grad_overrides
        )

        if isinstance(lop_overrides, OpFromGraph):
            if self._lop_op_interface:
                self._lop_op_cache[disconnected_output_grads] = lop_overrides
                lop_overrides.kwargs["on_unused_input"] = "ignore"
                return lop_overrides

            else:
                # We need to add a wrapper for the different input signature
                # TODO: Remove this once the grad interface is gone
                def lop_overrides(inps, grads):
                    return self.grad_overrides(*inps, *grads)

        # We try to compute the gradient with respect to connected outputs only
        connected_inner_outputs = [
            # We add an identity operation(copy) so that we don't override indirect
            # gradient contributions to an inner output coming from other inner outputs
            inner_out.copy()
            for inner_out, disconnected in zip(
                inner_outputs, disconnected_output_grads, strict=True
            )
            if not disconnected
        ]
        connected_output_grads = [
            out_t()
            for out_t, disconnected in zip(
                self.output_types, disconnected_output_grads, strict=True
            )
            if not disconnected
        ]
        fn_grad = partial(
            grad,
            cost=None,
            disconnected_inputs="ignore",
            return_disconnected="disconnected",
            null_gradients="return",
            known_grads=dict(
                zip(connected_inner_outputs, connected_output_grads, strict=True)
            ),
        )

        if self._lop_op_interface:
            callable_args = (
                inner_inputs,
                connected_inner_outputs,
                connected_output_grads,
            )
        else:
            callable_args = (inner_inputs, connected_output_grads)

        # we need to convert _lop_op into an OfG instance
        if lop_overrides is None:
            input_grads = fn_grad(wrt=inner_inputs)
        elif isinstance(lop_overrides, list):
            custom_input_grads = lop_overrides
            if len(custom_input_grads) != nin:
                raise ValueError(
                    f"Need to override {nin} gradients, got {len(custom_input_grads)}",
                    custom_input_grads,
                )
            # compute non-overriding downsteam grads from upstreams grads
            # it's normal some input may be disconnected, thus the 'ignore'
            wrt = [
                lin
                for lin, gov in zip(inner_inputs, custom_input_grads, strict=True)
                if gov is None
            ]
            default_input_grads = fn_grad(wrt=wrt) if wrt else []
            input_grads = self._combine_list_overrides(
                default_input_grads, custom_input_grads, callable_args
            )
        else:
            input_grads = self._call_custom_override(lop_overrides, callable_args, nin)

        # Filter out disconnected/null input generated from the inner graph grad
        # We append them in the outer wrapper function below
        connected_input_grads = [
            inp_grad
            for inp_grad in input_grads
            if not isinstance(inp_grad.type, DisconnectedType | NullType)
        ]
        lop_op = OpFromGraph(
            inputs=inner_inputs + connected_inner_outputs + connected_output_grads,
            outputs=connected_input_grads,
            inline=self.is_inline,
            name=(None if self.name is None else f"{self.name}_LOp"),
            # TODO: We can be eager here and exclude unused inputs in the OFG
            on_unused_input="ignore",
        )

        # Return a wrapper that combines connected and disconnected/null input gradients
        # And also filters out disconnected/null output gradients
        def wrapper(*inputs: Variable, **kwargs) -> list[Variable]:
            inputs, outputs, output_grads = (
                inputs[: -nout * 2],
                inputs[-nout * 2 : -nout],
                inputs[-nout:],
            )
            connected_outputs = [
                output
                for output, output_grad in zip(outputs, output_grads, strict=True)
                if not isinstance(output_grad.type, DisconnectedType | NullType)
            ]
            connected_output_grads = [
                output_grad
                for output_grad in output_grads
                if not isinstance(output_grad.type, DisconnectedType)
            ]
            connected_input_grads = iter(
                lop_op(*inputs, *connected_outputs, *connected_output_grads, **kwargs)
            )
            return [
                input_grad
                if isinstance(input_grad.type, DisconnectedType | NullType)
                else next(connected_input_grads)
                for input_grad in input_grads
            ]

        self._lop_op_cache[disconnected_output_grads] = wrapper
        return wrapper

    @config.change_flags(compute_test_value="off")
    def _build_and_cache_rop_op(self):
        """Converts rop_overrides from user supplied form to type(self) instance.

        Results are cached in self._rop_op_cache
        """
        if self._rop_op_cache is not None:
            return self._rop_op_cache

        inner_inputs = self.inner_inputs
        inner_outputs = self.inner_outputs
        nout = len(inner_outputs)
        rop_overrides = self.rop_overrides

        if isinstance(rop_overrides, OpFromGraph):
            self._rop_op_cache = rop_overrides
            return rop_overrides

        eval_points = [inp_t() for inp_t in self.input_types]
        fn_rop = partial(
            Rop,
            wrt=inner_inputs,
            eval_points=eval_points,
            use_op_rop_implementation=True,
        )

        callable_args = (inner_inputs, eval_points)
        if rop_overrides is None:
            output_grads = fn_rop(f=inner_outputs)
        elif isinstance(rop_overrides, list):
            custom_output_grads = rop_overrides
            if len(custom_output_grads) != nout:
                raise ValueError(
                    f"Need to override {int(nout)} Rop, got {len(custom_output_grads)}",
                    custom_output_grads,
                )
            # get outputs that does not have Rop override
            f = [
                output
                for output, custom_output_grad in zip(
                    inner_outputs, custom_output_grads, strict=True
                )
                if custom_output_grad is None
            ]
            default_output_grads = fn_rop(f=f) if f else []
            output_grads = self._combine_list_overrides(
                default_output_grads, custom_output_grads, callable_args
            )
        else:
            output_grads = self._call_custom_override(
                rop_overrides, callable_args, nout
            )

        # Filter out disconnected output gradients
        filtered_output_grads = [
            out_grad
            for out_grad in output_grads
            if not isinstance(out_grad.type, DisconnectedType | NullType)
        ]
        rop_op = OpFromGraph(
            inputs=inner_inputs + eval_points,
            outputs=filtered_output_grads,
            inline=self.is_inline,
            name=(None if self.name is None else self.name + "_rop"),
            on_unused_input="ignore",
        )

        # Return a wrapper that combines connected and disconnected output gradients
        def wrapper(*inputs: Variable, **kwargs) -> list[Variable | None]:
            connected_output_grads = iter(rop_op(*inputs, **kwargs))
            all_output_grads = []
            for out_grad in output_grads:
                if isinstance(out_grad.type, DisconnectedType):
                    # R_Op does not have DisconnectedType yet, None should be used instead
                    all_output_grads.append(None)
                elif isinstance(out_grad.type, NullType):
                    all_output_grads.append(out_grad)
                else:
                    all_output_grads.append(next(connected_output_grads))
            return all_output_grads

        self._rop_op_cache = wrapper
        return wrapper

    def L_op(self, inputs, outputs, output_grads):
        disconnected_output_grads = tuple(
            isinstance(og.type, DisconnectedType) for og in output_grads
        )
        lop_op = self._build_and_cache_lop_op(disconnected_output_grads)
        return lop_op(*inputs, *outputs, *output_grads, return_list=True)

    def R_op(self, inputs, eval_points):
        rop_op = self._build_and_cache_rop_op()
        return rop_op(*inputs, *eval_points, return_list=True)

    def __call__(self, *inputs, **kwargs):
        # The user interface doesn't expect the shared variable inputs of the
        # inner-graph, but, since `Op.make_node` does (and `Op.__call__`
        # dispatches to `Op.make_node`), we need to compensate here
        num_expected_inps = len(self.inner_inputs) - len(self.shared_inputs)

        if len(inputs) == num_expected_inps:
            actual_inputs = inputs + tuple(self.shared_inputs)
            return super().__call__(*actual_inputs, **kwargs)
        elif len(inputs) == len(self.inner_inputs):
            return super().__call__(*inputs, **kwargs)
        else:
            raise ValueError(f"Expected at least {num_expected_inps} input(s)")

    def make_node(self, *inputs):
        # The `inputs` received here should correspond to the inputs in the
        # `Apply` nodes we produce below
        if len(inputs) != len(self.inner_inputs):
            raise ValueError(f"Expected {len(self.inner_inputs)} input(s)")

        num_expected_inps = len(self.inner_inputs) - len(self.shared_inputs)
        non_shared_inputs = inputs[:num_expected_inps]

        non_shared_inputs = [
            inp_t.filter_variable(inp)
            for inp, inp_t in zip(non_shared_inputs, self.input_types, strict=True)
        ]

        new_shared_inputs = inputs[num_expected_inps:]
        inner_and_input_shareds = list(
            zip(self.shared_inputs, new_shared_inputs, strict=True)
        )

        if not all(inp_s == inn_s for inn_s, inp_s in inner_and_input_shareds):
            # The shared variables are not equal to the original shared
            # variables, so we construct a new `Op` that uses the new shared
            # variables instead.
            replace = dict(
                zip(
                    self.inner_inputs[num_expected_inps:],
                    new_shared_inputs,
                    strict=True,
                )
            )

            # If the new shared variables are inconsistent with the inner-graph,
            # such errors should arise in this step
            new_inner_outputs = clone_replace(
                self.inner_outputs, replace=replace, copy_inputs_over=True
            )

            # It's possible that the new shared variable inputs aren't actually
            # shared variables.  When they aren't we need to add them as new
            # inputs.
            unshared_inputs = [
                inp for inp in new_shared_inputs if not isinstance(inp, SharedVariable)
            ]
            new_inner_inputs = self.inner_inputs[:num_expected_inps] + unshared_inputs

            new_op = type(self)(
                inputs=new_inner_inputs,
                outputs=new_inner_outputs,
                inline=self.is_inline,
                lop_overrides=self.lop_overrides,
                grad_overrides=self.grad_overrides,
                rop_overrides=self.rop_overrides,
                connection_pattern=self._connection_pattern,
                name=self.name,
                destroy_map=self.destroy_map,
                **self.kwargs,
            )
            new_inputs = (
                list(non_shared_inputs) + unshared_inputs + new_op.shared_inputs
            )
        else:
            new_op = self
            new_inputs = list(non_shared_inputs) + new_op.shared_inputs

        apply_node = Apply(
            new_op,
            new_inputs,
            [type() for type in new_op.output_types],
        )
        return apply_node

    def connection_pattern(self, node):
        """
        Return connection pattern of subfgraph defined by inputs and outputs.

        """
        if self._connection_pattern is not None:
            return self._connection_pattern

        ret = io_connection_pattern(self.inner_inputs, self.inner_outputs)
        self._connection_pattern = ret
        return ret

    def infer_shape(self, fgraph, node, shapes):
        # TODO: Use `fgraph.shape_feature` to do this instead.
        out_shapes = infer_shape(self.inner_outputs, self.inner_inputs, shapes)

        # Clone the output shape so that shape are computed from outer inputs.
        # Note:
        # Here we could do it more simply like:
        # `ret = [pytensor.clone_replace(shp, replace=repl) for shp in out_shp]`
        # But doing it multiple time could duplicate common subgraph between
        # each shape call. PyTensor optimizer will clean this up later, but this
        # will make extra work for the optimizer.

        repl = dict(zip(self.inner_inputs, node.inputs, strict=True))
        clone_out_shapes = [s for s in out_shapes if isinstance(s, tuple)]
        cloned = clone_replace(sum(clone_out_shapes, ()), replace=repl)
        ret = []
        used = 0
        for i, out_shape in enumerate(out_shapes):
            if out_shape is None:
                ret.append(None)
            else:
                nb = len(out_shape)
                ret.append(cloned[used : used + nb])
                used += nb

        return ret

    @property
    def fn(self):
        """Lazily compile the inner function graph."""
        if getattr(self, "_fn", None) is not None:
            return self._fn

        self._fn = function(self.inner_inputs, self.inner_outputs, **self.kwargs)
        self._fn.trust_input = True

        return self._fn

    @property
    def inner_inputs(self):
        return self.fgraph.inputs

    @property
    def inner_outputs(self):
        return self.fgraph.outputs

    def clone(self):
        res = copy(self)
        res.fgraph = res.fgraph.clone(clone_inner_graphs=True)
        return res

    def perform(self, node, inputs, outputs):
        variables = self.fn(*inputs)
        # zip strict not specified because we are in a hot loop
        for output, variable in zip(outputs, variables):
            output[0] = variable
