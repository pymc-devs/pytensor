from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    Self,
    TypeVar,
    cast,
)

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.traversal import io_toposort
from pytensor.graph.utils import (
    MetaObject,
    add_tag_trace,
)


if TYPE_CHECKING:
    from pytensor.compile.function.types import Function
    from pytensor.graph.fg import FunctionGraph
    from pytensor.graph.type import Type

StorageCellType = list[Any | None]
StorageMapType = dict[Variable, StorageCellType]
ComputeMapType = dict[Variable, list[bool]]
InputStorageType = list[StorageCellType]
OutputStorageType = list[StorageCellType]
PerformMethodType = Callable[[Apply, list[Any], OutputStorageType], None]
BasicThunkType = Callable[[], None]
ThunkCallableType = Callable[
    [PerformMethodType, StorageMapType, ComputeMapType, Apply], None
]

C = TypeVar("C", bound=Callable)


class ThunkType(Protocol[C]):
    inputs: list[list[list[Any] | None]]
    outputs: list[list[list[Any] | None]]
    lazy: bool
    __call__: C
    perform: PerformMethodType


def is_thunk_type(thunk: ThunkCallableType) -> ThunkType:
    res = cast(ThunkType, thunk)
    return res


OpOutputType = TypeVar("OpOutputType", bound=Variable)


class Op(MetaObject, Generic[OpOutputType]):
    """A class that models and constructs operations in a graph.

    A `Op` instance has several responsibilities:

    * construct `Apply` nodes via :meth:`Op.make_node` method,
    * perform the numeric calculation of the modeled operation via the
      :meth:`Op.perform` method,
    * and (optionally) build the gradient-calculating sub-graphs via the
      :meth:`Op.grad` method.

    To see how `Op`, `Type`, `Variable`, and `Apply` fit together see the
    page on :doc:`graph`.

    For more details regarding how these methods should behave: see the `Op
    Contract` in the sphinx docs (advanced tutorial on `Op` making).

    """

    default_output: int | None = None
    """
    An ``int`` that specifies which output :meth:`Op.__call__` should return.  If
    ``None``, then all outputs are returned.

    A subclass should not change this class variable, but instead override it
    with a subclass variable or an instance variable.

    """

    view_map: dict[int, list[int]] = {}
    """
    A ``dict`` that maps output indices to the input indices of which they are
    a view.

    Examples
    ========

    .. code-block:: python

        view_map = {0: [1]} # first output is a view of second input
        view_map = {1: [0]} # second output is a view of first input

    """

    destroy_map: dict[int, list[int]] = {}
    """
    A ``dict`` that maps output indices to the input indices upon which they
    operate in-place.

    Examples
    ========

    .. code-block:: python

        destroy_map = {0: [1]} # first output operates in-place on second input
        destroy_map = {1: [0]} # second output operates in-place on first input

    """

    itypes: Sequence["Type"] | None = None
    otypes: Sequence["Type"] | None = None

    _output_type_depends_on_input_value = False
    """
    Whether the static output type depends on the inferred value of one of the inputs.
    (e.g, via constant folding or static shape inference).

    This information is needed when rebuilding a graph with new inputs,
    as nodes with these Ops must be rebuilt even if the input types haven't changed.
    """

    def make_node(self, *inputs: Variable) -> Apply[Self, OpOutputType]:
        """Construct an `Apply` node that represent the application of this operation to the given inputs.

        This must be implemented by sub-classes.

        Returns
        -------
        node: Apply
            The constructed `Apply` node.

        """
        if self.itypes is None:
            raise NotImplementedError(
                "You can either define itypes and otypes, or implement make_node"
            )

        if self.otypes is None:
            raise NotImplementedError(
                "You can either define itypes and otypes, or implement make_node"
            )

        if len(inputs) != len(self.itypes):
            raise ValueError(
                f"We expected {len(self.itypes)} inputs but got {len(inputs)}."
            )
        if not all(
            expected_type.is_super(var.type)
            for var, expected_type in zip(inputs, self.itypes, strict=True)
        ):
            raise TypeError(
                f"Invalid input types for Op {self}:\n"
                + "\n".join(
                    f"Input {i}/{len(inputs)}: Expected {inp}, got {out}"
                    for i, (inp, out) in enumerate(
                        zip(self.itypes, (inp.type for inp in inputs), strict=True),
                        start=1,
                    )
                    if inp != out
                )
            )
        return Apply(self, inputs, [cast(OpOutputType, o()) for o in self.otypes])

    def __call__(
        self, *inputs: Any, name=None, return_list=False, **kwargs
    ) -> OpOutputType | list[OpOutputType]:
        r"""Construct an `Apply` node using :meth:`Op.make_node` and return its outputs.

        This method is just a wrapper around :meth:`Op.make_node`.

        It is called by code such as:

        .. code-block:: python

           x = pytensor.tensor.matrix()
           y = pytensor.tensor.exp(x)


        `pytensor.tensor.exp` is an `Op` instance, so ``pytensor.tensor.exp(x)`` calls
        :meth:`pytensor.tensor.exp.__call__` (i.e. this method) and returns its single output
        `Variable`, ``y``.  The `Apply` node constructed by :meth:`self.make_node`
        behind the scenes is available via ``y.owner``.

        `Op` authors are able to determine which output is returned by this method
        via the :attr:`Op.default_output` property.

        Parameters
        ----------
        inputs : tuple of Variable
            The `Op`'s inputs.
        kwargs
            Additional keyword arguments to be forwarded to
            :meth:`Op.make_node` *except* for optional argument ``return_list`` (which
            defaults to ``False``). If ``return_list`` is ``True``, then the returned
            value is always a ``list``. Otherwise it is either a single `Variable`
            when the output of :meth:`Op.make_node` contains a single element, or this
            output (unchanged) when it contains multiple elements.

        Returns
        -------
        outputs : list of Variable or Variable
            Either a list of output `Variable`\s, or a single `Variable`.
            This is determined by the number of outputs produced by the
            `Op`, the value of the keyword ``return_list``, and the value of
            the :attr:`Op.default_output` property.

        """
        node = self.make_node(*inputs, **kwargs)
        if name is not None:
            if len(node.outputs) == 1:
                node.outputs[0].name = name
            elif self.default_output is not None:
                node.outputs[self.default_output].name = name
            else:
                for i, n in enumerate(node.outputs):
                    n.name = f"{name}_{i}"

        if self.default_output is not None:
            rval = node.outputs[self.default_output]
            if return_list:
                return [rval]
            return rval
        else:
            if return_list:
                return list(node.outputs)
            elif len(node.outputs) == 1:
                return node.outputs[0]
            else:
                return node.outputs

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    # Convenience so that subclass implementers don't have to import utils
    # just to self.add_tag_trace
    add_tag_trace = staticmethod(add_tag_trace)

    def grad(
        self, inputs: Sequence[Variable], output_grads: Sequence[OpOutputType]
    ) -> list[OpOutputType]:
        r"""Construct a graph for the gradient with respect to each input variable.

        Each returned `Variable` represents the gradient with respect to that
        input computed based on the symbolic gradients with respect to each
        output. If the output is not differentiable with respect to an input,
        then this method should return an instance of type `NullType` for that
        input.

        Using the reverse-mode AD characterization given in [1]_, for a
        :math:`C = f(A, B)` representing the function implemented by the `Op`
        and its two arguments :math:`A` and :math:`B`, given by the
        `Variable`\s in `inputs`, the values returned by `Op.grad` represent
        the quantities :math:`\bar{A} \equiv \frac{\partial S_O}{A}` and
        :math:`\bar{B}`, for some scalar output term :math:`S_O` of :math:`C`
        in

        .. math::

            \operatorname{Tr}\left(\bar{C}^\top dC\right) =
                \operatorname{Tr}\left(\bar{A}^\top dA\right) +
                \operatorname{Tr}\left(\bar{B}^\top dB\right)


        Parameters
        ----------
        inputs
            The input variables.
        output_grads
            The gradients of the output variables.

        Returns
        -------
        grads
            The gradients with respect to each `Variable` in `inputs`.

        References
        ----------
        .. [1] Giles, Mike. 2008. “An Extended Collection of Matrix Derivative Results for Forward and Reverse Mode Automatic Differentiation.”

        """
        raise NotImplementedError(f"grad not implemented for Op {self}")

    def L_op(
        self,
        inputs: Sequence[Variable],
        outputs: Sequence[OpOutputType],
        output_grads: Sequence[OpOutputType],
    ) -> list[OpOutputType]:
        r"""Construct a graph for the L-operator.

        The L-operator computes a row vector times the Jacobian.

        This method dispatches to :meth:`Op.grad` by default.  In one sense,
        this method provides the original outputs when they're needed to
        compute the return value, whereas `Op.grad` doesn't.

        See `Op.grad` for a mathematical explanation of the inputs and outputs
        of this method.

        Parameters
        ----------
        inputs
            The inputs of the `Apply` node using this `Op`.
        outputs
            The outputs of the `Apply` node using this `Op`
        output_grads
            The gradients with respect to each `Variable` in `inputs`.

        """
        return self.grad(inputs, output_grads)

    def R_op(
        self, inputs: list[Variable], eval_points: OpOutputType | list[OpOutputType]
    ) -> list[OpOutputType]:
        r"""Construct a graph for the R-operator.

        This method is primarily used by `Rop`.

        Parameters
        ----------
        inputs
            The `Op` inputs.
        eval_points
            A `Variable` or list of `Variable`\s with the same length as inputs.
            Each element of `eval_points` specifies the value of the corresponding
            input at the point where the R-operator is to be evaluated.

        Returns
        -------
        ``rval[i]`` should be ``Rop(f=f_i(inputs), wrt=inputs, eval_points=eval_points)``.

        """
        raise NotImplementedError()

    @abstractmethod
    def perform(
        self,
        node: Apply,
        inputs: Sequence[Any],
        output_storage: OutputStorageType,
    ) -> None:
        """Calculate the function on the inputs and put the variables in the output storage.

        Parameters
        ----------
        node
            The symbolic `Apply` node that represents this computation.
        inputs
            Immutable sequence of non-symbolic/numeric inputs.  These
            are the values of each `Variable` in :attr:`node.inputs`.
        output_storage
            List of mutable single-element lists (do not change the length of
            these lists).  Each sub-list corresponds to value of each
            `Variable` in :attr:`node.outputs`.  The primary purpose of this method
            is to set the values of these sub-lists.

        Notes
        -----
        The `output_storage` list might contain data. If an element of
        output_storage is not ``None``, it has to be of the right type, for
        instance, for a `TensorVariable`, it has to be a NumPy ``ndarray``
        with the right number of dimensions and the correct dtype.
        Its shape and stride pattern can be arbitrary. It is not
        guaranteed that such pre-set values were produced by a previous call to
        this :meth:`Op.perform`; they could've been allocated by another
        `Op`'s `perform` method.
        An `Op` is free to reuse `output_storage` as it sees fit, or to
        discard it and allocate new memory.

        """

    def do_constant_folding(self, fgraph: "FunctionGraph", node: Apply) -> bool:
        """Determine whether or not constant folding should be performed for the given node.

        This allows each `Op` to determine if it wants to be constant
        folded when all its inputs are constant. This allows it to choose where
        it puts its memory/speed trade-off. Also, it could make things faster
        as constants can't be used for in-place operations (see
        ``*IncSubtensor``).

        Parameters
        ----------
        node : Apply
            The node for which the constant folding determination is made.

        Returns
        -------
        res : bool

        """
        return True

    def prepare_node(
        self,
        node: Apply,
        storage_map: StorageMapType | None,
        compute_map: ComputeMapType | None,
        impl: str | None,
    ) -> None:
        """Make any special modifications that the `Op` needs before doing :meth:`Op.make_thunk`.

        This can modify the node inplace and should return nothing.

        It can be called multiple time with different `impl` values.

        .. warning::

            It is the `Op`'s responsibility to not re-prepare the node when it
            isn't good to do so.

        """

    def make_py_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType | None,
        no_recycling: list[Variable],
        debug: bool = False,
    ) -> ThunkType:
        """Make a Python thunk.

        Like :meth:`Op.make_thunk` but only makes Python thunks.

        """
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        if debug and hasattr(self, "debug_perform"):
            p = node.op.debug_perform
        else:
            p = node.op.perform

        if compute_map is None:

            @is_thunk_type
            def rval(
                p=p,
                i=node_input_storage,
                o=node_output_storage,
                n=node,
            ):
                return p(n, [x[0] for x in i], o)

        else:
            node_compute_map = [compute_map[r] for r in node.outputs]

            @is_thunk_type
            def rval(
                p=p,
                i=node_input_storage,
                o=node_output_storage,
                n=node,
                cm=node_compute_map,
            ):
                r = p(n, [x[0] for x in i], o)
                for entry in cm:
                    entry[0] = True
                return r

        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        setattr(rval, "perform", p)
        rval.lazy = False
        return rval

    def make_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        no_recycling: list[Variable],
        impl: str | None = None,
    ) -> ThunkType:
        r"""Create a thunk.

        This function must return a thunk, that is a zero-arguments
        function that encapsulates the computation to be performed
        by this op on the arguments of the node.

        Parameters
        ----------
        node
            Something previously returned by :meth:`Op.make_node`.
        storage_map
            A ``dict`` mapping `Variable`\s to single-element lists where a
            computed value for each `Variable` may be found.
        compute_map
            A ``dict`` mapping `Variable`\s to single-element lists where a
            boolean value can be found. The boolean indicates whether the
            `Variable`'s `storage_map` container contains a valid value
            (i.e. ``True``) or whether it has not been computed yet
            (i.e. ``False``).
        no_recycling
            List of `Variable`\s for which it is forbidden to reuse memory
            allocated by a previous call.
        impl : str
            Description for the type of node created (e.g. ``"c"``, ``"py"``,
            etc.)

        Notes
        -----
        If the thunk consults the `storage_map` on every call, it is safe
        for it to ignore the `no_recycling` argument, because elements of the
        `no_recycling` list will have a value of ``None`` in the `storage_map`.
        If the thunk can potentially cache return values (like `CLinker` does),
        then it must not do so for variables in the `no_recycling` list.

        :meth:`Op.prepare_node` is always called. If it tries ``'c'`` and it
        fails, then it tries ``'py'``, and :meth:`Op.prepare_node` will be
        called twice.
        """
        self.prepare_node(
            node, storage_map=storage_map, compute_map=compute_map, impl="py"
        )
        return self.make_py_thunk(node, storage_map, compute_map, no_recycling)

    def inplace_on_inputs(self, allowed_inplace_inputs: list[int]) -> "Op":
        """Try to return a version of self that tries to inplace in as many as `allowed_inplace_inputs`."""
        # TODO: Document this in the Create your own Op docs
        # By default, do nothing
        return self

    def __str__(self):
        return getattr(type(self), "__name__", super().__str__())

    def __repr__(self):
        props = getattr(self, "__props__", ())
        props = ",".join(f"{prop}={getattr(self, prop, '?')}" for prop in props)
        return f"{self.__class__.__name__}({props})"


class _NoPythonOp(Op):
    """A class used to indicate that an `Op` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError("No Python implementation is provided by this Op.")


class HasInnerGraph(ABC):
    r"""A mixin for an `Op` that contain an inner graph."""

    fgraph: "FunctionGraph"
    """A `FunctionGraph` of the inner function."""

    @property
    @abstractmethod
    def fn(self) -> "Function":
        """The compiled inner-graph function."""

    @property
    @abstractmethod
    def inner_inputs(self) -> list[Variable]:
        """The inner function's inputs."""

    @property
    @abstractmethod
    def inner_outputs(self) -> list[Variable]:
        """The inner function's outputs."""

    @abstractmethod
    def clone(self) -> Op:
        """Clone the `Op` and its inner-graph."""


def io_connection_pattern(inputs, outputs):
    """Return the connection pattern of a subgraph defined by given inputs and outputs."""
    inner_nodes = io_toposort(inputs, outputs)

    # Initialize 'connect_pattern_by_var' by establishing each input as
    # connected only to itself
    connect_pattern_by_var = {}
    nb_inputs = len(inputs)

    for i in range(nb_inputs):
        input = inputs[i]
        inp_connection_pattern = [i == j for j in range(nb_inputs)]
        connect_pattern_by_var[input] = inp_connection_pattern

    # Iterate through the nodes used to produce the outputs from the
    # inputs and, for every node, infer their connection pattern to
    # every input from the connection patterns of their parents.
    for n in inner_nodes:
        # Get the connection pattern of the inner node's op. If the op
        # does not define a connection_pattern method, assume that
        # every node output is connected to every node input
        try:
            op_connection_pattern = n.op.connection_pattern(n)
        except AttributeError:
            op_connection_pattern = [[True] * len(n.outputs)] * len(n.inputs)

        # For every output of the inner node, figure out which inputs it
        # is connected to by combining the connection pattern of the inner
        # node and the connection patterns of the inner node's inputs.
        for out_idx in range(len(n.outputs)):
            out = n.outputs[out_idx]
            out_connection_pattern = [False] * nb_inputs

            for inp_idx in range(len(n.inputs)):
                inp = n.inputs[inp_idx]

                if inp in connect_pattern_by_var:
                    inp_connection_pattern = connect_pattern_by_var[inp]

                    # If the node output is connected to the node input, it
                    # means it is connected to every inner input that the
                    # node inputs is connected to
                    if op_connection_pattern[inp_idx][out_idx]:
                        out_connection_pattern = [
                            out_connection_pattern[i] or inp_connection_pattern[i]
                            for i in range(nb_inputs)
                        ]

            # Store the connection pattern of the node output
            connect_pattern_by_var[out] = out_connection_pattern

    # Obtain the global connection pattern by combining the
    # connection patterns of the individual outputs
    global_connection_pattern = [[] for o in range(len(inputs))]
    for out in outputs:
        out_connection_pattern = connect_pattern_by_var.get(out)
        if out_connection_pattern is None:
            # the output is completely isolated from inputs
            out_connection_pattern = [False] * len(inputs)
        for i in range(len(inputs)):
            global_connection_pattern[i].append(out_connection_pattern[i])

    return global_connection_pattern
