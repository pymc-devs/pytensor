from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.type import Type
from pytensor.link.utils import gc_helper, map_storage, raise_with_op, streamline
from pytensor.utils import difference


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pytensor.compile.profiling import ProfileStats
    from pytensor.graph.op import (
        BasicThunkType,
        InputStorageType,
        OutputStorageType,
        StorageMapType,
    )
    from pytensor.tensor.variable import TensorVariable


ThunkAndContainersType = tuple["BasicThunkType", list["Container"], list["Container"]]


class Container:
    """
    This class joins a variable with its computed value.

    It is used in linkers, especially for the inputs and outputs of a Function.

    Parameters
    ----------
    r
        The `Variable` or `Type` to associate with the `Container`.
    storage
        A list of length 1, whose element is the value for `r`.
    readonly : bool
        True indicates that this should not be setable by Function[r] = val.
    strict : bool
        If True, we don't allow type casting.
    allow_downcast : bool
        If True (and `strict` is False), allow upcasting of type, but not
        downcasting. If False, prevent it. If None (default), allows only
        downcasting of float to floatX scalar.
    name : str
        A string (for pretty-printing?)

    """

    def __init__(
        self,
        r: Variable | Type,
        storage: list[Any],
        *,
        readonly: bool = False,
        strict: bool = False,
        allow_downcast: bool | None = None,
        name: str | None = None,
    ) -> None:
        if not (isinstance(storage, list) and len(storage) >= 1):
            raise TypeError("storage must be a list of length at least one")
        if isinstance(r, Variable):
            self.type = r.type
        else:
            self.type = r

        if name is None:
            # Some Type do not have a name field.
            self.name = getattr(r, "name", None)
        else:
            self.name = name

        self.storage = storage
        self.readonly = readonly
        self.strict = strict
        self.allow_downcast = allow_downcast

    def __get__(self) -> Any:
        return self.storage[0]

    def __set__(self, value: Any) -> None:
        if self.readonly:
            raise Exception(f"Cannot set readonly storage: {self.name}")
        try:
            if value is None:
                self.storage[0] = None
                return

            kwargs = {}
            if self.strict:
                kwargs["strict"] = True
            if self.allow_downcast is not None:
                kwargs["allow_downcast"] = self.allow_downcast

            try:
                # Use in-place filtering when/if possible
                self.storage[0] = self.type.filter_inplace(
                    value, self.storage[0], **kwargs
                )
            except NotImplementedError:
                self.storage[0] = self.type.filter(value, **kwargs)

        except Exception as e:
            e.args = (*e.args, f'Container name "{self.name}"')
            raise

    data = property(__get__, __set__)
    value = property(__get__, __set__)

    def __str__(self):
        return "<" + str(self.storage[0]) + ">"

    def __repr__(self):
        return "<" + repr(self.storage[0]) + ">"

    def __deepcopy__(self, memo: dict[int, Any]) -> "Container":
        data_was_in_memo = id(self.storage[0]) in memo
        r = type(self)(
            deepcopy(self.type, memo=memo),
            deepcopy(self.storage, memo=memo),
            readonly=deepcopy(self.readonly, memo=memo),
            strict=deepcopy(self.strict, memo=memo),
            allow_downcast=deepcopy(self.allow_downcast, memo=memo),
            name=deepcopy(self.name, memo=memo),
        )
        # Work around NumPy deepcopy of ndarray with 0 dimension that
        # don't return an ndarray.
        if r.storage[0] is not None and not self.type.is_valid_value(r.storage[0]):
            assert not data_was_in_memo
            assert self.type.is_valid_value(self.storage[0])
            # This should also work for read only container.
            r.storage[0] = self.type.filter(
                r.storage[0], strict=False, allow_downcast=False
            )
            memo[id(self.storage[0])] = r.storage[0]
        return r


class Linker(ABC):
    """
    Base type for all linkers.

    A linker takes a FunctionGraph and turns it into a callable.

    Parameters
    ----------
    allow_gc : optional, bool
        Configures if garbage collection is enabled.
    scheduler : callable
        A scheduling function that takes a FunctionGraph and returns
        a list of Apply nodes. Defaults to the .toposort() method of
        the FunctionGraph.
    """

    def __init__(
        self,
        *,
        allow_gc: bool | None = None,
        scheduler: Callable[[FunctionGraph], list[Apply]] | None = None,
    ) -> None:
        self._allow_gc = allow_gc
        self._scheduler = scheduler
        super().__init__()

    @property
    def allow_gc(self) -> bool | None:
        """Determines if the linker may allow garbage collection.

        Returns
        -------
        _allow_gc : optional, bool
            None means undefined.
        """
        return self._allow_gc

    def clone(self, allow_gc: bool | None = None) -> "Linker":
        new = copy(self)
        if allow_gc is not None:
            new._allow_gc = allow_gc
        return new

    @abstractmethod
    def make_thunk(
        self, **kwargs
    ) -> tuple[Callable, "InputStorageType", "OutputStorageType"]:
        """
        This function must return a triplet (function, input_variables,
        output_variables) where function is a thunk that operates on the
        returned variables. If inplace is True, the input_variables and
        output_variables lists will be the same as the inputs and outputs
        of the graph provided to the L{Linker}. Else, independent
        variables will be returned.

        Examples
        --------
        x, y = Variable(Double, None), Variable(Double, None)
        e = x + y
        fgraph = FunctionGraph([x, y], [e])
        fn, (new_x, new_y), (new_e, ) = MyLinker(fgraph).make_thunk(inplace)
        new_x.data = 1.0
        new_y.data = 2.0
        fn()
        print new_e.data # 3.0
        print e.data # 3.0 iff inplace == True (else unknown)

        """

    def schedule(self, fgraph: FunctionGraph) -> list[Apply]:
        """Runs the scheduler (if set) or the toposort on the FunctionGraph.

        Parameters
        ----------
        fgraph : :py:class:`aerasa.graph.fg.FunctionGraph`
            A graph to compute the schedule for.

        Returns
        -------
        nodes : list of :py:class:`pytensor.graph.basic.Apply` nodes
            The result of the scheduling or toposort operation.
        """
        if callable(self._scheduler):
            return self._scheduler(fgraph)
        return fgraph.toposort()


class LocalLinker(Linker):
    """
    Useful base class for L{Linker}s which keep all nodes in the graph, and run
    a thunk associated with each node.

    """

    def make_thunk(
        self,
        input_storage: Optional["InputStorageType"] = None,
        output_storage: Optional["OutputStorageType"] = None,
        storage_map: Optional["StorageMapType"] = None,
        **kwargs,
    ) -> tuple["BasicThunkType", "InputStorageType", "OutputStorageType"]:
        return self.make_all(
            input_storage=input_storage,
            output_storage=output_storage,
            storage_map=storage_map,
        )[:3]

    def make_all(
        self,
        input_storage: Optional["InputStorageType"] = None,
        output_storage: Optional["OutputStorageType"] = None,
        storage_map: Optional["StorageMapType"] = None,
    ) -> tuple[
        "BasicThunkType",
        "InputStorageType",
        "OutputStorageType",
        list[ThunkAndContainersType],
        list[Apply],
    ]:
        """
        This function should return a tuple of 5 things
            1. function to run the program
            2. input storage
            3. output storage
            4. thunks: list of nodes' functions in the order they will be run by the function in (1)
            5. order: list of nodes, in the order they will be run by the function in (1)
        """
        raise NotImplementedError(
            f"make_all method of {type(self)} is not implemented."
        )


class PerformLinker(LocalLinker):
    """
    Basic L{Linker} subclass that calls the perform method on each L{Op} in
    the L{FunctionGraph} in the order given by L{Linker.schedule}.

    """

    def __init__(
        self, allow_gc: bool | None = None, schedule: Callable | None = None
    ) -> None:
        if allow_gc is None:
            allow_gc = config.allow_gc
        self.fgraph: FunctionGraph | None = None
        super().__init__(allow_gc=allow_gc, scheduler=schedule)

    def accept(
        self,
        fgraph: FunctionGraph,
        no_recycling: Sequence[Variable] | None = None,
        profile: Union[bool, "ProfileStats"] | None = None,
    ) -> "PerformLinker":
        """Associate a `FunctionGraph` with this `Linker`.

        Parameters
        ----------
        fgraph
            A `PerformLinker` instance can have accepted one `FunctionGraph`
            instance at a time.
        no_recycling
            WRITEME

        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            return type(self)(allow_gc=self.allow_gc).accept(
                fgraph, no_recycling, profile
            )
            # raise Exception("Cannot accept from a Linker that is already tied to another FunctionGraph.")
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        return self

    def make_all(
        self,
        input_storage=None,
        output_storage=None,
        storage_map=None,
    ):
        fgraph = self.fgraph
        order = self.schedule(fgraph)
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = map_storage(
            fgraph, order, input_storage, output_storage, storage_map
        )

        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks = []
        for node in order:
            # Maker sure we don't use C version of the code, but rather only
            # the python version
            # Note : ops that implement their own make thunk don't usually
            # have this attribute defined !!
            thunks += [
                node.op.make_thunk(node, storage_map, compute_map, no_recycling, "py")
            ]
            thunks[-1].inputs = [storage_map[v] for v in node.inputs]
            thunks[-1].outputs = [storage_map[v] for v in node.outputs]

        computed, last_user = gc_helper(order)
        post_thunk_old_storage = [] if self.allow_gc else None

        for node in order:
            if self.allow_gc:
                post_thunk_old_storage.append(
                    [
                        storage_map[input]
                        for input in node.inputs
                        if (input in computed)
                        and (input not in fgraph.outputs)
                        and (node == last_user[input])
                    ]
                )

        if no_recycling is True:
            # True seems like some special code for *everything*?? -JB
            # FunctionMaker always passes a list I think   -JB
            no_recycling = list(storage_map.values())
            no_recycling = difference(no_recycling, input_storage)
        else:
            no_recycling = [
                storage_map[r] for r in no_recycling if r not in fgraph.inputs
            ]

        # The function that actually runs your program is one of the f's in streamline.
        f = streamline(
            fgraph, thunks, order, post_thunk_old_storage, no_recycling=no_recycling
        )

        f.allow_gc = (
            self.allow_gc
        )  # HACK: this is a way of passing an arg to Function.__call__
        f.storage_map = storage_map

        return (
            f,
            [
                Container(input, storage)
                for input, storage in zip(fgraph.inputs, input_storage, strict=True)
            ],
            [
                Container(output, storage, readonly=True)
                for output, storage in zip(fgraph.outputs, output_storage, strict=True)
            ],
            thunks,
            order,
        )


class WrapLinker(Linker):
    """
    This class makes it easier to run several L{LocalLinker}s in parallel, and
    offers some control over how each thunk is run.

    A wrapper function must be provided, and it can be used to execute the
    thunks, inspect the nodes, print stuff out, etc.

    The constructor initializes a WrapLinker.

    Parameters
    ----------
    linkers
        List of L{LocalLinker} subclasses, whose make_all() method returns
        thunks in the same order.
        For each node in the graph, each linker will provide a
        thunk.  This class makes it possible to iterate over each linker's
        program in parallel.
    wrapper : lambda (fgraph, i, i_node, i_thunk1, i_thunk2, ...) : None
        Does some user-defined action for the i'th element of the program.
        i_thunk<n> is the thunk returned by the n'th linker. (If you want
        to run the program, make sure to call the necessary thunks in this
        function.)

    Notes
    -----
    The outputs of the first linker will be returned.

    This linker ensures that each linker has its own storage for inputs and
    outputs and intermediate variables. There is no interference between
    linkers.

    """

    def __init__(
        self,
        linkers: Sequence[PerformLinker],
        wrapper: Callable,
    ) -> None:
        self.fgraph: FunctionGraph | None = None
        self.linkers = linkers
        self.wrapper = wrapper

    def __copy__(self) -> "WrapLinker":
        """
        Shallow copy of a WrapLinker.

        Returns
        -------
        object
            A copy of self, where each of the linkers in self.linkers
            have been shallow-copied.

        It is useful because in FunctionMaker, copy.copy is called on the
        Mode's linker, so that it is not modified inplace when linker.accept()
        is called. In this case, we want the wrapped linkers to be copied too.

        """
        other = self.__class__(
            linkers=[copy(x) for x in self.linkers], wrapper=self.wrapper
        )
        return other

    def clone(self, allow_gc=None):
        return self.__class__(
            linkers=[x.clone(allow_gc=allow_gc) for x in self.linkers],
            wrapper=self.wrapper,
        )

    def accept(
        self,
        fgraph: FunctionGraph,
        no_recycling: Sequence["TensorVariable"] | None = None,
        profile: Union[bool, "ProfileStats"] | None = None,
    ) -> "WrapLinker":
        """

        Parameters
        ----------
        fgraph : :py:class:`pytensor.graph.fg.FunctionGraph`
            The fgraph which we will link.
        no_recycling : a list of Variables that belong to fgraph.
            If a Variable is in no_recycling, L{WrapLinker} will clear
            the output storage associated to it (for each linker in linkers)
            during the computation to avoid reusing it.

        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            return type(self)(self.linkers, self.wrapper).accept(fgraph, no_recycling)

        self.fgraph = fgraph
        self.no_recycling = no_recycling
        self.linkers = [linker.accept(fgraph, no_recycling) for linker in self.linkers]
        return self

    def pre(
        self,
        f: "WrapLinker",
        inputs: list["NDArray"] | list[float | None],
        order: list[Apply],
        thunk_groups: list[tuple[Callable]],
    ) -> None:
        pass

    def make_thunk(self, **kwargs):
        no_recycling = self.no_recycling

        make_all = [self.linkers[0].make_all(**kwargs)]
        kwargs.pop("input_storage", None)
        make_all += [x.make_all(**kwargs) for x in self.linkers[1:]]

        fns, input_lists, output_lists, thunk_lists, order_lists = zip(
            *make_all, strict=True
        )

        order_list0 = order_lists[0]
        for order_list in order_lists[1:]:
            if order_list0 != order_list:
                raise Exception(
                    "All linkers to WrapLinker should execute operations in the same order."
                )

        inputs0 = input_lists[0]
        outputs0 = output_lists[0]

        thunk_groups = list(zip(*thunk_lists, strict=True))
        order = [x[0] for x in zip(*order_lists, strict=True)]

        to_reset = [
            thunk.outputs[j]
            for thunks, node in zip(thunk_groups, order, strict=True)
            for j, output in enumerate(node.outputs)
            if output in no_recycling
            for thunk in thunks
        ]

        wrapper = self.wrapper
        pre = self.pre

        def f():
            for inputs in input_lists[1:]:
                # zip strict not specified because we are in a hot loop
                for input1, input2 in zip(inputs0, inputs):
                    input2.storage[0] = copy(input1.storage[0])
            for x in to_reset:
                x[0] = None
            pre(self, [input.data for input in input_lists[0]], order, thunk_groups)
            # zip strict not specified because we are in a hot loop
            for i, (thunks, node) in enumerate(zip(thunk_groups, order)):
                try:
                    wrapper(self.fgraph, i, node, *thunks)
                except Exception:
                    raise_with_op(self.fgraph, node, *thunks)

        f.thunk_groups = thunk_groups

        return f, inputs0, outputs0


def WrapLinkerMany(
    linkers: list[PerformLinker], wrappers: list[Callable]
) -> WrapLinker:
    """
    Variant on WrapLinker that runs a series of wrapper functions instead of
    just one.

    """

    def wrapper(*args):
        for f in wrappers:
            f(*args)

    return WrapLinker(linkers, wrapper)


class JITLinker(PerformLinker):
    """A ``Linker`` that JIT compiles a ``FunctionGraph`` into a single runnable thunk.

    The entirety of ``Linker.fgraph`` is converted into a single JIT compiled
    thunk that is run by an PyTensor ``VM``.

    """

    @abstractmethod
    def fgraph_convert(
        self, fgraph, order, input_storage, output_storage, storage_map, **kwargs
    ):
        """Convert a ``FunctionGraph`` into a JIT-able function."""

    @abstractmethod
    def create_thunk_inputs(self, storage_map: dict[Variable, list[Any]]) -> list[Any]:
        """Pre-process inputs for the generated thunk.

        Parameters
        ==========
        storage_map
            A ``dict`` mapping ``Variable``s to their storage lists.

        Returns
        =======
        A list of thunk inputs
        """

    @abstractmethod
    def jit_compile(self, fn: Callable) -> Callable:
        """JIT compile a converted ``FunctionGraph``."""

    def input_filter(self, inp: Any) -> Any:
        """Apply a filter to the data input."""
        return inp

    def output_filter(self, var: Variable, out: Any) -> Any:
        """Apply a filter to the data output by a JITed function call."""
        return out

    def create_jitable_thunk(
        self, compute_map, order, input_storage, output_storage, storage_map
    ):
        r"""Create a thunk for each output of the `Linker`\s `FunctionGraph`.

        This is differs from the other thunk-making function in that it only
        produces thunks for the `FunctionGraph` output nodes.

        Parameters
        ----------
        compute_map: dict
            The compute map dictionary.
        order
        input_storage
        output_storage
        storage_map: dict
            The storage map dictionary.

        Returns
        -------
        thunks: list
            A tuple containing the thunks.
        output_nodes: list and their
            A tuple containing the output nodes.
        jit_fn: callable
            The JITed function that performs the computations.

        """
        # This is a bit hackish, but we only return one of the output nodes
        output_nodes = [o.owner for o in self.fgraph.outputs if o.owner is not None][:1]

        converted_fgraph = self.fgraph_convert(
            self.fgraph,
            order=order,
            input_storage=input_storage,
            output_storage=output_storage,
            storage_map=storage_map,
        )

        thunk_inputs = self.create_thunk_inputs(storage_map)
        thunk_outputs = [storage_map[n] for n in self.fgraph.outputs]
        fgraph_jit = self.jit_compile(converted_fgraph)

        def thunk(
            fgraph_jit=fgraph_jit,
            thunk_inputs=thunk_inputs,
            thunk_outputs=thunk_outputs,
        ):
            try:
                outputs = fgraph_jit(*(x[0] for x in thunk_inputs))
            except Exception:
                # TODO: Should we add a fake node that combines all outputs,
                #  since the error may come from any of them?
                raise_with_op(self.fgraph, output_nodes[0], thunk)

            # zip strict not specified because we are in a hot loop
            for o_storage, o_val in zip(thunk_outputs, outputs):
                o_storage[0] = o_val

        thunk.inputs = thunk_inputs
        thunk.outputs = thunk_outputs
        thunk.lazy = False

        thunks = [thunk]

        return thunks, output_nodes, fgraph_jit

    def make_all(self, input_storage=None, output_storage=None, storage_map=None):
        fgraph = self.fgraph
        nodes = self.schedule(fgraph)

        input_storage, output_storage, storage_map = map_storage(
            fgraph, nodes, input_storage, output_storage, storage_map
        )

        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks, nodes, jit_fn = self.create_jitable_thunk(
            compute_map, nodes, input_storage, output_storage, storage_map
        )

        [fn] = thunks
        fn.jit_fn = jit_fn
        fn.allow_gc = self.allow_gc
        fn.storage_map = storage_map

        return (
            fn,
            [
                Container(input, storage)
                for input, storage in zip(fgraph.inputs, input_storage, strict=True)
            ],
            [
                Container(output, storage, readonly=True)
                for output, storage in zip(fgraph.outputs, output_storage, strict=True)
            ],
            thunks,
            nodes,
        )
