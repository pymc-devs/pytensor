"""A container for specifying and manipulating a graph with distinct inputs and outputs."""

import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Union, cast

import pytensor
from pytensor.configdefaults import config
from pytensor.graph.basic import (
    Apply,
    AtomicVariable,
    Variable,
    clone_get_equiv,
)
from pytensor.graph.basic import as_string as graph_as_string
from pytensor.graph.features import AlreadyThere, Feature, ReplaceValidate
from pytensor.graph.op import Op
from pytensor.graph.traversal import (
    applys_between,
    graph_inputs,
    toposort,
    toposort_with_orderings,
    vars_between,
)
from pytensor.graph.utils import MetaObject, MissingInputError, TestValueError


ClientType = tuple[Apply, int]


class Output(Op):
    """A dummy `Op` that represents an output variable in a `FunctionGraph`."""

    __props__ = ("idx",)

    def __init__(self, idx):
        self.idx = idx

    def make_node(self, inp):
        return Apply(self, [inp], [])

    def perform(self, node, inputs, outputs):
        raise RuntimeError("Output Ops should never be evaluated")

    def __str__(self):
        return f"output[{self.idx}]"


class FunctionGraph(MetaObject):
    r"""
    A `FunctionGraph` represents a subgraph bound by a set of input variables and
    a set of output variables, ie a subgraph that specifies an PyTensor function.
    The inputs list should contain all the inputs on which the outputs depend.
    `Variable`\s of type `Constant` are not counted as inputs.

    The `FunctionGraph` supports the replace operation which allows to replace
    a variable in the subgraph by another, e.g. replace ``(x + x).out`` by
    ``(2 * x).out``. This is the basis for optimization in PyTensor.

    This class is also responsible for verifying that a graph is valid
    (ie, all the dtypes and broadcast patterns are compatible with the
    way the `Variable`\s are used) and for tracking the `Variable`\s with
    a :attr:`FunctionGraph.clients` ``dict`` that specifies which `Apply` nodes
    use the `Variable`.  The :attr:`FunctionGraph.clients` field, combined with
    the :attr:`Variable.owner` and each :attr:`Apply.inputs`, allows the graph
    to be traversed in both directions.

    It can also be extended with new features using
    :meth:`FunctionGraph.attach_feature`.  See `Feature` for event types and
    documentation.  Extra features allow the `FunctionGraph` to verify new
    properties of a graph as it is optimized.

    The constructor creates a `FunctionGraph` which operates on the subgraph
    bound by the inputs and outputs sets.

    This class keeps lists for the inputs and outputs and modifies them
    in-place.

    """

    def __init__(
        self,
        inputs: Sequence[Variable] | None = None,
        outputs: Sequence[Variable] | None = None,
        features: Sequence[Feature] | None = None,
        clone: bool = True,
        update_mapping: dict[Variable, Variable] | None = None,
        **clone_kwds,
    ):
        """
        Create a `FunctionGraph` which operates on the subgraph between the
        `inputs` and `outputs`.

        Parameters
        ----------
        inputs
            Input variables of the graph.
        outputs
            Output variables of the graph.
        features
            A list of features to be added to the `FunctionGraph`.
        clone
            If ``True``, the graph will be cloned.
        update_mapping
            Mapping between the `inputs` with updates and the `outputs`
            corresponding to their updates.
        clone_kwds
            Keywords passed to `clone_get_equiv` when `clone` is ``True``.
        """
        if outputs is None:
            raise ValueError("No outputs specified")

        if inputs is None:
            inputs = [
                i for i in graph_inputs(outputs) if not isinstance(i, AtomicVariable)
            ]

        if clone:
            _memo = clone_get_equiv(
                inputs,
                outputs,
                **clone_kwds,
            )
            outputs = [cast(Variable, _memo[o]) for o in outputs]
            inputs = [cast(Variable, _memo[i]) for i in inputs]

        self.execute_callbacks_time: float = 0.0
        self.execute_callbacks_times: dict[Feature, float] = defaultdict(float)

        if features is None:
            features = []

        self._features: list[Feature] = []
        # All apply nodes in the subgraph defined by inputs and
        # outputs are cached in this field
        self.apply_nodes: set[Apply] = set()

        # It includes inputs, outputs, and all intermediate variables
        # connecting the inputs and outputs.  It also contains irrelevant
        # outputs the nodes in `self.apply_nodes`.
        self.variables: set[Variable] = set()

        self.inputs: list[Variable] = []
        self.outputs: list[Variable] = []
        self.clients: dict[Variable, list[ClientType]] = {}

        for f in features:
            self.attach_feature(f)

        self.attach_feature(ReplaceValidate())

        for in_var in inputs:
            if in_var.owner is not None:
                raise ValueError(
                    "One of the provided inputs is the output of "
                    "an already existing node. "
                    "If that is okay, either discard that "
                    "input's owner or use graph.clone."
                )

            self.inputs.append(in_var)
            self.clients.setdefault(in_var, [])

        for output in outputs:
            self.add_output(output, reason="init")

        self.profile = None
        self.update_mapping = update_mapping

    def add_output(
        self, var: Variable, reason: str | None = None, import_missing: bool = False
    ):
        """Add a new variable as an output to this `FunctionGraph`."""
        self.outputs.append(var)
        self.import_var(var, reason=reason, import_missing=import_missing)
        self.clients[var].append((Output(len(self.outputs) - 1).make_node(var), 0))

    def add_input(self, var: Variable, check: bool = True) -> None:
        """Add a new variable as an input to this `FunctionGraph`.

        Parameters
        ----------
        var : pytensor.graph.basic.Variable

        """
        if check and var in self.inputs:
            return

        self.inputs.append(var)
        self.clients.setdefault(var, [])

    def get_clients(self, var: Variable) -> list[ClientType]:
        """Return a list of all the `(node, i)` pairs such that `node.inputs[i]` is `var`."""
        return self.clients[var]

    def add_client(self, var: Variable, new_client: ClientType) -> None:
        """Update the clients of `var` with `new_clients`.

        Parameters
        ----------
        var : Variable
            The `Variable` to be updated.
        new_client : (Apply, int)
            A ``(node, i)`` pair such that ``node.inputs[i]`` is `var`.

        """
        if not isinstance(new_client[0], Apply):
            raise TypeError("The first entry of `new_client` must be an `Apply` node")
        self.clients[var].append(new_client)

    def remove_client(
        self,
        var: Variable,
        client_to_remove: ClientType,
        reason: str | None = None,
        remove_if_empty: bool = False,
    ) -> None:
        """Recursively remove clients of a variable.

        This is the main method to remove variables or `Apply` nodes from
        a `FunctionGraph`.

        This will remove `var` from the `FunctionGraph` if it doesn't have any
        clients remaining. If it has an owner and all the outputs of the owner
        have no clients, it will also be removed.

        Parameters
        ----------
        var
            The clients of `var` that will be removed.
        client_to_remove
            A ``(node, i)`` pair such that ``node.inputs[i]`` will no longer be
            `var` in this `FunctionGraph`.
        remove_if_empty
            When ``True``, if `var`'s `Apply` node is removed, remove the
            entry for `var` in `self.clients`.

        """
        clients = self.clients
        removal_stack = [(var, client_to_remove)]
        while removal_stack:
            var, client_to_remove = removal_stack.pop()

            try:
                var_clients = clients[var]
                var_clients.remove(client_to_remove)
            except ValueError:
                # In this case, the original `var` could've been removed from
                # the current `var`'s client list before this call.
                # There's nothing inherently wrong with that, so we continue as
                # if it were removed here.
                var_clients = None

            if var_clients:
                continue

            # Now, `var` has no more clients, so check if we need to remove it
            # and its `Apply` node
            if var.owner is None:
                self.variables.remove(var)
            else:
                apply_node = var.owner
                if not any(clients[output] for output in apply_node.outputs):
                    # The `Apply` node is not used and is not an output, so we
                    # remove it and its outputs
                    if not hasattr(apply_node.tag, "removed_by"):
                        apply_node.tag.removed_by = []

                    apply_node.tag.removed_by.append(str(reason))

                    self.apply_nodes.remove(apply_node)

                    self.variables.difference_update(apply_node.outputs)

                    self.execute_callbacks("on_prune", apply_node, reason)

                    removal_stack.extend(
                        (in_var, (apply_node, i))
                        for i, in_var in enumerate(apply_node.inputs)
                    )

                    if remove_if_empty:
                        del clients[var]

    def get_output_client(self, i: int) -> ClientType:
        """Get the dummy Output Op client to output i.

        Raises lookup error if not found
        """
        for client in self.clients[self.outputs[i]]:
            if isinstance(client[0].op, Output) and client[0].op.idx == i:
                return client
        raise LookupError

    def import_var(
        self, var: Variable, reason: str | None = None, import_missing: bool = False
    ) -> None:
        """Import a `Variable` into this `FunctionGraph`.

        This will import the `var`'s `Apply` node and inputs.

        Parameters
        ----------
        variable : pytensor.graph.basic.Variable
            The variable to be imported.
        reason : str
            The name of the optimization or operation in progress.
        import_missing : bool
            Add missing inputs instead of raising an exception.

        """
        # Imports the owners of the variables
        apply = var.owner
        if apply is not None and apply not in self.apply_nodes:
            self.import_node(apply, reason=reason, import_missing=import_missing)
        elif (
            apply is None
            and not isinstance(var, AtomicVariable)
            and var not in self.inputs
        ):
            from pytensor.graph.null_type import NullType

            if isinstance(var.type, NullType):
                raise TypeError(
                    f"Computation graph contains a NaN. {var.type.why_null}"
                )
            if import_missing:
                self.inputs.append(var)
                self.clients.setdefault(var, [])
            else:
                raise MissingInputError(f"Undeclared input: {var}", variable=var)
        self.clients.setdefault(var, [])
        self.variables.add(var)

    def import_node(
        self,
        apply_node: Apply,
        check: bool = True,
        reason: str | None = None,
        import_missing: bool = False,
    ) -> None:
        """Recursively import everything between an ``Apply`` node and the ``FunctionGraph``'s outputs.

        Parameters
        ----------
        apply_node : Apply
            The node to be imported.
        check : bool
            Check that the inputs for the imported nodes are also present in the `FunctionGraph`.
        reason : str
            The name of the optimization or operation in progress.
        import_missing : bool
            Add missing inputs instead of raising an exception.
        """
        # We import the nodes in topological order. We only are interested in
        # new nodes, so we use all nodes we know of as inputs to interrupt the toposort
        self_variables = self.variables
        self_clients = self.clients
        self_apply_nodes = self.apply_nodes
        self_inputs = self.inputs
        for node in toposort(apply_node.outputs, blockers=self_variables):
            if check:
                for var in node.inputs:
                    if (
                        var.owner is None
                        and not isinstance(var, AtomicVariable)
                        and var not in self_inputs
                    ):
                        if import_missing:
                            self_inputs.append(var)
                            self_clients.setdefault(var, [])
                        else:
                            error_msg = (
                                f"Input {node.inputs.index(var)} ({var})"
                                " of the graph (indices start "
                                f"from 0), used to compute {node}, was not "
                                "provided and not given a value. Use the "
                                "PyTensor flag exception_verbosity='high', "
                                "for more information on this error."
                            )
                            raise MissingInputError(error_msg, variable=var)

            self_apply_nodes.add(node)
            tag = node.tag
            if not hasattr(tag, "imported_by"):
                tag.imported_by = [str(reason)]
            else:
                tag.imported_by.append(str(reason))
            for output in node.outputs:
                self_clients.setdefault(output, [])
                self_variables.add(output)
            for i, inp in enumerate(node.inputs):
                if inp not in self_variables:
                    self_clients.setdefault(inp, [])
                    self_variables.add(inp)
                self_clients[inp].append((node, i))
            self.execute_callbacks("on_import", node, reason)

    def change_node_input(
        self,
        node: Apply,
        i: int,
        new_var: Variable,
        reason: str | None = None,
        import_missing: bool = False,
        check: bool = True,
    ) -> None:
        """Change ``node.inputs[i]`` to `new_var`.

        ``new_var.type.is_super(old_var.type)`` must be ``True``, where
        ``old_var`` is the current value of ``node.inputs[i]`` which we want to
        replace.

        For each feature that has an `on_change_input` method, this method calls:
        ``feature.on_change_input(function_graph, node, i, old_var, new_var, reason)``

        Parameters
        ----------
        node
            The node for which an input is to be changed.
        i
            The index in `node.inputs` that we want to change.
        new_var
            The new variable to take the place of ``node.inputs[i]``.
        import_missing
            Add missing inputs instead of raising an exception.
        check
            When ``True``, perform a type check between the variable being
            replaced and its replacement.  This is primarily used by the
            `History` `Feature`, which needs to revert types that have been
            narrowed and would otherwise fail this check.
        """
        # TODO: ERROR HANDLING FOR LISTENERS (should it complete the change or revert it?)
        r = node.inputs[i]

        if r is new_var:
            return

        if check and not r.type.is_super(new_var.type):
            raise TypeError(
                f"The type of the replacement ({new_var.type}) must be "
                f"compatible with the type of the original Variable ({r.type})."
            )
        node.inputs[i] = new_var

        if isinstance(node.op, Output):
            self.outputs[node.op.idx] = new_var

        self.import_var(new_var, reason=reason, import_missing=import_missing)
        self.clients[new_var].append((node, i))
        self.remove_client(r, (node, i), reason=reason)
        # Precondition: the substitution is semantically valid However it may
        # introduce cycles to the graph, in which case the transaction will be
        # reverted later.
        self.execute_callbacks("on_change_input", node, i, r, new_var, reason=reason)

    def replace(
        self,
        var: Variable,
        new_var: Variable,
        reason: str | None = None,
        verbose: bool | None = None,
        import_missing: bool = False,
    ) -> None:
        """Replace a variable in the `FunctionGraph`.

        This is the main interface to manipulate the subgraph in `FunctionGraph`.
        For every node that uses `var` as input, makes it use `new_var` instead.

        Parameters
        ----------
        var
            The variable to be replaced.
        new_var
            The variable to replace `var`.
        reason
            The name of the optimization or operation in progress.
        verbose
            Print `reason`, `var`, and `new_var`.
        import_missing
            Import missing variables.

        """
        if verbose is None:
            verbose = config.optimizer_verbose

        if verbose:
            print_reason = True
            if config.optimizer_verbose_ignore:
                print_reason = str(reason) not in config.optimizer_verbose_ignore.split(
                    ","
                )

            if print_reason:
                print(  # noqa: T201
                    f"rewriting: rewrite {reason} replaces {var} of {var.owner} with {new_var} of {new_var.owner}"
                )

        new_var = var.type.filter_variable(new_var, allow_convert=True)

        if var not in self.variables:
            # TODO: Raise an actual exception here.
            # Old comment:
            # this variable isn't in the graph... don't raise an
            # exception here, just return silently because it makes it
            # easier to implement some optimizations for
            # multiple-output ops
            # raise ValueError()
            return

        if config.compute_test_value != "off":
            try:
                tval = pytensor.graph.op.get_test_value(var)
                new_tval = pytensor.graph.op.get_test_value(new_var)
            except TestValueError:
                pass
            else:
                tval_shape = getattr(tval, "shape", None)
                new_tval_shape = getattr(new_tval, "shape", None)
                if tval_shape != new_tval_shape:
                    raise AssertionError(
                        "The replacement variable has a test value with "
                        "a shape different from the original variable's "
                        f"test value. Original: {tval_shape}, new: {new_tval_shape}"
                    )

        for node, i in list(self.clients[var]):
            self.change_node_input(
                node, i, new_var, reason=reason, import_missing=import_missing
            )

    def replace_all(self, pairs: Iterable[tuple[Variable, Variable]], **kwargs) -> None:
        """Replace variables in the `FunctionGraph` according to ``(var, new_var)`` pairs in a list."""
        for var, new_var in pairs:
            self.replace(var, new_var, **kwargs)

    def remove_node(self, node: Apply, reason: str | None = None):
        """Remove an `Apply` node from the `FunctionGraph`.

        This will remove everything that depends on the outputs of `node`, as
        well as any "orphaned" variables and nodes created by `node`'s removal.
        """

        if node not in self.apply_nodes:
            return

        self.apply_nodes.remove(node)

        if not hasattr(node.tag, "removed_by"):
            node.tag.removed_by = []

        node.tag.removed_by.append(str(reason))

        # Remove the outputs of the node (i.e. everything "below" it)
        clients = self.clients
        for out in node.outputs:
            self.variables.remove(out)

            out_clients = clients.get(out, ())
            while out_clients:
                out_client, _out_idx = out_clients.pop()

                if isinstance(out_client.op, Output):
                    self.remove_output(out_client.op.idx, remove_client=False)

                    # TODO: We could short-circuit all of the graph walking and
                    # clear everything at once when all the outputs are gone.
                    # if not self.outputs:
                    #     self.clients = {inp: [] for inp in self.inputs}
                    #     self.variables = set()
                    #     while self.apply_nodes:
                    #         node = self.apply_nodes.pop()
                    #         if not hasattr(node.tag, "removed_by"):
                    #             node.tag.removed_by = []
                    #
                    #         node.tag.removed_by.append(str(reason))
                    #
                    #         self.execute_callbacks("on_prune", node, reason)
                else:
                    self.remove_node(out_client, reason=reason)

            clients.pop(out, None)

        # Remove all the arrows pointing to this `node`, and any orphaned
        # variables created by removing those arrows
        for inp_idx, inp in enumerate(node.inputs):
            inp_clients: list[ClientType] = clients.get(inp, [])

            arrow = (node, inp_idx)

            if arrow not in inp_clients:
                continue

            inp_clients.remove(arrow)

            if not inp_clients and inp not in self.outputs:
                if inp.owner:
                    # If this input has no clients (after removing this arrow),
                    # is not an input (i.e. it has a non-`None` owner) or an
                    # output to the `FunctionGraph`, then it's an orphan

                    # We need to check whether or not this orphaned input's
                    # node is still needed in the graph
                    inp_node = inp.owner

                    if not any(
                        out in self.variables
                        for out in inp_node.outputs
                        if out is not inp
                    ):
                        self.remove_node(inp_node, reason=reason)
                else:
                    # This is an unused input
                    self.variables.remove(inp)

        # The callbacks be triggered after everything has been removed so that
        # the `FunctionGraph` state subscribers see is valid.
        self.execute_callbacks("on_prune", node, reason)

    def remove_input(self, input_idx: int, reason: str | None = None):
        """Remove the input at index `input_idx`.

        Any node that depended on such input will also be removed.
        """
        var = self.inputs.pop(input_idx)

        for client, idx in list(self.clients[var]):
            self.remove_node(client, reason=reason)

    def remove_output(
        self, output_idx: int, reason: str | None = None, remove_client: bool = True
    ):
        """Remove the output at index `output_idx` and update the indices in the clients entries.

        `FunctionGraph.clients` contains entries like ``(output(i)(var), 0)`` under
        each output variable in `FunctionGraph.outputs`.  The ``i`` values
        correspond to each output's location within the `FunctionGraph.outputs`
        list, so, when an output is removed from the graph, all these entries
        need to be updated.  This method performs those updates.

        """
        outputs = self.outputs

        # We have to update all the output indexes to the right of the removed index
        for old_idx, out in enumerate(outputs[output_idx + 1 :], output_idx + 1):
            old_client = self.get_output_client(old_idx)
            out_clients = self.clients[out]
            out_clients[out_clients.index(old_client, 0)] = (
                Output(old_idx - 1).make_node(out),
                0,
            )

        # Remove the Output Op client from the clients list
        # This is false when called from `remove_node` which removes the clients ahead of time
        if remove_client:
            output_client = self.get_output_client(output_idx)
            self.remove_client(
                outputs[output_idx], output_client, reason=reason, remove_if_empty=True
            )
        outputs.pop(output_idx)

    def attach_feature(self, feature: Feature) -> None:
        """Add a ``graph.features.Feature`` to this function graph and trigger its ``on_attach`` callback."""
        # Filter out literally identical `Feature`s
        if feature in self._features:
            return  # the feature is already present

        # Filter out functionally identical `Feature`s.
        # `Feature`s may use their `on_attach` method to raise
        # `AlreadyThere` if they detect that some
        # installed `Feature` does the same thing already
        attach = getattr(feature, "on_attach", None)
        if attach is not None:
            try:
                attach(self)
            except AlreadyThere:
                return
        # It would be nice if we could require a specific class instead of
        # a "workalike" so we could do actual error checking
        # if not isinstance(feature, Feature):
        #    raise TypeError("Expected Feature instance, got "+\
        #            str(type(feature)))

        # Add the feature
        self._features.append(feature)

    def remove_feature(self, feature: Feature) -> None:
        """Remove a feature from the graph.

        Calls ``feature.on_detach(function_graph)`` if an ``on_detach`` method
        is defined.

        """
        try:
            # Why do we catch the exception anyway?
            self._features.remove(feature)
        except ValueError:
            return
        detach = getattr(feature, "on_detach", None)
        if detach is not None:
            detach(self)

    def execute_callbacks(self, name: str, *args, **kwargs) -> None:
        """Execute callbacks.

        Calls ``getattr(feature, name)(*args)`` for each feature which has
        a method called after name.

        """
        t0 = time.perf_counter()
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                # this is safe because there is no work done inside the
                # try; the AttributeError really must come from feature.${name}
                # not existing
                continue
            tf0 = time.perf_counter()
            fn(self, *args, **kwargs)
            self.execute_callbacks_times[feature] += time.perf_counter() - tf0
        self.execute_callbacks_time += time.perf_counter() - t0

    def collect_callbacks(self, name: str, *args) -> dict[Feature, Any]:
        """Collects callbacks

        Returns a dictionary d such that ``d[feature] == getattr(feature, name)(*args)``
        For each feature which has a method called after name.
        """
        d = {}
        for feature in self._features:
            try:
                fn = getattr(feature, name)
            except AttributeError:
                continue
            d[feature] = fn(*args)
        return d

    def toposort(self) -> list[Apply]:
        r"""Return a toposorted list of the nodes.

        Return an ordering of the graph's :class:`Apply` nodes such that:

        * all the nodes of the inputs of a node are before that node, and
        * they satisfy the additional orderings provided by
          :meth:`FunctionGraph.orderings`.

        """
        return list(toposort_with_orderings(self.outputs, orderings=self.orderings()))

    def orderings(self) -> dict[Apply, list[Apply]]:
        """Return a map of node to node evaluation dependencies.

        Each key node is mapped to a list of nodes that must be evaluated
        before the key nodes can be evaluated.

        This is used primarily by the :class:`DestroyHandler` :class:`Feature`
        to ensure that the clients of any destroyed inputs have already
        computed their outputs.

        Notes
        -----
        This only calls the :meth:`Feature.orderings` method of each
        :class:`Feature` attached to the :class:`FunctionGraph`. It does not
        take care of computing the dependencies by itself.

        """
        all_orderings: list[dict] = [
            orderings
            for feature in self._features
            if (
                hasattr(feature, "orderings") and (orderings := feature.orderings(self))
            )
        ]

        if not all_orderings:
            return {}
        elif len(all_orderings) == 1:
            # If there is only 1 ordering, we reuse it directly.
            return all_orderings[0].copy()
        else:
            # If there is more than 1 ordering, combine them.
            ords: dict[Apply, list[Apply]] = {}
            for orderings in all_orderings:
                for node, prereqs in orderings.items():
                    ords.setdefault(node, []).extend(prereqs)
            return ords

    def check_integrity(self) -> None:
        """Check the integrity of nodes in the graph."""
        nodes = set(applys_between(self.inputs, self.outputs))
        if self.apply_nodes != nodes:
            nodes_missing = nodes.difference(self.apply_nodes)
            nodes_excess = self.apply_nodes.difference(nodes)
            raise Exception(
                f"The following nodes are inappropriately cached:\nmissing: {nodes_missing}\nin excess: {nodes_excess}"
            )
        clients = self.clients
        for node in nodes:
            for i, variable in enumerate(node.inputs):
                var_clients = clients[variable]
                if (node, i) not in var_clients:
                    raise Exception(
                        f"Inconsistent clients list {(node, i)} in {var_clients}"
                    )
        variables = set(vars_between(self.inputs, self.outputs))
        if set(self.variables) != variables:
            vars_missing = variables.difference(self.variables)
            vars_excess = self.variables.difference(variables)
            raise Exception(
                f"The following variables are inappropriately cached:\nmissing: {vars_missing}\nin excess: {vars_excess}"
            )
        for variable in variables:
            if (
                variable.owner is None
                and variable not in self.inputs
                and not isinstance(variable, AtomicVariable)
            ):
                raise Exception(f"Undeclared input: {variable}")
            for cl_node, i in clients[variable]:
                if isinstance(cl_node.op, Output):
                    out_idx = cl_node.op.idx
                    if self.outputs[out_idx] is not variable:
                        raise Exception(
                            f"Inconsistent clients list: {variable}, {self.outputs[out_idx]}"
                        )
                elif cl_node not in nodes:
                    raise Exception(
                        f"Client not in FunctionGraph: {variable}, {(cl_node, i)}"
                    )

                if cl_node.inputs[i] is not variable:
                    raise Exception(
                        f"Inconsistent clients list: {variable}, {cl_node.inputs[i]}"
                    )

    def __repr__(self):
        return f"FunctionGraph({', '.join(graph_as_string(self.inputs, self.outputs))})"

    def clone(
        self, check_integrity=True, clone_inner_graphs: bool = False
    ) -> "FunctionGraph":
        """Clone the graph."""
        return self.clone_get_equiv(
            check_integrity, clone_inner_graphs=clone_inner_graphs
        )[0]

    def clone_get_equiv(
        self, check_integrity: bool = True, attach_feature: bool = True, **kwargs
    ) -> tuple[
        "FunctionGraph",
        dict[Union[Apply, Variable, "Op"], Union[Apply, Variable, "Op"]],
    ]:
        """Clone the graph and return a ``dict`` that maps old nodes to new nodes.

        Parameters
        ----------
        check_integrity
            Whether or not to check the resulting graph's integrity.
        attach_feature
            Whether or not to attach `self`'s features to the cloned graph.

        Returns
        -------
        e
            The cloned `FunctionGraph`. Every node in the cloned graph is cloned.
        equiv
            A ``dict`` that maps old nodes to the new nodes.
        """
        equiv = clone_get_equiv(self.inputs, self.outputs, **kwargs)

        e = FunctionGraph(
            [cast(Variable, equiv[i]) for i in self.inputs],
            [cast(Variable, equiv[o]) for o in self.outputs],
            clone=False,
            update_mapping=self.update_mapping,
        )

        if check_integrity:
            e.check_integrity()

        if attach_feature:
            for feature in self._features:
                e.attach_feature(feature.clone())
        return e, equiv

    def __getstate__(self):
        # This is needed as some features introduce instance methods
        # This is not picklable
        d = self.__dict__.copy()
        for feature in self._features:
            for attr in getattr(feature, "pickle_rm_attr", []):
                del d[attr]

        # XXX: The `Feature` `DispatchingFeature` takes functions as parameter
        # and they can be lambda functions, making them unpicklable.

        # execute_callbacks_times have reference to optimizer, and they can't
        # be pickled as the decorators with parameters aren't pickable.
        if "execute_callbacks_times" in d:
            del d["execute_callbacks_times"]

        return d

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        for feature in self._features:
            if hasattr(feature, "unpickle"):
                feature.unpickle(self)

    def __contains__(self, item: Variable | Apply) -> bool:
        if isinstance(item, Variable):
            return item in self.variables
        elif isinstance(item, Apply):
            return item in self.apply_nodes
        else:
            raise TypeError()

    def dprint(self, **kwargs):
        """Debug print itself

        Parameters
        ----------
        kwargs:
            Optional keyword arguments to pass to debugprint function.
        """
        from pytensor.printing import debugprint

        return debugprint(self, **kwargs)
