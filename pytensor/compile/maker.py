"""Function compilation: function() and FunctionMaker.

Compilation pipeline
--------------------

1. ``function()`` — the public API entry point.
   Validates inputs, sets up profiling, then calls
   ``construct_function_ins_and_outs`` to clone the graph, discover shared
   variables, and apply ``updates`` / ``givens``.  The result is a list of
   ``In`` (SymbolicInput) objects and cloned output variables, ready for
   compilation.  Finally it resolves the ``Mode``, obtains the ``FunctionMaker``
   class from ``mode.function_maker`` (overridable by e.g. ``DebugMode``),
   and calls ``FunctionMaker(...).create()``.

2. ``FunctionMaker.__init__`` — graph construction and optimization.
   Wraps raw Variables into ``SymbolicInput`` / ``SymbolicOutput``, builds
   a ``FunctionGraph`` via ``create_fgraph`` (which also extracts update
   outputs), runs the rewriter/optimizer via ``prepare_fgraph``, and
   configures the linker.

   If an existing ``fgraph`` is passed (e.g. by ``Scan``), ``create_fgraph``
   augments it with update outputs instead of creating a new one.
   ``Function.copy`` bypasses ``create_fgraph`` entirely via ``no_fgraph_prep``.

3. ``FunctionMaker.create`` — linking and Function assembly.
   Wires up input storage containers (sharing storage for shared variables),
   calls ``linker.make_thunk()`` to produce the compiled VM, and wraps
   everything in a ``Function`` object (see ``executor.py``).

4. ``Function.__call__`` — runtime execution (see ``executor.py``).
   Places user-provided values into input containers, runs the VM,
   copies update outputs back into shared-variable containers, and
   returns the results.
"""

import copy
import logging
import re
import time
import traceback as tb
import warnings
from collections.abc import Iterable
from pathlib import Path

import pytensor
import pytensor.compile.profiling
import pytensor.misc.pkl_utils
from pytensor.compile.aliasing import (
    add_supervisor_to_fgraph,
    infer_reuse_pattern,
    insert_deepcopy,
)
from pytensor.compile.executor import Function
from pytensor.compile.io import SymbolicInput, SymbolicOutput
from pytensor.compile.mode import Mode, get_mode
from pytensor.compile.profiling import ProfileStats
from pytensor.compile.rebuild import construct_function_ins_and_outs
from pytensor.configdefaults import config
from pytensor.graph.basic import Variable
from pytensor.graph.features import Feature, PreserveVariableAttributes
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.traversal import ancestors
from pytensor.link.basic import Container


_logger = logging.getLogger("pytensor.compile.maker")


def function_dump(
    filename: str | Path,
    inputs: Iterable[Variable],
    outputs: Variable | Iterable[Variable] | None = None,
    mode: str | Mode | None = None,
    updates: Iterable[tuple[Variable, Variable]]
    | dict[Variable, Variable]
    | None = None,
    givens: Iterable[tuple[Variable, Variable]]
    | dict[Variable, Variable]
    | None = None,
    no_default_updates: bool = False,
    accept_inplace: bool = False,
    name: str | None = None,
    rebuild_strict: bool = True,
    allow_input_downcast: bool | None = None,
    profile: bool | ProfileStats | None = None,
    on_unused_input: str | None = None,
    extra_tag_to_remove: str | None = None,
    trust_input: bool = False,
):
    """
    This is helpful to make a reproducible case for problems during PyTensor
    compilation.

    Ex:

    replace `pytensor.function(...)` by
    `pytensor.function_dump('filename.pkl', ...)`.

    If you see this, you were probably asked to use this function to
    help debug a particular case during the compilation of an PyTensor
    function. `function_dump` allows you to easily reproduce your
    compilation without generating any code. It pickles all the objects and
    parameters needed to reproduce a call to `pytensor.function()`. This
    includes shared variables and their values. If you do not want
    that, you can choose to replace shared variables values with zeros by
    calling set_value(...) on them before calling `function_dump`.

    To load such a dump and do the compilation:

    >>> import pickle
    >>> import pytensor
    >>> d = pickle.load(open("func_dump.bin", "rb"))  # doctest: +SKIP
    >>> f = pytensor.function(**d)  # doctest: +SKIP

    Note:
    The parameter `extra_tag_to_remove` is passed to the StripPickler used.
    To pickle graph made by Blocks, it must be:
    `['annotations', 'replacement_of', 'aggregation_scheme', 'roles']`

    """
    d = {
        "inputs": inputs,
        "outputs": outputs,
        "mode": mode,
        "updates": updates,
        "givens": givens,
        "no_default_updates": no_default_updates,
        "accept_inplace": accept_inplace,
        "name": name,
        "rebuild_strict": rebuild_strict,
        "allow_input_downcast": allow_input_downcast,
        "profile": profile,
        "on_unused_input": on_unused_input,
        "trust_input": trust_input,
    }
    with Path(filename).open("wb") as f:
        pickler = pytensor.misc.pkl_utils.StripPickler(
            f, protocol=-1, extra_tag_to_remove=extra_tag_to_remove
        )
        pickler.dump(d)


def function(
    inputs: Iterable[Variable],
    outputs: Variable | Iterable[Variable] | None = None,
    mode: str | Mode | None = None,
    updates: Iterable[tuple[Variable, Variable]]
    | dict[Variable, Variable]
    | None = None,
    givens: Iterable[tuple[Variable, Variable]]
    | dict[Variable, Variable]
    | None = None,
    no_default_updates: bool = False,
    accept_inplace: bool = False,
    name: str | None = None,
    rebuild_strict: bool = True,
    allow_input_downcast: bool | None = None,
    profile: bool | ProfileStats | None = None,
    on_unused_input: str | None = None,
    trust_input: bool = False,
):
    """
    Return a :class:`callable object <pytensor.compile.function_maker.Function>`
    that will calculate `outputs` from `inputs`.

    Parameters
    ----------
    inputs : list of either Variable or In instances.
        Function parameters, these are not allowed to be shared variables.
    outputs : list of Variables or Out instances.
        Expressions to compute.
    mode : string or `Mode` instance.
        Compilation mode.
    updates : iterable over pairs (shared_variable, new_expression). List, tuple
              or dict.
        Updates the values for SharedVariable inputs according to these
        expressions.
    givens : iterable over pairs (Var1, Var2) of Variables. List, tuple or dict.
             The Var1 and Var2 in each pair must have the same Type.
        Specific substitutions to make in the computation graph (Var2 replaces
        Var1).
    no_default_updates: either bool or list of Variables
        If True, do not perform any automatic update on Variables. If False
        (default), perform them all. Else, perform automatic updates on all
        Variables that are neither in "updates" nor in "no_default_updates".
    accept_inplace : bool
        True iff the graph can contain inplace operations prior to the
        optimization phase (default is False). *Note* this parameter is unsupported,
        and its use is not recommended.
    name : str
        An optional name for this function. The profile mode will print the time
        spent in this function.
    rebuild_strict : bool
        True (Default) is the safer and better tested setting, in which case
        `givens` must substitute new variables with the same Type as the
        variables they replace.
        False is a you-better-know-what-you-are-doing setting, that permits
        `givens` to replace variables with new variables of any Type.
        The consequence of changing a Type is that all results depending on that
        variable may have a different Type too (the graph is rebuilt from inputs
        to outputs). If one of the new types does not make sense for one of the
        Ops in the graph, an Exception will be raised.
    allow_input_downcast: bool or None
        True means that the values passed as inputs when calling the function
        can be silently down-casted to fit the dtype of the corresponding
        Variable, which may lose precision. False means that it will only be
        cast to a more general, or precise, type. None (default) is almost like
        False, but allows down-casting of Python float scalars to floatX.
    profile: None, True, or ProfileStats instance
        Accumulate profiling information into a given ProfileStats instance.
        If argument is `True` then a new ProfileStats instance will be used.
        If argument is a string, a new ProfileStats instance will be created
        with that string as its ``message`` attribute.
        This profiling object will be available via self.profile.
    on_unused_input
        What to do if a variable in the 'inputs' list is not used in the graph.
        Possible values are 'raise', 'warn', 'ignore' and None.
    trust_input: bool, default False
        If True, no input validation checks are performed when the function is
        called. This includes checking the number of inputs, their types and
        that multiple inputs are not aliased to each other. Failure to meet any
        of these conditions can lead to computational errors or to the
        interpreter crashing.

    Returns
    -------
    :class:`pytensor.compile.function_maker.Function` instance
        A callable object that will compute the outputs (given the inputs) and
        update the implicit function arguments according to the `updates`.

    Notes
    -----
    Regarding givens: Be careful to make sure that these
    substitutions are independent--behaviour when Var1 of one pair
    appears in the graph leading to Var2 in another expression is
    undefined.  Replacements specified with givens are different
    from optimizations in that Var2 is not expected to be
    equivalent to Var1.

    """
    if not isinstance(inputs, list | tuple):
        raise TypeError(
            "Input variables of a PyTensor function should be "
            "contained in a list, even when there is a single "
            "input."
        )

    # Check for duplicate input variables
    seen = {}
    for i, inp in enumerate(inputs):
        v = getattr(inp, "variable", inp)
        if v in seen:
            raise ValueError(
                f"Variable {v} is used twice in inputs to pytensor.function, "
                f"at indices {seen[v]} and {i}. Please do not duplicate "
                "variables in the inputs list."
            )
        seen[v] = i

    if name is None:
        # Determine possible file names
        source_file = re.sub(r"\.pyc?", ".py", __file__)
        compiled_file = source_file + "c"

        stack = tb.extract_stack()
        idx = len(stack) - 1

        last_frame = stack[idx]
        if last_frame[0] == source_file or last_frame[0] == compiled_file:
            func_frame = stack[idx - 1]
            while "pytensor/graph" in func_frame[0] and idx > 0:
                idx -= 1
                # This can happen if we call var.eval()
                func_frame = stack[idx - 1]
            name = func_frame[0] + ":" + str(func_frame[1])

    if profile is None:
        profile = config.profile or config.print_global_stats
        if profile is False:
            profile = None
    if profile is True:
        profile = ProfileStats(message=name)
    elif isinstance(profile, str):
        profile = ProfileStats(message=profile)

    inputs, cloned_outputs = construct_function_ins_and_outs(
        inputs,
        outputs,
        updates=updates,
        givens=givens,
        no_default_updates=no_default_updates,
        rebuild_strict=rebuild_strict,
        allow_input_downcast=allow_input_downcast,
    )

    mode = get_mode(mode)
    m = mode.function_maker(
        inputs,
        cloned_outputs,
        mode,
        accept_inplace=accept_inplace,
        profile=profile,
        on_unused_input=on_unused_input,
        name=name,
        trust_input=trust_input,
    )
    return m.create()


class UnusedInputError(Exception):
    """
    A symbolic input passed to function is not needed.

    """



class FunctionMaker:
    """Compile a ``FunctionGraph`` and produce a ``Function``.

    ``__init__`` builds the ``FunctionGraph`` (via ``create_fgraph``), runs graph
    rewrites (via ``prepare_fgraph``), and configures the linker.
    ``create`` then links the graph to produce a VM and wraps everything in a
    ``Function``.

    The resulting ``FunctionMaker`` instance is stored as ``Function.maker``
    and is used for pickling, copying, and accessing compilation artifacts
    at runtime.

    ``Mode`` classes can override which maker is used via the
    ``mode.function_maker`` property (e.g. ``DebugMode`` substitutes its own
    ``_Maker`` subclass).

    Parameters
    ----------
    inputs : list of SymbolicInput instances
    outputs : list of SymbolicOutput instances
        Outputs may also be a single Variable (not a list), in which case the
        functions produced by FunctionMaker will return their output value
        directly.
    mode : Mode instance
        Telling FunctionMaker how to rewrite and link. None means to use the
        `config.mode`.
    accept_inplace : bool
        True iff it is acceptable to have inplace operations in the graph from
        the inputs to the outputs.
    on_unused_input : {'raise', 'warn', 'ignore', None}
        What to do if a variable in the 'inputs' list is not used in the graph.
    name : str
        An optional name for this function. If used, the profile mode will
        print the time spent in this function.
    trust_input : bool, default False
        If True, no input validation checks are performed when the function is
        called.
    """

    @staticmethod
    def create_fgraph(
        input_specs: list[SymbolicInput],
        output_specs: list[SymbolicOutput],
        accept_inplace: bool = False,
        fgraph: FunctionGraph | None = None,
        features: list[type[Feature]] = [PreserveVariableAttributes],
        force_clone=False,
    ) -> tuple[FunctionGraph, list[SymbolicOutput]]:
        """Make or set up a `FunctionGraph` from input and output specs.

        Any `SymbolicInput` whose `update` field is not ``None`` will add
        a corresponding output to the `FunctionGraph`.  Returns the
        `FunctionGraph` and a list of `SymbolicOutput` for the updates.

        If `accept_inplace` is ``False``, the graph will be checked for
        in-place operations and an exception raised if any are found.

        If `fgraph` is ``None``, a new `FunctionGraph` is created.
        The graph is cloned only if `force_clone` is ``True`` or if
        any input variable has an owner (i.e. is not a root variable).

        """
        updates = []
        update_mapping = {}
        out_idx = len(output_specs)
        for idx, input_spec in enumerate(input_specs):
            if input_spec.update is not None:
                updates.append(input_spec.update)
                update_mapping[out_idx] = idx
                out_idx += 1

        found_updates = []
        if fgraph and fgraph.update_mapping is None:
            fgraph.update_mapping = update_mapping
            for update in updates:
                fgraph.add_output(update, reason="create_fgraph")

            found_updates.extend(map(SymbolicOutput, updates))
        elif fgraph is None:
            input_vars = [spec.variable for spec in input_specs]
            clone = force_clone or any(var.owner is not None for var in input_vars)

            fgraph = FunctionGraph(
                input_vars,
                [spec.variable for spec in output_specs] + updates,
                update_mapping=update_mapping,
                clone=clone,
            )

            found_updates.extend(map(SymbolicOutput, updates))

        add_supervisor_to_fgraph(
            fgraph=fgraph, input_specs=input_specs, accept_inplace=accept_inplace
        )

        for feature in features:
            fgraph.attach_feature(feature())

        return fgraph, found_updates

    @staticmethod
    def wrap_in(input):
        if isinstance(input, (SymbolicInput)):
            return input
        elif isinstance(input, Variable):
            # r -> SymbolicInput(variable=r)
            return SymbolicInput(input)
        elif isinstance(input, list | tuple):
            # (r, u) -> SymbolicInput(variable=r, update=u)
            if len(input) == 2:
                return SymbolicInput(input[0], update=input[1])
            else:
                raise TypeError(
                    f"Expected two elements in the list or tuple; got {input}"
                )
        else:
            raise TypeError(
                f"Unknown input type: {type(input)} ({input}), expected Variable "
                "instance"
            )

    @staticmethod
    def wrap_out(output):
        if isinstance(output, SymbolicOutput):
            return output
        elif isinstance(output, Variable):
            return SymbolicOutput(output)
        else:
            raise TypeError(f"Unknown output type: {type(output)} ({output})")

    @staticmethod
    def check_unused_inputs(inputs, outputs, on_unused_input):
        if on_unused_input is None:
            on_unused_input = config.on_unused_input

        if on_unused_input == "ignore":
            return

        # There should be two categories of variables in inputs:
        #  - variables that have to be provided (used_inputs)
        #  - shared variables that will be updated
        used_inputs = list(
            ancestors(
                (
                    [o.variable for o in outputs]
                    + [
                        i.update
                        for i in inputs
                        if getattr(i, "update", None) is not None
                    ]
                ),
                blockers=[i.variable for i in inputs],
            )
        )

        msg = (
            "pytensor.function was asked to create a function computing "
            "outputs given certain inputs, but the provided input "
            "variable at index %i is not part of the computational graph "
            "needed to compute the outputs: %s.\n%s"
        )
        warn_msg = (
            "To make this warning into an error, you can pass the "
            "parameter on_unused_input='raise' to pytensor.function. "
            "To disable it completely, use on_unused_input='ignore'."
        )
        err_msg = (
            "To make this error into a warning, you can pass the "
            "parameter on_unused_input='warn' to pytensor.function. "
            "To disable it completely, use on_unused_input='ignore'."
        )

        for i in inputs:
            if (i.variable not in used_inputs) and (i.update is None):
                if on_unused_input == "warn":
                    warnings.warn(
                        msg % (inputs.index(i), i.variable, warn_msg), stacklevel=6
                    )
                elif on_unused_input == "raise":
                    raise UnusedInputError(msg % (inputs.index(i), i.variable, err_msg))
                else:
                    raise ValueError(
                        "Invalid value for keyword on_unused_input of pytensor.function: "
                        f"'{on_unused_input}'.\n"
                        "Valid values are 'raise', 'warn', and 'ignore'."
                    )

    @staticmethod
    def prepare_fgraph(
        inputs,
        outputs,
        additional_outputs,
        fgraph: FunctionGraph,
        mode: "Mode",
        profile,
    ):
        rewriter = mode.optimizer

        try:
            start_rewriter = time.perf_counter()

            rewriter_profile = None
            rewrite_time = None

            with config.change_flags(
                mode=mode,
                traceback__limit=config.traceback__compile_limit,
            ):
                rewriter_profile = rewriter(fgraph)

                end_rewriter = time.perf_counter()
                rewrite_time = end_rewriter - start_rewriter
                _logger.debug(f"Rewriting took {rewrite_time:f} seconds")

                # Add deep copy to respect the memory interface
                insert_deepcopy(fgraph, inputs, outputs + additional_outputs)
        finally:
            # If the rewriter got interrupted
            if rewrite_time is None:
                end_rewriter = time.perf_counter()
                rewrite_time = end_rewriter - start_rewriter

            pytensor.compile.profiling.total_graph_rewrite_time += rewrite_time

            if profile:
                if rewriter_profile is None and hasattr(rewriter, "pre_profile"):
                    rewriter_profile = rewriter.pre_profile

                profile.rewriting_time += rewrite_time

                if config.profile_optimizer:
                    profile.rewriter_profile = (rewriter, rewriter_profile)
            elif config.profile_optimizer and profile is not False:
                # If False, it means the profiling for that function was
                # explicitly disabled
                warnings.warn(
                    (
                        "config.profile_optimizer requires config.profile to "
                        " be set to True as well"
                    ),
                    stacklevel=3,
                )

        if not hasattr(mode.linker, "accept"):
            raise ValueError(
                "'linker' parameter of FunctionMaker should be "
                f"a Linker with an accept method or one of {list(pytensor.compile.mode.predefined_linkers)}"
            )

    def __init__(
        self,
        inputs,
        outputs,
        mode=None,
        accept_inplace=False,
        function_class=Function,
        profile=None,
        on_unused_input=None,
        fgraph=None,
        name=None,
        no_fgraph_prep=False,
        trust_input=False,
    ):
        if profile:
            self._compile_start = time.perf_counter()

        # Save the provided mode, not the instantiated mode.
        # The instantiated mode don't pickle and if we unpickle an PyTensor
        # function and it get re-compiled, we want the current rewriter to be
        # used, not the rewriter when it was saved.
        self.mode = mode
        mode = pytensor.compile.mode.get_mode(mode)

        # Assert old way of working isn't used
        if getattr(mode, "profile", None):
            raise TypeError("profile passed via 'mode'. This isn't supported anymore")
        self.profile = profile
        if profile and config.cxx:
            # This is very important:
            # 1) We preload the cache here to not have its timing
            #    included with the rewrites.
            # 2) Do not refresh the cache here by default. It cause
            #    too much execution time during testing as we compile
            #    much more functions then the number of compile c
            #    module.
            start_get_cache = time.perf_counter()
            pytensor.link.c.basic.get_module_cache().refresh()
            get_cache_time = time.perf_counter() - start_get_cache
            self.profile.linker_time += get_cache_time
            self.profile.preload_cache_time += get_cache_time

        # Handle the case where inputs and/or outputs is a single
        # Variable (not in a list)
        unpack_single = False
        return_none = False
        if outputs is None:
            return_none = True
            outputs = []
        if not isinstance(outputs, list | tuple):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, list | tuple):
            inputs = [inputs]

        # Wrap them in In or Out instances if needed.
        inputs = [self.wrap_in(i) for i in inputs]

        outputs = [self.wrap_out(o) for o in outputs]

        # Check if some input variables are unused
        self.check_unused_inputs(inputs, outputs, on_unused_input)

        fgraph, found_updates = self.create_fgraph(
            inputs, outputs, accept_inplace, fgraph=fgraph
        )

        if fgraph.profile is None:
            fgraph.profile = profile

        self.fgraph = fgraph

        if not no_fgraph_prep:
            self.prepare_fgraph(inputs, outputs, found_updates, fgraph, mode, profile)

        assert len(fgraph.outputs) == len(outputs + found_updates)

        # The 'no_borrow' outputs are the ones for which that we can't
        # return the internal storage pointer.
        no_borrow = [
            output
            for output, spec in zip(
                fgraph.outputs, outputs + found_updates, strict=True
            )
            if not spec.borrow
        ]

        linker = copy.copy(mode.linker)

        if no_borrow:
            self.linker = linker.accept(
                fgraph,
                no_recycling=infer_reuse_pattern(fgraph, no_borrow),
                profile=profile,
            )
        else:
            self.linker = linker.accept(fgraph, profile=profile)

        if hasattr(linker, "accept_var_updates"):
            # TODO: This is a hack that makes `VMLinker` aware of updates; Clean up

            # Reconstruct the full "updates" dictionary, mapping from FunctionGraph input
            # variables to the fgraph outputs that will replace their values.
            updated_vars = {
                fgraph.inputs[in_idx]: fgraph.outputs[out_idx]
                for out_idx, in_idx in fgraph.update_mapping.items()
            }
            self.linker.accept_var_updates(updated_vars)

        fgraph.name = name
        self.inputs = inputs

        # TODO: Get rid of all this `expanded_inputs` nonsense
        self.expanded_inputs = inputs
        self.outputs = outputs
        self.unpack_single = unpack_single
        self.return_none = return_none
        self.accept_inplace = accept_inplace
        self.function_class = function_class
        self.on_unused_input = on_unused_input  # Used for the pickling/copy
        self.name = name
        self.trust_input = trust_input
        self.required = [(i.value is None) for i in self.inputs]

    def create(self, input_storage=None, storage_map=None):
        """
        Create a function.

        Parameters
        ----------
        input_storage
            A list matching the inputs list and providing default values if the
            default for an input is None, then that input is a required input.
            For an input with an update, the default acts as initialization.
        """

        if input_storage is None:
            input_storage = [getattr(i, "value", None) for i in self.inputs]
        # list of independent one-element lists, will be passed to the linker
        input_storage_lists = []

        # The following loop is to fill in the input_storage_lists and
        # defaults lists.
        assert len(self.expanded_inputs) == len(input_storage)
        for i, (input, input_storage_i) in enumerate(
            zip(self.expanded_inputs, input_storage, strict=True)
        ):
            # Replace any default value given as a variable by its
            # container.  Note that this makes sense only in the
            # context of shared variables, but for now we avoid
            # dealing directly with them to avoid dependency on the
            # shared variables work-in-progress repository.
            if isinstance(input_storage_i, Variable):
                input_storage_i = input_storage_i.container

            if isinstance(input_storage_i, Container):
                # If the default is a Container, this means we want to
                # share the same storage. This is done by appending
                # input_storage_i.storage to input_storage_lists.
                input_storage_lists.append(input_storage_i.storage)

            else:
                # Normal case: one new, independent storage unit
                input_storage_lists.append([input_storage_i])

            required = self.required[i]

            # shared variables need neither be input by the user nor refed
            if input.shared:
                assert not required

        # Get a function instance
        start_linker = time.perf_counter()
        start_import_time = pytensor.link.c.cmodule.import_time

        with config.change_flags(traceback__limit=config.traceback__compile_limit):
            _fn, _i, _o = self.linker.make_thunk(
                input_storage=input_storage_lists, storage_map=storage_map
            )

        end_linker = time.perf_counter()

        linker_time = end_linker - start_linker
        pytensor.compile.profiling.total_time_linker += linker_time
        _logger.debug(f"Linker took {linker_time:f} seconds")
        if self.profile:
            self.profile.linker_time += linker_time
            _fn.time_thunks = self.profile.flag_time_thunks
            import_time = pytensor.link.c.cmodule.import_time - start_import_time
            self.profile.import_time += import_time

        fn = self.function_class(
            vm=_fn,
            input_storage=_i,
            output_storage=_o,
            outputs=self.outputs,
            unpack_single=self.unpack_single,
            return_none=self.return_none,
            maker=self,
            trust_input=self.trust_input,
            name=self.name,
        )

        fn.profile = self.profile

        if self.profile and hasattr(self, "_compile_start"):
            self.profile.compile_time += time.perf_counter() - self._compile_start
            self.profile.nb_nodes = len(self.fgraph.apply_nodes)

        return fn
