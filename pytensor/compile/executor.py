"""Compiled function runtime: the Function callable and its pickle support."""

import copy
import copyreg
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np

import pytensor
from pytensor.compile.debug import profiling
from pytensor.compile.debug.profiling import ProfileStats
from pytensor.compile.io import In, SymbolicOutput
from pytensor.configdefaults import config
from pytensor.graph.basic import clone_get_equiv
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import HasInnerGraph
from pytensor.graph.utils import get_variable_trace_string
from pytensor.link.basic import Container
from pytensor.link.utils import raise_with_op


if TYPE_CHECKING:
    from pytensor.compile.maker import FunctionMaker
    from pytensor.link.vm import VM


class AliasedMemoryError(Exception):
    """
    Memory is aliased that should not be.

    """


# A sentinel for duplicate entries
DUPLICATE = object()


class Function:
    r"""A class that wraps the execution of a `VM` making it easier for use as a "function".

    `Function` is the callable object that does computation.  It has the storage
    of inputs and outputs, performs the packing and unpacking of inputs and
    return values. It implements the square-bracket indexing so that you can
    look up the value of a symbolic node.

    Functions are copyable via `Function.copy` and the `copy.copy` interface.
    When a function is copied, this instance is duplicated. Contrast with
    self.maker (instance of `FunctionMaker`) that is shared between copies.
    The meaning of copying a function is that the containers and their current
    values will all be duplicated. This requires that mutable inputs be
    copied, whereas immutable inputs may be shared between copies.

    A Function instance is hashable, on the basis of its memory address (its
    id).
    A Function instance is only equal to itself.
    A Function instance may be serialized using the `pickle` or
    `cPickle` modules.  This will save all default inputs, the graph,
    and WRITEME to the pickle file.

    A `Function` instance has a `Function.trust_input` field that defaults to
    ``False``. When ``True``, the `Function` will skip all checks on the
    inputs.

    """

    __slots__ = (
        "_clear_input_storage_data",
        "_clear_output_storage_data",
        "_finder",
        "_has_updates",
        "_input_storage_data",
        "_inv_finder",
        "_n_returned_outputs",
        "_n_unnamed_inputs",
        "_named_inputs",
        "_nodes_with_inner_function",
        "_potential_aliased_input_groups",
        "_update_input_storage",
        "input_storage",
        "maker",
        "name",
        "output_storage",
        "outputs",
        "profile",
        "return_none",
        "trust_input",
        "unpack_single",
        "vm",
    )

    pickle_aliased_memory_strategy = "warn"
    """
    How to deal with pickling finding aliased storage.

    Meaningful settings are: 'ignore', 'warn', 'raise'.

    If the value is 'warn', then a message will be printed to stderr
    if aliased storage is detected during pickle.dump.

    If the value is 'raise', then an AliasedMemoryError will be raised
    if aliased storage is detected during pickle.dump.
    """

    def __init__(
        self,
        vm: "VM",
        input_storage: list[Container],
        output_storage: list[Container],
        outputs,
        unpack_single: bool,
        return_none: bool,
        maker: "FunctionMaker",
        trust_input: bool = False,
        name: str | None = None,
    ):
        """
        Parameters
        ----------
        vm
            A `VM` instance that evaluates the graph when called.
        input_storage
            List of storage cells for each input.
        output_storage
            List of storage cells for each output.
        outputs
            TODO
        unpack_single
            For outputs lists of length 1, should the 0'th element be
            returned directly?
        return_none
            Whether the function should return ``None`` or not.
        maker
            The `FunctionMaker` that created this instance.
        trust_input : bool, default False
            If True, no input validation checks are performed when the function is
            called. This includes checking the number of inputs, their types and
            that multiple inputs are not aliased to each other. Failure to meet any
            of these conditions can lead to computational errors or to the
            interpreter crashing.
        name
            A string name.
        """
        self.vm = vm
        self.input_storage = input_storage
        self.output_storage = output_storage
        self.outputs = outputs
        self.unpack_single = unpack_single
        self.return_none = return_none
        self.maker = maker
        self.profile = None  # reassigned in FunctionMaker.create
        self.trust_input = trust_input  # If True, we don't check the input parameter
        self.name = name

        assert len(self.input_storage) == len(self.maker.fgraph.inputs)
        assert len(self.output_storage) == len(self.maker.fgraph.outputs)

        # Group indexes of inputs that are potentially aliased to each other
        # Note: Historically, we only worried about aliasing inputs if they belonged to the same type,
        #  even though there could be two distinct types that use the same kinds of underlying objects.
        potential_aliased_input_groups = []
        for inp in maker.inputs:
            # If the input is a shared variable, the memory region is under PyTensor control
            # and can't be aliased.
            if not (
                isinstance(inp, In)
                and inp.borrow
                and not inp.shared
                and hasattr(inp.variable.type, "may_share_memory")
            ):
                continue

            for group in potential_aliased_input_groups:
                # If one is super of the other, that means one could be replaced by the other
                if any(
                    inp.variable.type.is_super(other_inp.variable.type)
                    or other_inp.variable.type.is_super(inp.variable.type)
                    for other_inp in group
                ):
                    group.append(inp)
                    break
            else:  # no break
                # Input makes a new group
                potential_aliased_input_groups.append([inp])

        # Potential aliased inputs are those that belong to the same group
        self._potential_aliased_input_groups: tuple[tuple[int, ...], ...] = tuple(
            tuple(maker.inputs.index(inp) for inp in group)
            for group in potential_aliased_input_groups
            if len(group) > 1
        )

        self._finder = finder = {}
        self._inv_finder = inv_finder = {}
        self._named_inputs = named_inputs = []
        self._n_unnamed_inputs = 0
        # Initialize the storage
        # this loop works by modifying the elements (as variable c) of
        # self.input_storage inplace.
        for i, (input, c) in enumerate(
            zip(self.maker.expanded_inputs, self.input_storage, strict=True)
        ):
            c.strict = getattr(input, "strict", False)
            c.allow_downcast = getattr(input, "allow_downcast", None)
            c.required = input.value is None
            c.implicit = input.implicit
            # this is a count of how many times the input has been
            # provided (reinitialized to 0 on __call__)
            c.provided = 0
            finder[i] = c
            finder[input.variable] = c
            if input.name not in finder:
                finder[input.name] = c
            else:
                finder[input.name] = DUPLICATE
            if input.name is None:
                self._n_unnamed_inputs += 1
            else:
                named_inputs.append(input.name)
            inv_finder[c] = input

        update_storage = [
            container
            for inp, container in zip(
                self.maker.expanded_inputs, input_storage, strict=True
            )
            if inp.update is not None
        ]
        # Updates are the last inner outputs that are not returned by Function.__call__
        self._has_updates = len(update_storage) > 0
        self._n_returned_outputs = len(self.output_storage) - len(update_storage)

        # Function.__call__ is responsible for updating the inputs, unless the vm promises to do it itself
        self._update_input_storage: tuple[int, Container] = ()
        if getattr(vm, "need_update_inputs", True):
            self._update_input_storage = tuple(
                zip(
                    range(self._n_returned_outputs, len(output_storage)),
                    update_storage,
                    strict=True,
                )
            )

        self._input_storage_data = tuple(
            container.storage for container in input_storage
        )

        # In every function call we place inputs in the input_storage, and the vm places outputs in the output_storage
        # After the call, we want to erase (some of) these references, to allow Python to GC them if unused
        self._clear_input_storage_data = tuple(
            container.storage for container in input_storage if container.required
        )
        # This is only done when `vm.allow_gc` is True, which can change at runtime.
        self._clear_output_storage_data = tuple(
            container.storage
            for container, variable in zip(
                self.output_storage, self.maker.fgraph.outputs, strict=True
            )
            if variable.owner is not None  # Not a constant output
        )

        self._nodes_with_inner_function = [
            node
            for node in self.maker.fgraph.apply_nodes
            if isinstance(node.op, HasInnerGraph)
        ]

    def __copy__(self):
        """
        Copy a function. Copied function have separate intermediate
        storages and output storages with original function
        """
        return self.copy()

    def copy(
        self,
        share_memory: bool = False,
        swap: dict | None = None,
        delete_updates: bool = False,
        name: str | None = None,
        profile: bool | str | ProfileStats | None = None,
    ):
        """
        Copy this function. Copied function will have separated maker and
        fgraph with original function. User can choose whether to separate
        storage by changing the share_memory arguments.

        Parameters
        ----------
        share_memory : boolean
            When True, two function share intermediate storages(storages except input and
            output storages). Otherwise two functions will only share partial
            storages and same maker. If two functions share memory and
            allow_gc=False, this will increase executing speed and save memory.

        swap : dict
            Dictionary that map old SharedVariables to new
            SharedVariables. Default is None.
            NOTE: The shared variable swap in only done in the new returned
            function, not in the user graph.

        delete_updates : boolean
            If True, Copied function will not have updates.
        name : string
            If provided, will be the name of the new
            Function. Otherwise, it will be old + " copy"

        profile : bool | str | ProfileStats | None
            as pytensor.function profile parameter

        Returns
        -------
        pytensor.Function
            Copied pytensor.Function

        Examples
        --------
        Copy a function but swap its shared state:

        >>> import pytensor
        >>> import pytensor.tensor as pt
        >>> state = pytensor.shared(0)
        >>> x = pt.iscalar("x")
        >>> f = pytensor.function([x], state, updates={state: state + x})
        >>> other_state = pytensor.shared(100)
        >>> g = f.copy(swap={state: other_state})

        ``copy`` and random shared variables:

        >>> rng = pt.random.shared_rng(seed=123)
        >>> next_rng, draw = rng.uniform()
        >>> f_rng = pytensor.function([], draw, updates={rng: next_rng})
        >>> g_rng = f_rng.copy()
        >>> # `f_rng` and `g_rng` share RNG state by default.
        >>> _ = f_rng()  # doctest: +SKIP
        >>> _ = g_rng()  # doctest: +SKIP

        To give the copied function an independent RNG stream, swap the RNG
        shared variable:

        >>> new_rng = pt.random.shared_rng(seed=123)
        >>> g_independent = f_rng.copy(swap={rng: new_rng})
        """

        # helper function
        def checkSV(sv_ori, sv_rpl):
            """
            Assert two SharedVariable follow some restrictions:
                1. same type
                2. same shape or dim?
            """
            SharedVariable = pytensor.tensor.sharedvar.SharedVariable
            assert isinstance(sv_ori, SharedVariable), (
                "Key of swap should be SharedVariable, given:",
                sv_ori,
                " type",
                type(sv_ori),
            )
            assert isinstance(sv_rpl, SharedVariable), (
                "Value of swap should be SharedVariable, given:",
                sv_rpl,
                "type",
                type(sv_ori),
            )
            assert sv_rpl.type.in_same_class(sv_ori.type), (
                "Type of given SharedVariable conflicts with original one",
                "Type of given SharedVariable:",
                sv_rpl.type,
                "Type of original SharedVariable:",
                sv_ori.type,
            )

        maker = self.maker

        # Copy Ins and their storage.
        # so that they have different storage as their value
        ins = [copy.copy(input) for input in maker.inputs]

        # Delete update output in fgraph and updates In instances if needed
        if delete_updates:
            # The first len(maker.outputs) variables are original variables.
            # The rest are the updates.
            out_vars = maker.fgraph.outputs[: len(maker.outputs)]
        else:
            out_vars = maker.fgraph.outputs

        # Init new fgraph using copied variables and get memo
        # memo: a dict that map old variables to new variables
        memo = clone_get_equiv(maker.fgraph.inputs, out_vars)
        fg_cpy = FunctionGraph(
            [memo[i] for i in maker.fgraph.inputs],
            [memo[o] for o in out_vars],
            clone=False,
        )
        fg_cpy.update_mapping = maker.fgraph.update_mapping

        # Re initialize Outs and swap update and variable in Ins
        # By doing this, we can pass FunctionMaker.check_unused_inputs()
        if delete_updates:
            outs = list(map(SymbolicOutput, fg_cpy.outputs[: len(maker.outputs)]))
        else:
            outs = list(map(SymbolicOutput, fg_cpy.outputs))

        for out_ori, out_cpy in zip(maker.outputs, outs, strict=False):
            out_cpy.borrow = out_ori.borrow

        # swap SharedVariable
        if swap is not None:
            exist_svs = [i.variable for i in maker.inputs]

            # Check if given ShareVariables exist
            for sv in swap:
                if sv not in exist_svs:
                    raise ValueError(f"SharedVariable: {sv.name} not found")

            # Swap SharedVariable in fgraph and In instances
            for index, (i, in_v) in enumerate(zip(ins, fg_cpy.inputs, strict=True)):
                # Variables in maker.inputs are defined by user, therefore we
                # use them to make comparison and do the mapping.
                # Otherwise we don't touch them.
                var = maker.inputs[index].variable

                if var in swap:
                    swap_sv = swap[var]
                    checkSV(i.variable, swap_sv)

                    # swap variable and value of In instances
                    i.variable = swap_sv
                    i.value = swap_sv.container

                    # In the fgraph we use the cloned SharedVariable
                    swap_sv = swap_sv.clone()

                    # Swap SharedVariable in fgraph
                    # if inputs was replaced, change self.inputs
                    fg_cpy.inputs[index] = swap_sv
                    fg_cpy.replace(in_v, swap_sv, reason="Swap SV")

        # Delete update if needed
        rev_update_mapping = {v: k for k, v in fg_cpy.update_mapping.items()}
        for n, (inp, in_var) in enumerate(zip(ins, fg_cpy.inputs, strict=True)):
            inp.variable = in_var
            if not delete_updates and inp.update is not None:
                out_idx = rev_update_mapping[n]
                inp.update = fg_cpy.outputs[out_idx]
            else:
                inp.update = None

        if delete_updates:
            fg_cpy.update_mapping = {}

        # Construct new storage_map that map new variable to old storage,
        # so that the ensuing function shares storage with the original one
        storage_map = self.vm.storage_map
        new_storage_map = {}
        # TODO: We could share the output storage, but we must make sure
        # 2 different function call won't override each other values. This
        # is already done elsewhere, so to reuse it the user would need to
        # use Out(var, borrow=True) and maybe the mutable=True flag too.
        # But to be safe for now as it isn't documented and we aren't sure
        # it is well tested, we don't share the part of the storage_map.
        if share_memory:
            i_o_vars = maker.fgraph.inputs + maker.fgraph.outputs
            for key, val in storage_map.items():
                if key not in i_o_vars:
                    new_storage_map[memo[key]] = val

        if not name and self.name:
            name = self.name + " copy"

        input_storage = [i.value for i in ins]
        # reinitialize new maker and create new function
        if profile is None:
            profile = config.profile or config.print_global_stats
        if profile is True:
            profile = profiling.ProfileStats(message=name)
        elif isinstance(profile, str):
            profile = profiling.ProfileStats(message=profile)

        f_cpy = type(maker)(
            inputs=ins,
            outputs=outs,
            fgraph=fg_cpy,
            mode=maker.mode,
            profile=profile,
            # When removing updates containing variables
            # not used in the output function, copy
            # generates an unused implicit input.
            # We ignore the resulting errors,
            # but could change it to 'warn' if this might
            # cause problems.
            on_unused_input="ignore",
            function_class=maker.function_class,
            # As this is an rewritten graph, it can contain inplace. DebugMode
            # check that.
            accept_inplace=True,
            no_fgraph_prep=True,
            name=name,
        ).create(input_storage, storage_map=new_storage_map)

        for in_ori, in_cpy, ori, cpy in zip(
            maker.inputs,
            f_cpy.maker.inputs,
            self.input_storage,
            f_cpy.input_storage,
            strict=True,
        ):
            # Share immutable ShareVariable and constant input's storage
            swapped = swap is not None and in_ori.variable in swap

            # Using the original storage if SharedVariable will not be updated
            # and is not swapped
            if not in_ori.mutable and not swapped:
                cpy.data = ori.data
                in_cpy.value = in_ori.value

            # Reconstruct Function.finder which map Variable defined by user
            # to container, to make Function.value and Function.data work well.
            # Replace variable in new maker.inputs by the original ones.
            # So that user can swap SharedVariable in a swapped function
            container = f_cpy._finder.pop(in_cpy.variable)
            if not swapped:
                f_cpy._finder[in_ori.variable] = container
                in_cpy.variable = in_ori.variable
            else:
                f_cpy._finder[swap[in_ori.variable]] = container
                in_cpy.variable = swap[in_ori.variable]

        f_cpy.trust_input = self.trust_input
        f_cpy.unpack_single = self.unpack_single
        return f_cpy

    def _validate_inputs(self, args, kwargs):
        input_storage = self.input_storage

        if len(args) + len(kwargs) > len(input_storage):
            raise TypeError("Too many parameter passed to pytensor function")

        for arg_container in input_storage:
            arg_container.provided = 0

        # Set positional arguments
        for arg_container, arg in zip(input_storage, args):
            try:
                arg_container.storage[0] = arg_container.type.filter(
                    arg,
                    strict=arg_container.strict,
                    allow_downcast=arg_container.allow_downcast,
                )

            except Exception as e:
                i = input_storage.index(arg_container)
                function_name = "pytensor function"
                argument_name = "argument"
                if self.name:
                    function_name += ' with name "' + self.name + '"'
                if hasattr(arg, "name") and arg.name:
                    argument_name += ' with name "' + arg.name + '"'
                where = get_variable_trace_string(self.maker.inputs[i].variable)
                if len(e.args) == 1:
                    e.args = (
                        "Bad input "
                        + argument_name
                        + " to "
                        + function_name
                        + f" at index {int(i)} (0-based). {where}"
                        + e.args[0],
                    )
                else:
                    e.args = (
                        "Bad input "
                        + argument_name
                        + " to "
                        + function_name
                        + f" at index {int(i)} (0-based). {where}"
                    ) + e.args
                raise
            arg_container.provided += 1

        # Set keyword arguments
        if kwargs:  # for speed, skip the items for empty kwargs
            for key, arg in kwargs.items():
                try:
                    kwarg_container = self._finder[key]
                except KeyError:
                    # Print informative error message.
                    msg = Function._get_info_on_inputs(
                        self._named_inputs, self._n_unnamed_inputs
                    )
                    raise TypeError(f"Unknown input: {key}. {msg}")
                if kwarg_container is DUPLICATE:
                    raise TypeError(
                        f"Ambiguous name: {key} - please check the names of the inputs of your function for duplicates."
                    )
                kwarg_container.value = arg
                kwarg_container.provided += 1

        # Collect aliased inputs among the storage space
        for potential_group in self._potential_aliased_input_groups:
            args_share_memory: list[list[int]] = []
            for i in potential_group:
                i_type = self.maker.inputs[i].variable.type
                i_val = input_storage[i].storage[0]

                # Check if value is aliased with any of the values in one of the groups
                for j_group in args_share_memory:
                    if any(
                        i_type.may_share_memory(input_storage[j].storage[0], i_val)
                        for j in j_group
                    ):
                        j_group.append(i)
                        break
                else:  # no break
                    # Create a new group
                    args_share_memory.append([i])

            # Check for groups of more than one argument that share memory
            for group in args_share_memory:
                if len(group) > 1:
                    # copy all but the first
                    for i in group[1:]:
                        input_storage[i].storage[0] = copy.copy(
                            input_storage[i].storage[0]
                        )

        # Check if inputs are missing, or if inputs were set more than once, or
        # if we tried to provide inputs that are supposed to be implicit.
        for arg_container in input_storage:
            if arg_container.required and not arg_container.provided:
                raise TypeError(
                    f"Missing input: {getattr(self._inv_finder[arg_container], 'variable', self._inv_finder[arg_container])}"
                )
            if arg_container.provided > 1:
                raise TypeError(
                    f"Multiple values for input: {getattr(self._inv_finder[arg_container], 'variable', self._inv_finder[arg_container])}"
                )
            if arg_container.implicit and arg_container.provided > 0:
                raise TypeError(
                    f"Tried to provide value for implicit input: {getattr(self._inv_finder[arg_container], 'variable', self._inv_finder[arg_container])}"
                )

    def __call__(self, *args, **kwargs):
        """
        Evaluates value of a function on given arguments.

        Parameters
        ----------
        args : list
            List of inputs to the function. All inputs are required, even when
            some of them are not necessary to calculate requested subset of
            outputs.

        kwargs : dict
            The function inputs can be passed as keyword argument. For this, use
            the name of the input or the input instance as the key.

        Returns
        -------
        list
            List of outputs of the function.
        """
        if self.profile:
            t0 = time.perf_counter()

        # Reinitialize each container's 'provided' counter
        if self.trust_input:
            for storage_data, arg in zip(self._input_storage_data, args):
                storage_data[0] = arg
            if kwargs:  # for speed, skip the items for empty kwargs
                for k, arg in kwargs.items():
                    self._finder[k].storage[0] = arg
        else:
            self._validate_inputs(args, kwargs)

        # Do the actual work
        try:
            if self.profile:
                t0_fn = time.perf_counter()
                outputs = self.vm()
                dt_fn = time.perf_counter() - t0_fn
                self.maker.mode.fn_time += dt_fn
                self.profile.vm_call_time += dt_fn
            else:
                outputs = self.vm()
        except Exception:
            if hasattr(self.vm, "position_of_error"):
                # this is a new vm-provided function or c linker
                # they need this because the exception manipulation
                # done by raise_with_op is not implemented in C.
                thunk = None
                if hasattr(self.vm, "thunks"):
                    thunk = self.vm.thunks[self.vm.position_of_error]
                raise_with_op(
                    self.maker.fgraph,
                    node=self.vm.nodes[self.vm.position_of_error],
                    thunk=thunk,
                    storage_map=getattr(self.vm, "storage_map", None),
                )
            else:
                # old-style linkers raise their own exceptions
                raise

        if outputs is None:
            # Not all VMs can return outputs directly (mainly CLinker?)
            outputs = [x.storage[0] for x in self.output_storage]

        # Set updates and filter them out from the returned outputs
        if self._has_updates:
            for i, input_storage in self._update_input_storage:
                input_storage.storage[0] = outputs[i]
            outputs = outputs[: self._n_returned_outputs]

        # Remove input and output values from storage data
        if self.vm.allow_gc:
            for storage_data in self._clear_input_storage_data:
                storage_data[0] = None
            for storage_data in self._clear_output_storage_data:
                storage_data[0] = None

        if self.profile:
            profile = self.profile
            dt_call = time.perf_counter() - t0
            profiling.total_fct_exec_time += dt_call
            self.maker.mode.call_time += dt_call
            profile.fct_callcount += 1
            profile.fct_call_time += dt_call
            if hasattr(self.vm, "update_profile"):
                self.vm.update_profile(profile)
            if profile.ignore_first_call:
                profile.reset()
                profile.ignore_first_call = False

        return (
            outputs[0] if self.unpack_single else None if self.return_none else outputs
        )

    def free(self):
        """
        When allow_gc = False, clear the Variables in storage_map
        """
        # 1.no allow_gc return False
        # 2.has allow_gc, if allow_gc is False, return True
        if not self.vm.allow_gc:
            for inp_storage in self._clear_input_storage_data:
                inp_storage[0] = None

            if hasattr(self.vm, "storage_map"):
                storage_map = self.vm.storage_map
                for key, value in storage_map.items():
                    if key.owner is not None:  # Not a constant
                        value[0] = None

            for node in self._nodes_with_inner_function:
                if hasattr(node.op.fn, "free"):
                    node.op.fn.free()

    @staticmethod
    def _get_info_on_inputs(named_inputs, n_unnamed_inputs):
        """Return a human-readable description of named and un-named inputs."""
        n_named_inputs = len(named_inputs)

        def get_plural(n):
            if n > 1:
                return "s"
            else:
                return ""

        if n_named_inputs == 0:
            if n_unnamed_inputs == 0:
                msg = "The function is supposed to have no input."
            else:
                if n_unnamed_inputs == 1:
                    msg = (
                        "The function has a single input variable which has no "
                        "name, and thus cannot be assigned through a keyword"
                        " argument (use 'name=...' in a Variable's "
                        "constructor to give it a name)."
                    )
                else:
                    msg = (
                        f"The function has {n_unnamed_inputs} inputs, but none of them is named,"
                        " and thus they cannot be assigned through keyword "
                        "arguments (use 'name=...' in a Variable's "
                        "constructor to give it a name)."
                    )
        else:
            if n_unnamed_inputs == 0:
                msg = f"The function has {n_named_inputs} named input{get_plural(n_named_inputs)} ({', '.join(named_inputs)})."
            else:
                msg = (
                    f"The function has {n_named_inputs} named input{get_plural(n_named_inputs)} ({', '.join(named_inputs)}), and {n_unnamed_inputs} unnamed "
                    f"input{get_plural(n_unnamed_inputs)} which thus cannot be accessed through keyword "
                    f"argument{get_plural(n_unnamed_inputs)} (use 'name=...' in a variable's constructor "
                    "to give it a name)."
                )
        return msg

    def get_shared(self):
        """
        Return the shared variable read or updated by by this function.
        """
        return [i.variable for i in self.maker.inputs if i.implicit]

    def dprint(self, **kwargs):
        """Debug print itself

        Parameters
        ----------
        kwargs:
            Optional keyword arguments to pass to debugprint function.
        """
        from pytensor.printing import debugprint

        return debugprint(self, **kwargs)


# pickling/deepcopy support for Function


def _pickle_Function(f):
    f.free()

    input_storage = f.input_storage.copy()
    inputs_data = [x.data for x in f.input_storage]

    # HACK to detect aliased storage.
    # This is here because aliased relationships are not [currently]
    # preserved across the pickle operation
    if f.pickle_aliased_memory_strategy != "ignore":
        all_data = input_storage + inputs_data
        for i, d_i in enumerate(all_data):
            for j, d_j in enumerate(all_data):
                if (
                    (i < j)
                    and isinstance(d_i, np.ndarray)
                    and isinstance(d_j, np.ndarray)
                ):
                    if np.may_share_memory(d_i, d_j):
                        if f.pickle_aliased_memory_strategy == "warn":
                            warnings.warn(
                                "aliased relationship between "
                                f"Function arguments {d_i}, {d_j} "
                                "will not be preserved by "
                                "un-pickling operation"
                            )
                        else:
                            raise AliasedMemoryError(d_i, d_j)

    rval = (_constructor_Function, (f.maker, input_storage, inputs_data, f.trust_input))
    return rval


def _constructor_Function(maker, input_storage, inputs_data, trust_input=False):
    if not config.unpickle_function:
        return None

    f = maker.create(input_storage)
    assert len(f.input_storage) == len(inputs_data)
    for container, x in zip(f.input_storage, inputs_data, strict=True):
        if x is not None:
            assert (
                (container.data is x)
                or (isinstance(x, np.ndarray) and (container.data == x).all())
                or (container.data == x)
            )
    f.trust_input = trust_input
    return f


copyreg.pickle(Function, _pickle_Function)
