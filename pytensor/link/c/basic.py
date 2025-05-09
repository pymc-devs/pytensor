"""
Defines Linkers that deal with C implementations.

"""

import logging
import sys
from collections import defaultdict
from copy import copy
from io import StringIO
from typing import TYPE_CHECKING, Any, Optional

from pytensor.compile.compilelock import lock_ctx
from pytensor.configdefaults import config
from pytensor.graph.basic import (
    AtomicVariable,
    Constant,
    NoParams,
    io_toposort,
    vars_between,
)
from pytensor.graph.utils import MethodNotDefined
from pytensor.link.basic import Container, Linker, LocalLinker, PerformLinker
from pytensor.link.c.cmodule import (
    METH_VARARGS,
    DynamicModule,
    ExtFunction,
    GCC_compiler,
    dlimport_workdir,
)
from pytensor.link.c.cmodule import get_module_cache as _get_module_cache
from pytensor.link.c.interface import CLinkerObject, CLinkerOp, CLinkerType
from pytensor.link.utils import gc_helper, map_storage, raise_with_op, streamline
from pytensor.npy_2_compat import ndarray_c_version
from pytensor.utils import difference, uniq


if TYPE_CHECKING:
    from pytensor.graph.fg import FunctionGraph
    from pytensor.link.c.cmodule import ModuleCache

_logger = logging.getLogger("pytensor.link.c.basic")


def get_module_cache(init_args: dict[str, Any] | None = None) -> "ModuleCache":
    """

    Parameters
    ----------
    init_args
        If not None, the (k, v) pairs in this dictionary will be forwarded to
        the ModuleCache constructor as keyword arguments.

    """
    return _get_module_cache(config.compiledir, init_args=init_args)


class CodeBlock:
    """
    Represents a computation unit composed of declare, behavior, and cleanup.

    The constructor initializes a L{CodeBlock} with templatized declare,
    behavior and cleanup. The sub parameter will be used in the other
    arguments' templates. sub should contain a key called 'id' that maps to an
    identifier for this block. The identifier will be used to determine the
    failure code and a label to jump to. It should also contain a key called
    'failure_var' that contains the name of the variable that contains the error
    code.

    Parameters
    ----------
    declare
        C code that declares variables for use by the computation.
    behavior
        C code that performs the computation.
    cleanup
        C code that cleans up things allocated or incref-ed in behavior.

    """

    def __init__(self, declare, behavior, cleanup, sub):
        self.declare = declare
        self.behavior = behavior
        # the dummy is because gcc throws an error when a label's
        # right next to a closing brace (maybe there's an ignore flag
        # for that...)
        # we need the label even if cleanup is empty because the
        # behavior block jumps there on failure
        id = sub["id"]
        self.cleanup = f"__label_{id}:\n{cleanup}\ndouble __DUMMY_{id};\n"


def failure_code(sub, use_goto=True):
    """
    Code contained in sub['fail'], usually substituted for %(fail)s.

    It sets information about current error, then goto the code
    actually handling the failure, which is defined in struct_gen().

    Parameters
    ----------
    sub: dict
        Contains other code snippets that can be substituted,
        in particular 'failure_var' and 'id'.
    use_goto: bool, True by default
        Include a "goto" statement to the failure label.
        Passing False is sometimes required, in which cases we have to
        be careful to avoid executing incorrect code.

    """
    if use_goto:
        goto_statement = "goto __label_%(id)i;" % sub
    else:
        goto_statement = ""
    id = sub["id"]
    failure_var = sub["failure_var"]
    return f"""{{
        {failure_var} = {id};
        if (!PyErr_Occurred()) {{
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }}
        {goto_statement}}}"""


def failure_code_init(sub):
    """
    Code for failure in the struct init.

    Parameters:
    ----------
    sub
      Dictionary used to template the struct.
      * failure_var -> must contain a variable name to use for
      the failure code.
    """
    id = sub["id"]
    return f"""{{
        if (!PyErr_Occurred()) {{
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
            }}
        return {id};
}}"""


def code_gen(blocks):
    """
    From a list of L{CodeBlock} instances, returns a string
    that executes them all in sequence.

    Eg for C{(decl1, task1,
    cleanup1)} and C{(decl2, task2, cleanup2)} the returned string
    will be of the form:

        decl1
        decl2
        {
         task1
         {
          task2
          cleanup2
         }
         cleanup1
        }

    Parameters:
    ----------
    blocks
         List of CodeBlock instances such that
         * declarations, behavior and cleanup are in the run()
         method of the struct
    """
    decl = ""
    head = ""
    tail = ""
    for block in blocks:
        decl += block.declare
        head = f"{head}\n{{\n{block.behavior}"
        tail = f"{block.cleanup}\n}}\n{tail}"
    return decl + head + tail


def struct_gen(args, struct_builders, blocks, sub):
    """
    Generates a struct conforming to the following specifications:

    Parameters
    ----------
     args
        All of the PyObject* type, stored in the struct
        they represent the storage and must be length 1 python lists.
     struct_builders
        List of L{CodeBlock} instances such that
        * declarations are in the struct
        * behavior is in the constructor
        * cleanup is in the destructor
     blocks
        List of CodeBlock instances such that
        * declarations, behavior and cleanup are in the run()
        method of the struct
     sub
        Dictionary used to template the struct.
        * failure_var -> must contain a variable name to use for
        the failure code.

    Returns
    -------
    object
        In a nutshell, this returns code for a struct that represents
        a function with state. The state's initialization and destruction
        are handled by struct_builders and the actual behavior of the
        function is handled by blocks.

    """
    struct_decl = ""
    struct_init_head = ""
    struct_cleanup = ""

    for block in struct_builders:
        # decl are declarations that go in the struct
        # init_head are in the constructor
        # init_tail and cleanup do the same thing, but the former will
        #     be executed if any step in the constructor fails and the
        #     latter only at destruction time.
        struct_decl += block.declare
        struct_init_head = struct_init_head + f"\n{block.behavior}"
        struct_cleanup += block.cleanup

    behavior = code_gen(blocks)

    # declares the storage
    storage_decl = "\n".join(f"PyObject* {arg};" for arg in args)
    # in the constructor, sets the storage to the arguments
    storage_set = "\n".join(f"this->{arg} = {arg};" for arg in args)
    # increments the storage's refcount in the constructor
    storage_incref = "\n".join(f"Py_XINCREF({arg});" for arg in args)
    # decrements the storage's refcount in the destructor
    storage_decref = "\n".join(f"Py_XDECREF(this->{arg});" for arg in args)

    args_decl = ", ".join(f"PyObject* {arg}" for arg in args)

    # The following code stores the exception data in __ERROR, which
    # is a special field of the struct. __ERROR is a list of length 3
    # that holds the type, the value and the traceback. After storing
    # the error, we return the failure code so we know which code
    # block failed.
    failure_var = sub["failure_var"]
    do_return = f"""
        if ({failure_var}) {{
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {{err_type = Py_None;Py_INCREF(Py_None);}}
            if (!err_msg) {{err_msg = Py_None; Py_INCREF(Py_None);}}
            if (!err_traceback) {{err_traceback = Py_None; Py_INCREF(Py_None);}}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {{Py_XDECREF(old_err_type);}}
            {{Py_XDECREF(old_err_msg);}}
            {{Py_XDECREF(old_err_traceback);}}
        }}
        // The failure code is returned to index what code block failed.
        return {failure_var};
        """

    # TODO: add some error checking to make sure storage_<x> are
    # 1-element lists and __ERROR is a 3-elements list.

    name = sub["name"]
    struct_code = f"""
    namespace {{
    struct {name} {{
        PyObject* __ERROR;

        {storage_decl}
        {struct_decl}

        {name}() {{
            // This is only somewhat safe because we:
            //  1) Are not a virtual class
            //  2) Do not use any virtual classes in the members
            //  3) Deal with mostly POD and pointers

            // If this changes, we would have to revise this, but for
            // now I am tired of chasing segfaults because
            // initialization code had an error and some pointer has
            // a junk value.
            #ifndef PYTENSOR_DONT_MEMSET_STRUCT
            memset(this, 0, sizeof(*this));
            #endif
        }}
        ~{name}(void) {{
            cleanup();
        }}

        int init(PyObject* __ERROR, {args_decl}) {{
            {storage_incref}
            {storage_set}
            {struct_init_head}
            this->__ERROR = __ERROR;
            return 0;
        }}
        void cleanup(void) {{
            {struct_cleanup}
            {storage_decref}
        }}
        int run(void) {{
            int {failure_var} = 0;
            {behavior}
            {do_return}
        }}
    }};
    }}
    """

    return struct_code


# The get_<x> functions complete the return value of r.get_<x>()
# with handling of the py_<name> variable.


def get_nothing(fgraph, r, name, sub):
    """
    WRITEME

    """
    return ""


def get_c_declare(fgraph, r, name, sub):
    """
    Wrapper around c_declare that declares py_name.

    """
    # The declaration will be used by the Apply node that
    # is computing it (`r.owner`), and by each of the clients.
    # If some of these have `check_input=True` in their `.op`,
    # it means they need `r`'s dtype to be declared, so
    # we have to pass `check_input=True` to `c_declare`.
    if any(
        getattr(c.op, "check_input", config.check_input) for (c, _) in fgraph.clients[r]
    ) or (r.owner and getattr(r.owner.op, "check_input", config.check_input)):
        c_declare = r.type.c_declare(name, sub, True)
    else:
        c_declare = r.type.c_declare(name, sub, False)
    pre = f"""
    PyObject* py_{name};
    """
    return pre + c_declare


def get_c_init(fgraph, r, name, sub):
    """
    Wrapper around c_init that initializes py_name to Py_None.

    """
    pre = f"""
    py_{name} = Py_None;
    {{Py_XINCREF(py_{name});}}
    """
    return pre + r.type.c_init(name, sub)


def get_c_extract(fgraph, r, name, sub):
    """
    Wrapper around c_extract that initializes py_name from storage.

    """
    # `c_extract` is called when getting the value of an apply node's
    # input from the compute map, before being used by its clients.
    # If one of the clients has `check_input=True`, we need to perform
    # checks on the variable.
    # However that code is not used by C code of the apply node creating
    # this variable, so there is no need to check `r.owner.op.check_input`.
    if any(
        getattr(c.op, "check_input", config.check_input)
        for (c, _) in fgraph.clients[r]
        if not isinstance(c, str)
    ):
        # check_broadcast is just an hack to easily remove just the
        # broadcast check on the old GPU back-end. This check isn't
        # done in the new GPU back-end or on the CPU.
        if any(
            getattr(c.op, "check_broadcast", True)
            for (c, _) in fgraph.clients[r]
            if not isinstance(c, str)
        ):
            c_extract = r.type.c_extract(name, sub, True)
        else:
            c_extract = r.type.c_extract(name, sub, True, check_broadcast=False)
    else:
        c_extract = r.type.c_extract(name, sub, False)

    pre = f"""
    py_{name} = PyList_GET_ITEM(storage_{name}, 0);
    {{Py_XINCREF(py_{name});}}
    """
    return pre + c_extract


def get_c_extract_out(fgraph, r, name, sub):
    """
    Wrapper around c_extract_out that initializes py_name from storage.

    """
    # `c_extract_out` is used to extract an output variable from
    # the compute map, to be used as pre-allocated memory for `r`
    # before its value gets computed.
    # If the node producing `r` has `check_input=True`, it may
    # also perform type checks on the initial value of the output,
    # so we need to pass `check_input=True` to `c_extract_out`.
    # However, that code is not used by potential clients of `r`,
    # so we do not need to check them.
    check_input = getattr(r.owner.op, "check_input", config.check_input)
    # check_broadcast is just an hack to easily remove just the
    # broadcast check on the old GPU back-end. This check isn't
    # done in the new GPU back-end or on the CPU.
    if getattr(r.owner.op, "check_broadcast", True):
        c_extract = r.type.c_extract_out(name, sub, check_input)
    else:
        c_extract = r.type.c_extract_out(name, sub, check_input, check_broadcast=False)

    pre = f"""
    py_{name} = PyList_GET_ITEM(storage_{name}, 0);
    {{Py_XINCREF(py_{name});}}
    """
    return pre + c_extract


def get_c_cleanup(fgraph, r, name, sub):
    """
    Wrapper around c_cleanup that decrefs py_name.

    """
    post = f"""
    {{Py_XDECREF(py_{name});}}
    """
    return r.type.c_cleanup(name, sub) + post


def get_c_sync(fgraph, r, name, sub):
    """
    Wrapper around c_sync that syncs py_name with storage.

    """
    failure_var = sub["failure_var"]
    sync = r.type.c_sync(name, sub)
    return f"""
    if (!{failure_var}) {{
      {sync}
      PyObject* old = PyList_GET_ITEM(storage_{name}, 0);
      {{Py_XINCREF(py_{name});}}
      PyList_SET_ITEM(storage_{name}, 0, py_{name});
      {{Py_XDECREF(old);}}
    }}
    """


def apply_policy(fgraph, policy, r, name, sub):
    """
    Apply the list of policies to name.r,sub

    Parameters
    ----------
    policy
        List of functions that map a L{Variable} to a string,
        or a single such function.
    r: L{Variable}

    Returns
    -------
    object
        C{policy[0](r) + policy[1](r) + ...}.

    """
    if isinstance(policy, list | tuple):
        ret = ""
        for sub_policy in policy:
            ret += sub_policy(fgraph, r, name, sub)
        return ret
    return policy(fgraph, r, name, sub)


def struct_variable_codeblocks(fgraph, variable, policies, id, symbol_table, sub):
    """
    Update "sub" dict and create two codeblocks with different failure modes

    Parameters
    ----------
    variable : a Variable
    policies : a pair of tuples
        (declare_policy, behavior_policy, cleanup_policy) -- at construction.
        (declare_policy, behavior_policy, cleanup_policy)) -- at execution.
        The first list will produce an element of the 'struct_builders' argument
        in struct_gen. The second list will produce an element of the 'blocks'
        argument in struct_gen.
    id
        The id assigned to this variable's task in the computation.
    symbol_table
        A dict that maps variables to variable names. It is not read by this
        function but a variable name for the variable is computed and added to
        the table.
    sub
        Dictionary for use by L{CodeBlock}.

    """

    name = f"V{id}"
    if variable not in symbol_table:
        symbol_table[variable] = name
    sub = dict(sub)
    #    sub['name'] = name
    sub["id"] = id
    sub["fail"] = failure_code_init(sub)
    sub["py_ptr"] = f"py_{name}"
    sub["stor_ptr"] = f"storage_{name}"
    # struct_declare, struct_behavior, struct_cleanup, sub)
    struct_builder = CodeBlock(
        *[apply_policy(fgraph, policy, variable, name, sub) for policy in policies[0]]
        + [sub]
    )
    sub["id"] = id + 1
    sub["fail"] = failure_code(sub)
    sub["py_ptr"] = f"py_{name}"
    sub["stor_ptr"] = f"storage_{name}"
    # run_declare, run_behavior, run_cleanup, sub)
    block = CodeBlock(
        *[apply_policy(fgraph, policy, variable, name, sub) for policy in policies[1]]
        + [sub]
    )

    return struct_builder, block


class CLinker(Linker):
    r"""Generates and compiles C code for a :class:`FunctionGraph`.

    If a Variable is in no_recycling, CLinker will clear the output storage
    associated to it during the computation (to avoid reusing it).

    """

    def __init__(self, schedule=None):
        self.fgraph = None
        super().__init__(scheduler=schedule)

    def accept(
        self, fgraph: "FunctionGraph", no_recycling=None, profile=None
    ) -> "CLinker":
        r"""Associate this `Linker` with `fgraph`.

        The `no_recycling` argument can contain a list of `Variable`\s that
        belong to `fgraph`.
        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            # A linker can be tied to only one FunctionGraph.
            return type(self)(self.schedule).accept(fgraph, no_recycling, profile)
        self.fgraph = fgraph
        self.fetch_variables()
        self.no_recycling = no_recycling
        return self

    def fetch_variables(self):
        """Fills the inputs, outputs, variables, orphans, temps and node_order fields."""
        fgraph = self.fgraph
        self.inputs = fgraph.inputs
        self.outputs = fgraph.outputs

        self.node_order = self.schedule(fgraph)

        # list(fgraph.variables)
        # We need to include the unused inputs in our variables,
        # otherwise we can't pass them to the module.
        self.variables = [var for var in self.inputs if not len(fgraph.clients[var])]
        self.variables += list(vars_between(self.inputs, self.outputs))

        # This adds a hidden input which is the params for each node
        # that needs it
        self.node_params = dict()
        for node in self.node_order:
            if not isinstance(node.op, CLinkerOp):
                continue
            try:
                params = node.op.get_params(node)
            except MethodNotDefined:
                params = NoParams
            if params is not NoParams:
                # try to avoid creating more than one variable for the
                # same params.
                if params in self.node_params:
                    var = self.node_params[params]
                    assert var.type == node.params_type
                    fgraph.clients[var].append((node, "params"))
                else:
                    var = Constant(node.params_type, params)
                    fgraph.clients[var] = [(node, "params")]
                    self.node_params[params] = var
                    self.variables.append(var)

        # The orphans field is listified to ensure a consistent order.
        # list(fgraph.orphans.difference(self.outputs))
        self.orphans = [
            r
            for r in self.variables
            if isinstance(r, AtomicVariable) and r not in self.inputs
        ]
        # C type constants (pytensor.scalar.ScalarType). They don't request an object
        self.consts = []
        # Move c type from orphans (pytensor.scalar.ScalarType) to self.consts
        for variable in self.orphans:
            if (
                isinstance(variable, Constant)
                and isinstance(variable.type, CLinkerType)
                # This apparently checks--in a somewhat "dynamic" way--whether
                # or not a literal value is available (in C).
                and variable.type.c_literal(variable.data)
            ):
                self.consts.append(variable)
                self.orphans.remove(variable)

        self.temps = list(
            set(self.variables)
            .difference(self.inputs)
            .difference(self.outputs)
            .difference(self.orphans)
        )

    def code_gen(self):
        """Construct and populate a C ``struct`` for the generated code.

        Generates code for a ``struct`` instance that does the computation of the `FunctionGraph` and
        stores it in the ``struct_code`` field of the instance.

        If :attr:`CLinker.reuse_storage` is ``True``, outputs and temporaries
        will be stored in the ``struct`` so they can be reused each time the
        generated code is called, which means that the output of a call will be
        invalidated by the next. If the value is ``False``, that problem is
        avoided.

        This method caches its computations.

        """

        if getattr(self, "struct_code", False):
            return self.struct_code

        no_recycling = self.no_recycling

        c_support_code_apply = []
        c_init_code_apply = []

        symbol = {}

        # (init_)tasks contains a list of pairs (Op/Variable, task_name)
        # e.g. (x, 'get') or (x+y, 'code')
        init_tasks = []
        tasks = []

        # (init_)blocks contain CodeBlock instances. There is a direct
        # correspondence with (init_)tasks.
        init_blocks = []
        blocks = []

        failure_var = "__failure"
        id = 1

        for variable in self.variables:
            if not isinstance(variable.type, CLinkerType):
                raise NotImplementedError(f"Type of {variable} cannot produce C code")

            sub = dict(failure_var=failure_var)

            # it might be possible to inline constant variables as C literals
            # policy = [[what to declare in the struct,
            #            what to do at construction,
            #            what to do at destruction],
            #           [what to declare in each run,
            #            what to do at the beginning of each run,
            #            what to do at the end of each run]]
            if variable in self.consts:
                symbol[variable] = "(" + variable.type.c_literal(variable.data) + ")"
                continue
            elif variable in self.inputs:
                # We need to extract the new inputs at each run
                # they do not need to be relayed to Python, so we don't sync.
                # If the variable is both an input and an output, there is
                # no need to synchronize either, it is already up-to-date.
                policy = [
                    [get_nothing, get_nothing, get_nothing],
                    [get_c_declare, get_c_extract, get_c_cleanup],
                ]
            elif variable in self.orphans:
                if not isinstance(variable, AtomicVariable):
                    raise TypeError(
                        "All orphans to CLinker must be Constant instances.  "
                        f"Got {variable}"
                    )
                # orphans are not inputs so we'll just get fetch them
                # when we initialize the struct and assume they stay
                # the same
                policy = [
                    [get_c_declare, get_c_extract, get_c_cleanup],
                    [get_nothing, get_nothing, get_nothing],
                ]
            elif variable in self.temps:
                # temps don't need to be extracted from Python, so we
                # call c_init rather than c_extract they do not need
                # to be relayed to Python, so we don't sync
                if variable.type.c_is_simple() or variable in no_recycling:
                    policy = [
                        [get_nothing, get_nothing, get_nothing],
                        [get_c_declare, get_c_init, get_c_cleanup],
                    ]
                else:
                    # it is useful for complex temps to reuse storage
                    # at each run, so we only clean up in the
                    # destructor
                    policy = [
                        [get_c_declare, get_c_init, get_c_cleanup],
                        [get_nothing, get_nothing, get_nothing],
                    ]
            elif variable in self.outputs:
                if variable.type.c_is_simple() or variable in no_recycling:
                    # Do not extract output from Python
                    policy = [
                        [get_nothing, get_nothing, get_nothing],
                        [get_c_declare, get_c_init, (get_c_sync, get_c_cleanup)],
                    ]
                else:
                    # We try to use the output that is pre-allocated.
                    # The linker will usually just reuse the storage
                    # from last run, but in the first execution,
                    # it will be None.
                    # We clean-up at each run to enable garbage collection
                    # in the Linker.
                    policy = [
                        [get_nothing, get_nothing, get_nothing],
                        [get_c_declare, get_c_extract_out, (get_c_sync, get_c_cleanup)],
                    ]
            else:
                raise Exception(
                    "this shouldn't be possible, please report this exception"
                )

            builder, block = struct_variable_codeblocks(
                self.fgraph, variable, policy, id, symbol, sub
            )

            # each Variable generates two CodeBlocks, one to
            # declare/initialize/destroy struct variables and the
            # other to declare/extract/cleanup each time the function
            # is run.
            # Typically, only one of the two actually does anything
            # (see all the possible combinations above)

            init_tasks.append((variable, "init", id))
            init_blocks.append(builder)

            tasks.append((variable, "get", id + 1))
            blocks.append(block)

            id += 2

        for node_num, node in enumerate(self.node_order):
            op = node.op

            if not isinstance(op, CLinkerOp):
                raise NotImplementedError(f"{op} cannot produce C code")

            sub = dict(failure_var=failure_var)

            try:
                params = op.get_params(node)
            except MethodNotDefined:
                params = NoParams
            if params is not NoParams:
                params_var = symbol[self.node_params[params]]

            # The placeholder will be replaced by a hash of the entire
            # code (module + support code) in DynamicModule.code.
            # This ensures that, when defining functions in support code,
            # we cannot have two different functions, in different modules,
            # that have the same name.
            name = f"node_<<<<HASH_PLACEHOLDER>>>>_{node_num}"
            isyms = [symbol[r] for r in node.inputs]
            osyms = [symbol[r] for r in node.outputs]

            # Make the CodeBlock for c_code
            sub["id"] = id
            sub["fail"] = failure_code(sub)
            if params is not NoParams:
                sub["params"] = params_var

            sub_struct = dict()
            sub_struct["id"] = id + 1
            sub_struct["fail"] = failure_code_init(sub)
            if params is not NoParams:
                # Since params inputs are always constants they are
                # guaranteed to be available in the struct init code.
                sub_struct["params"] = params_var

            c_support_code_apply.append(op.c_support_code_apply(node, name))
            c_init_code_apply.append(op.c_init_code_apply(node, name))
            struct_init = op.c_init_code_struct(node, name, sub_struct)

            struct_support = op.c_support_code_struct(node, name)
            struct_cleanup = op.c_cleanup_code_struct(node, name)

            behavior = op.c_code(node, name, isyms, osyms, sub)

            assert isinstance(
                behavior, str
            ), f"{node.op} didn't return a string for c_code"
            # To help understand what is following. It help read the c code.
            # This prevent different op that generate the same c code
            # to be merged, I suppose this won't happen...
            behavior = "// Op class " + node.op.__class__.__name__ + "\n" + behavior

            cleanup = op.c_code_cleanup(node, name, isyms, osyms, sub)

            _logger.info(f"compiling un-versioned Apply {node}")

            blocks.append(CodeBlock("", behavior, cleanup, sub))
            tasks.append((node, "code", id))
            id += 1

            init_blocks.append(
                CodeBlock(struct_support, struct_init, struct_cleanup, {"id": id})
            )
            init_tasks.append((node, "init", id))
            id += 1

        # List of arg names for use in struct_gen. Note the call to
        # uniq: duplicate inputs must only be passed once because they
        # are mapped to the same name.  Duplicates are defined by (a
        # is b), rather than (a==b) since Constant instances can
        # compare equal to equivalent Constant instances.
        args = []
        args += [
            f"storage_{symbol[variable]}"
            for variable in uniq(self.inputs + self.outputs + self.orphans)
        ]

        # <<<<HASH_PLACEHOLDER>>>> will be replaced by a hash of the whole
        # code in the file, including support code, in DynamicModule.code.
        struct_name = f"__struct_compiled_op_{'<<<<HASH_PLACEHOLDER>>>>'}"
        struct_code = struct_gen(
            args, init_blocks, blocks, dict(failure_var=failure_var, name=struct_name)
        )

        self.struct_code = struct_code
        self.struct_name = struct_name
        self.args = args
        self.r2symbol = symbol
        self.init_blocks = init_blocks
        self.init_tasks = init_tasks
        self.blocks = blocks
        self.tasks = tasks
        all_info = self.inputs + self.outputs + self.orphans
        self.c_support_code_apply = c_support_code_apply
        self.c_init_code_apply = c_init_code_apply

        if (self.init_tasks, self.tasks) != self.get_init_tasks():
            print("init_tasks\n", self.init_tasks, file=sys.stderr)  # noqa: T201
            print(self.get_init_tasks()[0], file=sys.stderr)  # noqa: T201
            print("tasks\n", self.tasks, file=sys.stderr)  # noqa: T201
            print(self.get_init_tasks()[1], file=sys.stderr)  # noqa: T201
            assert (self.init_tasks, self.tasks) == self.get_init_tasks()

        # List of indices that should be ignored when passing the arguments
        # (basically, everything that the previous call to uniq eliminated)
        self.dupidx = [
            i
            for i, x in enumerate(all_info)
            if all_info.count(x) > 1 and all_info.index(x) != i
        ]
        return self.struct_code

    def support_code(self):
        """
        Returns a list of support code strings that are needed by
        one or more Variables or Ops.
        The support code from Variables is added before the support code from Ops.This might contain duplicates.
        """
        ret = []
        if config.cmodule__debug:
            ret.append(
                """
            #ifndef DEBUG
            #define DEBUG
            #endif
            """
            )
        # generic support code
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            support_code = x.c_support_code()
            if isinstance(support_code, list):
                ret.extend(support_code)
            else:
                ret.append(support_code)
        return ret

    def compile_args(self):
        """
        Returns a list of compile args that are needed by one
        or more Variables or Ops.

        This might contain duplicates.

        """
        ret = ["-O3"]
        # this is the param the -ffast-math activate. I put the explicitly as
        # FillMissing must disable some of them. Putting -ffast-math would
        # make it disable all other parameter at the same time.
        ret += [
            "-fno-math-errno",
            # "-funsafe-math-optimizations",
            # "-fno-signaling-nans",
            # "-fcx-limited-range",
            # "-fno-rounding-math",
            # "-ffinite-math-only",
            # the current code generate label event if they are not used.
            # Could use gcc attribute for those label only
            "-Wno-unused-label",
            "-Wno-unused-variable",  # idem as the precedent
            "-Wno-write-strings",  # generated by our code generator...
        ]

        c_compiler = self.c_compiler()

        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if isinstance(x, CLinkerObject):
                ret += x.c_compile_args(c_compiler=c_compiler)

        ret = uniq(ret)  # to remove duplicate
        # The args set by the compiler include the user flags. We do not want
        # to reorder them
        ret += c_compiler.compile_args()
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if isinstance(x, CLinkerObject):
                no_comp = x.c_no_compile_args(c_compiler=c_compiler)

                for i in no_comp:
                    try:
                        ret.remove(i)
                    except ValueError:
                        pass  # in case the value is not there
        return ret

    def headers(self):
        """
        Returns a list of headers that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.

        """
        ret = []
        c_compiler = self.c_compiler()
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if isinstance(x, CLinkerObject):
                ret += x.c_headers(c_compiler=c_compiler)
        return uniq(ret)

    def init_code(self):
        """
        Return a list of code snippets that have to be inserted
        in the module initialization code.

        The return value will not contain duplicates.

        """
        ret = []
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if isinstance(x, CLinkerObject):
                ret += x.c_init_code()
        return uniq(ret)

    def c_compiler(self):
        c_compiler = None
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            # FIXME: Why would a `Type` have a `c_compiler` field?!
            if hasattr(x, "c_compiler"):
                x_compiler = x.c_compiler()
            else:
                continue

            if c_compiler is None:
                c_compiler = x_compiler
            else:
                if x_compiler and (x_compiler != c_compiler):
                    raise Exception(
                        "Nodes have requested specific different compilers",
                        (c_compiler, x_compiler),
                    )
        if c_compiler is None:
            return GCC_compiler
        else:
            return c_compiler

    def header_dirs(self):
        """
        Returns a list of lib directories that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.

        """
        ret = []
        c_compiler = self.c_compiler()
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if isinstance(x, CLinkerObject):
                ret += x.c_header_dirs(c_compiler=c_compiler)
        # filter out empty strings/None
        return [r for r in uniq(ret) if r]

    def libraries(self):
        """
        Returns a list of libraries that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.

        """
        ret = []
        c_compiler = self.c_compiler()
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if isinstance(x, CLinkerObject):
                ret += x.c_libraries(c_compiler=c_compiler)
        return uniq(ret)

    def lib_dirs(self):
        """
        Returns a list of lib directories that are needed by one
        or more Types or Ops.

        The return value will not contain duplicates.

        """
        ret = []
        c_compiler = self.c_compiler()
        for x in [y.type for y in self.variables] + [y.op for y in self.node_order]:
            if isinstance(x, CLinkerObject):
                ret += x.c_lib_dirs(c_compiler=c_compiler)
        # filter out empty strings/None
        return [r for r in uniq(ret) if r]

    def __compile__(
        self,
        input_storage=None,
        output_storage=None,
        storage_map=None,
        cache: Optional["ModuleCache"] = None,
    ):
        """Compile `self.fgraph`.

        Parameters
        ----------
        input_storage: list or None
            List of lists of length 1. In order to use the thunk returned
            by this method, the inputs must be put in that storage.
            If None, storage will be allocated.
        output_storage: list of lists of length 1
            The thunk returned by this method will put the variables of the
            computation in these lists. If None, storage will be allocated.

        Returns
        -------
        object
            Thunk, input_storage, output_storage, error_storage.

        """
        error_storage = [None, None, None]
        if input_storage is None:
            input_storage = tuple([None] for variable in self.inputs)
        if output_storage is None:
            map = {}
            output_storage = []
            # Initialize the map with the inputs, as some outputs may
            # be inputs as well.
            for i, variable in enumerate(self.inputs):
                map[variable] = input_storage[i]
            for variable in self.outputs:
                if variable not in map:
                    map[variable] = [None]
                output_storage.append(map[variable])
        input_storage = tuple(input_storage)
        output_storage = tuple(output_storage)
        thunk, module = self.cthunk_factory(
            error_storage,
            input_storage,
            output_storage,
            storage_map,
            cache,
        )
        return (
            thunk,
            module,
            [
                Container(input, storage)
                for input, storage in zip(
                    self.fgraph.inputs, input_storage, strict=True
                )
            ],
            [
                Container(output, storage, readonly=True)
                for output, storage in zip(
                    self.fgraph.outputs, output_storage, strict=True
                )
            ],
            error_storage,
        )

    def get_init_tasks(self):
        vars = [v for v in self.variables if v not in self.consts]
        id = 1
        init_tasks = [(v, "init", id + 2 * i) for i, v in enumerate(vars)]
        tasks = [(v, "get", id + 2 * i + 1) for i, v in enumerate(vars)]

        id += 2 * len(vars)
        tasks.extend(
            (node, "code", id + 2 * i) for i, node in enumerate(self.node_order)
        )
        init_tasks.extend(
            (node, "init", id + 2 * i + 1) for i, node in enumerate(self.node_order)
        )
        return init_tasks, tasks

    def make_thunk(
        self,
        input_storage=None,
        output_storage=None,
        storage_map=None,
        cache: Optional["ModuleCache"] = None,
        **kwargs,
    ):
        """Compile this linker's `self.fgraph` and return a function that performs the computations.

        The return values can be used as follows:

        .. code-block::

            f, istor, ostor = clinker.make_thunk()
            istor[0].data = first_input
            istor[1].data = second_input
            f()
            first_output = ostor[0].data


        Parameters
        ----------
        input_storage: list or None
            List of lists of length 1. In order to use
            the thunk returned by `CLinker.__compile__`, the inputs must be put in
            that storage. If None, storage will be allocated.
        output_storage: list of lists of length 1.
            The thunk returned by `CLinker.__compile__` will put the variables
            of the computation in these lists. If None, storage will
            be allocated.
        storage_map: dict that map variables to storages.
            This is used when you need to customize the storage of
            this thunk
        cache
            A cache in which to store the compilation results.

        Returns
        -------
        thunk, input_storage, output_storage

        """
        init_tasks, tasks = self.get_init_tasks()
        cthunk, module, in_storage, out_storage, error_storage = self.__compile__(
            input_storage, output_storage, storage_map, cache
        )

        res = _CThunk(cthunk, init_tasks, tasks, error_storage, module)
        res.nodes = self.node_order
        return res, in_storage, out_storage

    def cmodule_key(self):
        """
        Return a complete hashable signature of the module we compiled.

        This function must have the property that no two programs that
        compute different things yield the same key.

        The key returned by this function is of the form (version, signature)
        The signature has the following form:
        {{{
            'CLinker.cmodule_key', compilation args, libraries,
            header_dirs, numpy ABI version, config hash,
            (op0, input_signature0, output_signature0),
            (op1, input_signature1, output_signature1),
            ...
            (opK, input_signatureK, output_signatureK),
        }}}

        Note that config hash now uses sha256, and not md5.

        The signature is a tuple, some elements of which are sub-tuples.

        The outer tuple has a brief header, containing the compilation options
        passed to the compiler, the libraries to link against, a sha256 hash
        of pytensor.config (for all config options where "in_c_key" is True).
        It is followed by elements for every node in the topological ordering
        of `self.fgraph`.

        Input Signature
        ---------------

        Each input signature is a tuple with an element for each input
        to the corresponding Apply node. Each element identifies the
        type of the node input, and the nature of that input in the
        graph.

        The nature of a typical variable is encoded by integer pairs
        ``((a,b),c)``:
        ``a`` is the topological position of the input's owner
              (-1 for graph inputs),
        ``b`` is the index of the variable in the owner's output list.
        ``c`` is a flag indicating whether the variable is in the
              no_recycling set.

        If a variable is also a graph output, then its position in the
        outputs list is also bundled with this tuple (after the b).

        The nature of a Constant instance is defined as its signature,
        together with two integers: the topological position of the
        first Apply using that Constant instance, and the lowest index
        into that Apply's inputs that refers to that Constant.  (These
        two integers are a surrogate for the id() of the Constant.
        The integers are important because merge-able constants have
        the same signature, but require separate containers in C
        code.)  The membership in no_recycling is also included in the
        signature.

        Output Signature
        ----------------

        The outputs of a node are entirely determined by the node's Op
        and the nature of the inputs, but the set of outputs that may
        be re-used by the computation (the elements of
        self.no_recycling) can affect the code that is generated.

        The format of each Op's output signature is a (version, no_recycle)
        pair, where version is incremented if codegen() changes how it
        handles the outputs, and no_recycle is simply a list of
        booleans, indicating whether each output is in the
        no_recycling set. Older versions of compiled modules only have the
        no_recycle list.

        """
        return self.cmodule_key_(
            self.fgraph,
            self.no_recycling,
            compile_args=self.compile_args(),
            libraries=self.libraries(),
            header_dirs=self.header_dirs(),
            c_compiler=self.c_compiler(),
        )

    def cmodule_key_variables(
        self,
        inputs,
        outputs,
        no_recycling,
        compile_args=None,
        libraries=None,
        header_dirs=None,
        insert_config_hash=True,
        c_compiler=None,
    ):
        # Assemble a dummy fgraph using the provided inputs and outputs. It is
        # only used to compute the cmodule key so it only need to expose an
        # `inputs` and an `outputs` attribute as well as a toposort() method
        # which returns a deterministic result.
        class FakeFunctionGraph:
            def __init__(self, inputs, outputs):
                self.inputs = inputs
                self.outputs = outputs
                self.clients = defaultdict(list)

            def toposort(self):
                # Calling io_toposort() here is fine because the results will
                # only be used to compute the cmodule key which requires that
                # the result of the toposort be deterministic. The ordering
                # doesn't need to include information about inplace operations
                # because that information will be included explicitly in
                # cmodule_key_().
                return io_toposort(self.inputs, self.outputs)

        fgraph = FakeFunctionGraph(inputs, outputs)
        return self.cmodule_key_(
            fgraph,
            no_recycling,
            compile_args,
            libraries,
            header_dirs,
            insert_config_hash,
            c_compiler,
        )

    def cmodule_key_(
        self,
        fgraph,
        no_recycling,
        compile_args=None,
        libraries=None,
        header_dirs=None,
        insert_config_hash=True,
        c_compiler=None,
    ):
        """
        Do the actual computation of cmodule_key in a static method
        to allow it to be reused in scalar.Composite.__eq__.

        """
        if compile_args is None:
            compile_args = []
        if libraries is None:
            libraries = []
        if header_dirs is None:
            header_dirs = []
        order = self.schedule(fgraph)
        # set of variables that have been computed by nodes we have
        # seen 'so far' in the loop below
        fgraph_computed_set = set()
        fgraph_inputs_dict = {i: (-1, pos) for pos, i in enumerate(fgraph.inputs)}
        constant_ids = dict()
        op_pos = {}  # Apply -> topological position

        # First we put the header, compile_args, library names and config hash
        # into the signature.
        sig = ["CLinker.cmodule_key"]  # will be cast to tuple on return
        if compile_args is not None:
            # We must sort it as the order from a set is not guaranteed.
            # In  particular, 2 sets with the same content can give different
            # order depending on the order you put data in it.
            # Sets are used to remove duplicate elements.
            args = sorted(compile_args)
            args = tuple(args)
            sig.append(args)
        if libraries is not None:
            # see comments for compile_args
            args = sorted(libraries)
            args = tuple(args)
            sig.append(args)

        if header_dirs is not None:
            args = sorted(header_dirs)
            args = tuple(args)
            sig.append(args)

        # We must always add the numpy ABI version here as
        # DynamicModule always add the include <numpy/arrayobject.h>
        sig.append(f"NPY_ABI_VERSION=0x{ndarray_c_version:X}")
        if c_compiler:
            sig.append("c_compiler_str=" + c_compiler.version_str())

        # IMPORTANT: The 'md5' prefix is used to isolate the compilation
        # parameters from the rest of the key. If you want to add more key
        # elements, they should be before this md5 hash if and only if they
        # can lead to a different compiled file with the same source code.

        # NOTE: config md5 is not using md5 hash, but sha256 instead. Function
        # string instances of md5 will be updated at a later release.
        if insert_config_hash:
            sig.append("md5:" + config.get_config_hash())
        else:
            sig.append("md5: <omitted>")

        error_on_play = [False]

        def in_sig(i, topological_pos, i_idx):
            # assert that every input to every node is one of'
            # - an fgraph input
            # - an output from a node in the FunctionGraph
            # - a Constant

            # It is important that a variable (i)
            # yield a 'position' that reflects its role in code_gen()
            if isinstance(i, AtomicVariable):  # orphans
                if id(i) not in constant_ids:
                    isig = (i.signature(), topological_pos, i_idx)
                    # If the PyTensor constant provides a strong hash
                    # (no collision for transpose, 2, 1, 0, -1, -2,
                    # 2 element swapped...) we put this hash in the signature
                    # instead of the value. This makes the key file much
                    # smaller for big constant arrays. Before this, we saw key
                    # files up to 80M.
                    if hasattr(isig[0], "pytensor_hash"):
                        isig = (isig[0].pytensor_hash(), topological_pos, i_idx)
                    try:
                        hash(isig)
                    except Exception:
                        # generic constants don't have a hashable signature
                        error_on_play[0] = True
                        return None
                    constant_ids[id(i)] = isig
                else:
                    isig = constant_ids[id(i)]
                # print 'SIGNATURE', i.signature()
                # return i.signature()
            elif i in fgraph_inputs_dict:  # inputs
                isig = fgraph_inputs_dict[i]
            else:
                if i.owner is None:
                    assert all(all(out is not None for out in o.outputs) for o in order)
                    assert all(input.owner is None for input in fgraph.inputs)
                    raise Exception(
                        f"Owner of {i} (clients {fgraph.clients.get(i)}) is None"
                    )

                if i in fgraph.outputs:
                    isig = (
                        op_pos[i.owner],  # outputs
                        i.owner.outputs.index(i),
                        fgraph.outputs.index(i),
                    )
                else:
                    isig = (op_pos[i.owner], i.owner.outputs.index(i))  # temps
            return (isig, i in no_recycling)

        version = []
        for node_pos, node in enumerate(order):
            if hasattr(node.op, "c_code_cache_version_apply"):
                version.append(node.op.c_code_cache_version_apply(node))

            props = getattr(node.op, "__props__", None)

            if props:
                version.append(props)

            version.extend(
                i.type.c_code_cache_version()
                for i in node.inputs
                if isinstance(i.type, CLinkerObject)
            )
            version.extend(
                o.type.c_code_cache_version()
                for o in node.outputs
                if isinstance(o.type, CLinkerObject)
            )

            # add the signature for this node
            sig.append(
                (
                    node.op,
                    tuple(
                        (i.type, in_sig(i, node_pos, ipos))
                        for ipos, i in enumerate(node.inputs)
                    ),
                    (
                        1,  # Increment if cmodule change its handling of outputs
                        tuple(o in no_recycling for o in node.outputs),
                    ),
                )
            )

            if error_on_play[0]:
                # if one of the signatures is not hashable
                # then bypass the cache mechanism and
                # compile fresh every time
                return None

            op_pos[node] = node_pos
            fgraph_computed_set.update(node.outputs)

        # Add not used input in the key
        # If inputs don't define a 'clients' attribute (as is the case if
        # fgraph is not a real FunctionGraph but a FakeFunctionGraph, a
        # lightweight class designed to imitate FunctionGraph), pretend they
        # have none. This if fine because the goal is only to have all of the
        # graph's information used to compute the key. If we mistakenly
        # pretend that inputs with clients don't have any, were are only using
        # those inputs more than once to compute the key.
        sig.extend(
            (var.type, in_sig(var, -1, ipos))
            for ipos, var in enumerate(fgraph.inputs)
            if not len(fgraph.clients[var])
        )

        # crystalize the signature and version
        sig = tuple(sig)
        version = tuple(version)
        for v in version:
            if not v:
                # one of the ops or types here is unversioned,
                # so this fgraph is entirely unversioned
                return ((), sig)
        return version, sig

    def get_src_code(self):
        mod = self.get_dynamic_module()
        return mod.code()

    def compile_cmodule(self, location=None):
        """
        This compiles the source code for this linker and returns a
        loaded module.

        """
        if location is None:
            location = dlimport_workdir(config.compiledir)
        mod = self.get_dynamic_module()
        c_compiler = self.c_compiler()
        libs = self.libraries()
        preargs = self.compile_args()
        # We want to compute the code without the lock
        src_code = mod.code()
        with lock_ctx():
            try:
                _logger.debug(f"LOCATION {location}")
                module = c_compiler.compile_str(
                    module_name=mod.code_hash,
                    src_code=src_code,
                    location=location,
                    include_dirs=self.header_dirs(),
                    lib_dirs=self.lib_dirs(),
                    libs=libs,
                    preargs=preargs,
                )
            except Exception as e:
                e.args += (str(self.fgraph),)
                raise
        return module

    def get_dynamic_module(self):
        """
        Return a cmodule.DynamicModule instance full of the code for our fgraph.

        This method is cached on the first call so it can be called
        multiple times without penalty.

        """
        if not hasattr(self, "_mod"):
            self.code_gen()

            mod = DynamicModule()

            # The code of instantiate
            # the 1 is for error_storage
            code = self.instantiate_code(1 + len(self.args))
            instantiate = ExtFunction("instantiate", code, method=METH_VARARGS)
            # ['error_storage'] + argnames,
            # local_dict = d,
            # global_dict = {})

            # Static methods that can run and destroy the struct built by
            # instantiate.
            static = f"""
        static int {self.struct_name}_executor({self.struct_name} *self) {{
            return self->run();
        }}

        static void {self.struct_name}_destructor(PyObject *capsule) {{
            {self.struct_name} *self = ({self.struct_name} *)PyCapsule_GetContext(capsule);
            delete self;
        }}
    """

            # We add all the support code, compile args, headers and libs we need.
            for support_code in self.support_code() + self.c_support_code_apply:
                mod.add_support_code(support_code)
            mod.add_support_code(self.struct_code)
            mod.add_support_code(static)
            mod.add_function(instantiate)
            for header in self.headers():
                mod.add_include(header)
            for init_code_block in self.init_code() + self.c_init_code_apply:
                mod.add_init_code(init_code_block)
            self._mod = mod
        return self._mod

    def cthunk_factory(
        self,
        error_storage,
        in_storage,
        out_storage,
        storage_map=None,
        cache: Optional["ModuleCache"] = None,
    ):
        """
        Returns a thunk that points to an instance of a C struct that
        can carry on the computation of this linker's fgraph

        Parameters:
        ----------
        error_storage -> list of length 3
        in_storage -> list of lists of length 1, one per input
        out_storage -> list of lists of length 1, one per output

        Returns a thunk that points to an instance of a C struct that
        can carry on the computation of this linker's fgraph. That thunk,
        when executed, will fetch its inputs from in_storage, put its
        outputs in out_storage and if an error occurs will put the
        type, value and traceback of the exception in error_storage.
        """
        try:
            key = self.cmodule_key()
        except KeyError:
            key = None

        if key is None:
            # If we can't get a key, then forget the cache mechanism.
            module = self.compile_cmodule()
        else:
            # Set compute_map as None as clinker do not support lazy evaluation
            for node in self.node_order:
                node.op.prepare_node(node, storage_map, None, "c")
            if cache is None:
                cache = get_module_cache()
            module = cache.module_from_key(key=key, lnk=self)

        vars = self.inputs + self.outputs + self.orphans
        # List of indices that should be ignored when passing the arguments
        # (basically, everything that the previous call to uniq eliminated)
        dupidx = [
            i for i, x in enumerate(vars) if vars.count(x) > 1 and vars.index(x) != i
        ]

        out_storage = [
            x for i, x in enumerate(out_storage) if (i + len(in_storage)) not in dupidx
        ]
        in_storage = [x for i, x in enumerate(in_storage) if i not in dupidx]
        if storage_map is None:
            orphd = [
                [orphan.data] if isinstance(orphan, Constant) else []
                for orphan in self.orphans
            ]
        else:
            orphd = [storage_map[orphan] for orphan in self.orphans]

        ret = module.instantiate(error_storage, *(in_storage + out_storage + orphd))
        return ret, module

    def instantiate_code(self, n_args):
        code = StringIO()
        struct_name = self.struct_name
        print(
            "static PyObject * instantiate(PyObject * self, PyObject *argtuple) {",
            file=code,
        )
        print("  assert(PyTuple_Check(argtuple));", file=code)
        print(f"  if ({n_args} != PyTuple_Size(argtuple)){{ ", file=code)
        print(
            f'     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected {n_args}, got %%i", (int)PyTuple_Size(argtuple));',
            file=code,
        )
        print("     return NULL;", file=code)
        print("  }", file=code)
        print(
            f"  {struct_name}* struct_ptr = new {struct_name}();",
            file=code,
        )
        print(
            "  if (struct_ptr->init(",
            ",".join(f"PyTuple_GET_ITEM(argtuple, {n})" for n in range(n_args)),
            ") != 0) {",
            file=code,
        )
        print("    delete struct_ptr;", file=code)
        print("    return NULL;", file=code)
        print("  }", file=code)
        print(
            f"""\
    PyObject* thunk = PyCapsule_New((void*)(&{struct_name}_executor), NULL, {struct_name}_destructor);
    if (thunk != NULL && PyCapsule_SetContext(thunk, struct_ptr) != 0) {{
        PyErr_Clear();
        Py_DECREF(thunk);
        thunk = NULL;
    }}
""",
            file=code,
        )
        print("  return thunk; }", file=code)
        return code.getvalue()


class _CThunk:
    """
    A thunk with a C implementation.

    Parameters
    ----------
    cthunk
        A CObject pointer that is used to run the thunk.
    init_tasks
        WRITEME
    tasks
        WRITEME
    error_storage
        WRITEME
    module
        The module that was used to compile this cthunk.
        Mostly only useful for tests.

    """

    def __init__(self, cthunk, init_tasks, tasks, error_storage, module):
        # Lazy import to avoid compilation when importing pytensor.
        from pytensor.link.c.cutils import run_cthunk

        self.run_cthunk = run_cthunk
        self.cthunk = cthunk
        self.init_tasks = init_tasks
        self.tasks = tasks
        self.error_storage = error_storage
        self.module = module
        self.nodes = None

    def find_task(self, failure_code):
        """
        Maps a failure code to the task that is associated to it.

        """
        failure_code -= 1
        n = len(self.init_tasks)
        # note that the failure code is distributed in two lists
        if failure_code < 2 * n:
            return [self.init_tasks, self.tasks][failure_code % 2][failure_code // 2]
        else:
            return self.tasks[failure_code - n]

    def __call__(self):
        failure = self.run_cthunk(self.cthunk)
        if failure:
            task, taskname, id = self.find_task(failure)
            try:
                trace = task.trace
            except AttributeError:
                trace = ()
            try:
                exc_type, _exc_value, exc_trace = self.error_storage
                if task in self.nodes:
                    self.position_of_error = self.nodes.index(task)
                # this can be used to retrieve the location the Op was declared
                exc_value = exc_type(_exc_value)
                exc_value.__thunk_trace__ = trace
            except Exception:
                print(  # noqa: T201
                    (
                        "ERROR retrieving error_storage."
                        "Was the error set in the c code?"
                    ),
                    end=" ",
                    file=sys.stderr,
                )
                print(self.error_storage, file=sys.stderr)  # noqa: T201
                raise
            raise exc_value.with_traceback(exc_trace)

    def __str__(self):
        return f"{type(self).__name__}({self.module})"


class OpWiseCLinker(LocalLinker):
    """
    Uses CLinker on the individual Ops that comprise an fgraph and loops
    over them in Python. The variable is slower than a compiled version of
    the whole fgraph, but saves on compilation time because small changes
    in the computation graph won't necessarily trigger any recompilation,
    only local changes in the Variables or Ops that are used.

    If fallback_on_perform is True, OpWiseCLinker will use an op's
    perform method if no C version can be generated.

    no_recycling can contain a list of Variables that belong to the fgraph.
    If a Variable is in no_recycling, CLinker will clear the output storage
    associated to it prior to computation (to avoid reusing it).

    Notes
    -----
    This is in a sense the 'default' linker for PyTensor. The
    overhead of using the OpWiseCLinker as compared with the CLinker
    is only noticeable for graphs of very small tensors (such as 20
    elements or less).

    """

    __cache__: dict = {}

    def __init__(
        self, fallback_on_perform=True, allow_gc=None, nice_errors=True, schedule=None
    ):
        if allow_gc is None:
            allow_gc = config.allow_gc
        self.fgraph = None
        self.fallback_on_perform = fallback_on_perform
        self.nice_errors = nice_errors
        super().__init__(allow_gc=allow_gc, scheduler=schedule)

    def accept(self, fgraph, no_recycling=None, profile=None):
        """
        Associate linker with fgraph
        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            # A linker can be tied to only one FunctionGraph.
            return type(self)(
                fallback_on_perform=self.fallback_on_perform,
                allow_gc=self.allow_gc,
                nice_errors=self.nice_errors,
                schedule=self.schedule,
            ).accept(fgraph, no_recycling, profile)
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        return self

    def make_all(
        self, profiler=None, input_storage=None, output_storage=None, storage_map=None
    ):
        fgraph = self.fgraph
        order = self.schedule(fgraph)
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = map_storage(
            fgraph, order, input_storage, output_storage, storage_map
        )
        if self.allow_gc:
            computed, last_user = gc_helper(order)
            post_thunk_old_storage = []
        else:
            post_thunk_old_storage = None

        compute_map = {}
        for k in storage_map:
            compute_map[k] = [k.owner is None]

        thunks = []
        for node in order:
            # make_thunk will try by default C code, otherwise
            # it fall back to python.
            thunks += [node.op.make_thunk(node, storage_map, compute_map, no_recycling)]
            thunks[-1].inputs = [storage_map[v] for v in node.inputs]
            thunks[-1].outputs = [storage_map[v] for v in node.outputs]

        for node in order:
            if self.allow_gc:
                post_thunk_old_storage.append(
                    [
                        storage_map[input]
                        for input in node.inputs
                        if (
                            (input in computed)
                            and (input not in fgraph.outputs)
                            and node == last_user[input]
                        )
                    ]
                )

        if no_recycling is True:
            no_recycling = list(storage_map.values())
            no_recycling = difference(no_recycling, input_storage)
        else:
            no_recycling = [
                storage_map[r] for r in no_recycling if r not in fgraph.inputs
            ]

        f = streamline(
            fgraph,
            thunks,
            order,
            post_thunk_old_storage,
            no_recycling=no_recycling,
            nice_errors=self.nice_errors,
        )

        f.allow_gc = self.allow_gc

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


def _default_checker(x, y):
    """
    Default checker for DualLinker. This checks that the
    variables contain the same data using ==.


    Parameters:
    ----------
    x,y
        the variables to compare data
    """
    if x[0] != y[0]:
        raise Exception("Output mismatch.", {"performlinker": x[0], "clinker": y[0]})


class DualLinker(Linker):
    """
    Runs the fgraph in parallel using PerformLinker and CLinker.

    The thunk/function produced by DualLinker uses PerformLinker as the
    "main" implementation: the inputs and outputs are fed to/taken from
    the Ops' perform. However, DualLinker also instantiates a copy of
    the fgraph on which it runs OpWiseCLinker. At each step, the variables
    of perform and of the C implementation are verified using a checker
    function.

    """

    def __init__(self, checker=_default_checker, schedule=None):
        """
        Initialize a DualLinker.

        The checker argument must be a function that takes two lists
        of length 1. The first one passed will contain the output
        computed by PerformLinker and the second one the output
        computed by OpWiseCLinker. The checker should compare the data
        fields of the two variables to see if they match. By default,
        DualLinker uses ==. A custom checker can be provided to
        compare up to a certain error tolerance.

        If a mismatch occurs, the checker should raise an exception to
        halt the computation. If it does not, the computation will
        carry on and errors will snowball. The checker can sidestep
        the problem by fiddling with the data, but it should be
        careful not to share data between the two outputs (or inplace
        operations that use them will interfere).

        no_recycling can contain a list of Variables that belong to the fgraph.
        If a Variable is in no_recycling, CLinker will clear the output storage
        associated to it during the computation (to avoid reusing it).

        """
        self.fgraph = None
        self.checker = checker
        super().__init__(scheduler=schedule)

    def accept(self, fgraph, no_recycling=None, profile=None):
        """
        Update/tie self with fgraph
        """
        if no_recycling is None:
            no_recycling = []
        if self.fgraph is not None and self.fgraph is not fgraph:
            return type(self)(self.checker, self.schedule).accept(
                fgraph, no_recycling, profile
            )
        self.fgraph = fgraph
        self.no_recycling = no_recycling
        return self

    def make_thunk(self, **kwargs):
        """
        Compiles this linker's fgraph and returns a function to perform the
        computations
        """
        fgraph = self.fgraph
        no_recycling = self.no_recycling

        _f, i1, o1, thunks1, order1 = (
            PerformLinker(schedule=self.schedule)
            .accept(fgraph, no_recycling=no_recycling)
            .make_all(**kwargs)
        )
        kwargs.pop("input_storage", None)
        _f, i2, o2, thunks2, order2 = (
            OpWiseCLinker(schedule=self.schedule)
            .accept(fgraph, no_recycling=no_recycling)
            .make_all(**kwargs)
        )

        def f():
            # zip strict not specified because we are in a hot loop
            for input1, input2 in zip(i1, i2):
                # Set the inputs to be the same in both branches.
                # The copy is necessary in order for inplace ops not to
                # interfere.
                input2.storage[0] = copy(input1.storage[0])
            for thunk1, thunk2, node1, node2 in zip(thunks1, thunks2, order1, order2):
                for output, storage in zip(node1.outputs, thunk1.outputs):
                    if output in no_recycling:
                        storage[0] = None
                for output, storage in zip(node2.outputs, thunk2.outputs):
                    if output in no_recycling:
                        storage[0] = None
                try:
                    thunk1()
                    thunk2()
                    for output1, output2 in zip(thunk1.outputs, thunk2.outputs):
                        self.checker(output1, output2)
                except Exception:
                    raise_with_op(fgraph, node1)

        return f, i1, o1


if config.cmodule__preload_cache:
    get_module_cache()
