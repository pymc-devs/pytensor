"""Symbolic Op for raising an exception."""

from textwrap import indent

import numpy as np

from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.replace import _vectorize_node
from pytensor.link.c.op import COp
from pytensor.link.c.params_type import ParamsType
from pytensor.link.c.type import Generic
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import DenseTensorType


class ExceptionType(Generic):
    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


exception_type = ExceptionType()


class CheckAndRaise(COp):
    """An `Op` that checks conditions and raises an exception if they fail.

    This `Op` returns its "value" argument if its condition arguments are all
    ``True``; otherwise, it raises a user-specified exception.

    """

    _f16_ok = True
    __props__ = ("msg", "exc_type")
    view_map = {0: [0]}

    check_input = False
    params_type = ParamsType(exc_type=exception_type)

    def __init__(self, exc_type, msg=""):
        if not issubclass(exc_type, Exception):
            raise ValueError("`exc_type` must be an Exception subclass")

        self.exc_type = exc_type
        self.msg = msg

    def __str__(self):
        name = self.__class__.__name__
        exc_name = self.exc_type.__name__
        if len(self.msg) > 30:
            msg = self.msg[:27] + "..."
        else:
            msg = self.msg
        return f"{name}{{raises={exc_name}, msg='{msg}'}}"

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if self.msg == other.msg and self.exc_type == other.exc_type:
            return True

        return False

    def __hash__(self):
        return hash((self.msg, self.exc_type))

    def make_node(self, value: Variable, *conds: Variable):
        """

        Parameters
        ==========
        value
            The value to return if `conds` all evaluate to ``True``; otherwise,
            `self.exc_type` is raised.
        conds
            The conditions to evaluate.
        """
        import pytensor.tensor as pt

        if not isinstance(value, Variable):
            value = pt.as_tensor_variable(value)

        conds = [
            pt.as_tensor_variable(c) if not isinstance(c, Variable) else c
            for c in conds
        ]

        assert all(c.type.ndim == 0 for c in conds)

        return Apply(
            self,
            [value, *conds],
            [value.type()],
        )

    def perform(self, node, inputs, outputs):
        (out,) = outputs
        val, *conds = inputs
        out[0] = val
        if not np.all(conds):
            raise self.exc_type(self.msg)

    def grad(self, input, output_gradients):
        return output_gradients + [DisconnectedType()()] * (len(input) - 1)

    def connection_pattern(self, node):
        return [[1]] + [[0]] * (len(node.inputs) - 1)

    def c_code(self, node, name, inames, onames, props):
        if not isinstance(node.inputs[0].type, DenseTensorType | ScalarType):
            raise NotImplementedError(
                f"CheckAndRaise c_code not implemented for input type {node.inputs[0].type}"
            )
        value_name, *cond_names = inames
        out_name = onames[0]
        check = []
        fail_code = props["fail"]
        param_struct_name = props["params"]
        msg = self.msg.replace('"', '\\"').replace("\n", "\\n")

        for idx, cond_name in enumerate(cond_names):
            if isinstance(node.inputs[0].type, DenseTensorType):
                check.append(
                    f"""
            if(PyObject_IsTrue((PyObject *){cond_name}) == 0) {{
                PyObject * exc_type = {param_struct_name}->exc_type;
                Py_INCREF(exc_type);
                PyErr_SetString(exc_type, "{msg}");
                Py_XDECREF(exc_type);
                {indent(fail_code, " " * 4)}
            }}
                    """
                )
            else:
                check.append(
                    f"""
            if({cond_name} == 0) {{
                PyObject * exc_type = {param_struct_name}->exc_type;
                Py_INCREF(exc_type);
                PyErr_SetString(exc_type, "{msg}");
                Py_XDECREF(exc_type);
                {indent(fail_code, " " * 4)}
            }}
                    """
                )

        check = "\n".join(check)

        if isinstance(node.inputs[0].type, DenseTensorType):
            res = f"""
            {check}
            Py_XDECREF({out_name});
            {out_name} = {value_name};
            Py_INCREF({value_name});
            """
        else:
            res = f"""
            {check}
            {out_name} = {value_name};
            """
        return res

    def c_code_cache_version(self):
        return (1, 1)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


class Assert(CheckAndRaise):
    """Implements assertion in a computational graph.

    Returns the first parameter if the condition is ``True``; otherwise,
    triggers `AssertionError`.

    Notes
    -----
    This `Op` is a debugging feature. It can be removed from the graph
    because of optimizations, and can hide some possible optimizations to
    the optimizer. Specifically, removing happens if it can be determined
    that condition will always be true. Also, the output of the Op must be
    used in the function computing the graph, but it doesn't have to be
    returned.

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> from pytensor.raise_op import Assert
    >>> x = pt.vector("x")
    >>> assert_op = Assert("This assert failed")
    >>> func = pytensor.function([x], assert_op(x, x.size < 2))

    """

    def __init__(self, msg="PyTensor Assert failed!"):
        super().__init__(AssertionError, msg)

    def __str__(self):
        if len(self.msg) > 30:
            msg = self.msg[:27] + "..."
        else:
            msg = self.msg
        return f"Assert{{msg='{msg}'}}"


assert_op = Assert()


@_vectorize_node.register(CheckAndRaise)
def vectorize_check_and_raise(op, node, batch_x, batch_cond):
    from pytensor.tensor.extra_ops import broadcast_arrays
    from pytensor.tensor.shape import shape_padright

    batch_cond_dims = batch_cond.type.ndim

    if batch_cond_dims:
        out = op(batch_x, batch_cond.all())
        # Condition may broadcast batch dims of x
        # We broadcast after the Check Op, so it can be removed more easily if not needed
        x_core_ndim = node.inputs[0].type.ndim
        batch_out, _ = broadcast_arrays(out, shape_padright(batch_cond, x_core_ndim))
        return batch_out.owner
    else:
        return op.make_node(batch_x, batch_cond)
