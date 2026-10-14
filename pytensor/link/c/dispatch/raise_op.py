from collections.abc import Hashable
from textwrap import indent

from pytensor.graph.basic import Apply
from pytensor.link.c.dispatch.basic import CImpl, c_funcify
from pytensor.link.c.params_type import Params, ParamsType
from pytensor.link.c.type import Generic
from pytensor.raise_op import CheckAndRaise
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import DenseTensorType


class ExceptionType(Generic):
    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


exception_type = ExceptionType()


class CheckAndRaiseImpl(CImpl):
    """C implementation of `CheckAndRaise`.

    The exception type is a runtime ``PyObject`` passed through a `ParamsType`;
    the message is baked into the generated code (it is part of the op's props).
    """

    op: CheckAndRaise
    params_type = ParamsType(exc_type=exception_type)

    def get_params(self, node: Apply) -> Params:
        return self.params_type.get_params(self.op)

    def c_code_cache_version(self) -> tuple[Hashable, ...]:
        return (2,)

    def c_code(
        self,
        node: Apply,
        name: str,
        inputs: list[str],
        outputs: list[str],
        sub: dict[str, str],
    ) -> str:
        if not isinstance(node.inputs[0].type, DenseTensorType | ScalarType):
            raise NotImplementedError(
                f"CheckAndRaise c_code not implemented for input type {node.inputs[0].type}"
            )
        value_name, *cond_names = inputs
        out_name = outputs[0]
        fail_code = sub["fail"]
        param_struct_name = sub["params"]
        msg = self.op.msg.replace('"', '\\"').replace("\n", "\\n")

        all_conds = " && ".join(cond_names)
        check = f"""
         if(!({all_conds})) {{
            PyObject * exc_type = {param_struct_name}->exc_type;
            Py_INCREF(exc_type);
            PyErr_SetString(exc_type, "{msg}");
            Py_XDECREF(exc_type);
            {indent(fail_code, " " * 4)}
        }}
        """

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

        return "\n".join((check, res))


@c_funcify.register(CheckAndRaise)
def c_funcify_check_and_raise(op, node=None, **kwargs) -> CheckAndRaiseImpl:
    return CheckAndRaiseImpl(op)
