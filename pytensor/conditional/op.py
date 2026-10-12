"""
IfElse introduces lazy evaluation in PyTensor (coupled with the CVM/VM
linkers). It resembles the if clause of any programming language, that
has a `then` and `else` branch, and executes either one or the other
according to the condition provided.

This op differs from the already existent `switch` op, that evaluates both
branches of the clause and afterwards picks (according to the condition)
which value to report. Note also that `switch` is an elemwise operation (so
it picks each entry of a matrix according to the condition) while `ifelse`
is a global operation with a scalar condition.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pytensor.basic import as_symbolic
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import _NoPythonOp
from pytensor.graph.type import HasDataType, HasShape
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.basic import zeros_like


if TYPE_CHECKING:
    from pytensor.tensor import TensorLike


class IfElse(_NoPythonOp):
    r"""An `Op` that provides conditional graph evaluation.

    According to a scalar condition, this `Op` evaluates and then
    returns all the tensors provided on the "then"-branch, otherwise it
    evaluates and returns the tensors provided on the "else"-branch. The `Op`
    supports multiple tensors on each branch, with the condition that the same
    number of tensors are on the "then"-branch as on the "else"-branch and
    there is a one to one correspondence between their dtypes and numbers of
    dimensions.

    The "then"-branch is defined as the first ``N`` tensors (after the
    condition), while the "else"-branch is defined as the last ``N`` tensors.

    Example usage:

    .. code-block::

        rval = ifelse(condition,
                      rval_if_true_1, ..., rval_if_true_N,
                      rval_if_false_1, ..., rval_if_false_N)

    .. note:

        `Linker`\s other than `CVM`, and some other `VM` subclasses, are
        incompatible with this `Op`, and will ignore its lazy characteristic,
        computing both the true and false branches before returning one.

    """

    __props__ = ("as_view", "n_outs")

    def __init__(self, n_outs, as_view=False, name=None):
        if as_view:
            # check destroyhandler and others to ensure that a view_map with
            # multiple inputs can work
            view_map = {}
            for idx in range(n_outs):
                view_map[idx] = [idx + 1]
            self.view_map = view_map
        self.as_view = as_view
        self.n_outs = n_outs
        self.name = name

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if self.as_view != other.as_view:
            return False
        if self.n_outs != other.n_outs:
            return False
        return True

    def __hash__(self):
        return hash((type(self), self.as_view, self.n_outs))

    def __str__(self):
        args = []
        if self.name is not None:
            args.append(self.name)
        if self.as_view:
            args.append("inplace")
        return f"if{{{','.join(args)}}}"

    def infer_shape(self, node, inputs_shapes):
        # By construction, corresponding then/else pairs have the same number
        # of dimensions

        ts_shapes = inputs_shapes[1:][: self.n_outs]
        fs_shapes = inputs_shapes[1:][self.n_outs :]
        # All elements of all shape tuples for the true and false outputs are
        # unpacked into the inputs of a separate ifelse, and then the outputs
        # of that ifelse are packed back into shape tuples.
        new_ts_inputs = []
        for ts_shape in ts_shapes:
            if isinstance(ts_shape, list | tuple):
                new_ts_inputs += list(ts_shape)
            else:
                # It can be None for generic objects
                return [None] * self.n_outs

        new_fs_inputs = []
        for fs_shape in fs_shapes:
            if isinstance(fs_shape, list | tuple):
                new_fs_inputs += list(fs_shape)
            else:
                # It can be None for generic objects
                return [None] * self.n_outs

        assert len(new_ts_inputs) == len(new_fs_inputs)
        if len(new_ts_inputs + new_fs_inputs) > 0:
            name_tokens = ["shape"]
            if self.name is not None:
                name_tokens.append(self.name)

            new_ifelse = IfElse(
                n_outs=len(new_ts_inputs),
                as_view=False,
                name="_".join(name_tokens),
            )
            new_outs = new_ifelse(
                node.inputs[0],
                *(new_ts_inputs + new_fs_inputs),
                return_list=True,
            )
        else:
            new_outs = []

        # generate pairs of shapes
        out_shapes = []
        for out in node.outputs:
            out_shapes.append(tuple(new_outs[: out.ndim]))
            new_outs = new_outs[out.ndim :]

        # new_outs should be an empty list after last iteration
        assert len(new_outs) == 0

        return out_shapes

    def make_node(self, condition: "TensorLike", *true_false_branches: Any):  # type: ignore[override]
        if len(true_false_branches) != 2 * self.n_outs:
            raise ValueError(
                f"Wrong number of arguments: expected "
                f"{int(2 * self.n_outs)}, got {len(true_false_branches)}"
            )

        condition = as_tensor_variable(condition)

        if condition.type.ndim > 0:
            raise TypeError("The condition argument must be a truthy scalar value")

        inputs_true_branch = true_false_branches[: self.n_outs]
        inputs_false_branch = true_false_branches[self.n_outs :]

        output_vars = []
        new_inputs_true_branch = []
        new_inputs_false_branch = []
        for input_t, input_f in zip(
            inputs_true_branch, inputs_false_branch, strict=True
        ):
            if not isinstance(input_t, Variable):
                input_t = as_symbolic(input_t)
            if not isinstance(input_f, Variable):
                input_f = as_symbolic(input_f)

            if type(input_f.type) != type(input_t.type):  # noqa: E721
                raise TypeError(
                    f"Input types {type(input_t.type)} and {type(input_f.type)} do not match."
                )

            if isinstance(input_t.type, HasDataType) and isinstance(
                input_f.type, HasDataType
            ):
                # TODO: Be smarter about dtype casting.
                # up_dtype = ps.upcast(input_t.type.dtype, input_f.type.dtype)

                if input_t.type.dtype != input_f.type.dtype:
                    raise TypeError(
                        "IfElse requires compatible dtypes for both branches: got "
                        f"true_branch={input_t.type.dtype}, false_branch={input_f.type.dtype}"
                    )

            if isinstance(input_t.type, HasShape) and isinstance(
                input_f.type, HasShape
            ):
                if input_t.type.ndim != input_f.type.ndim:
                    raise TypeError(
                        "IfElse requires compatible ndim values for both branches: got "
                        f"true_branch={input_t.type.ndim}, false_branch={input_f.type.ndim}"
                    )

                # We can only use static shape information that corresponds
                # in both branches, because the outputs of this `Op` are
                # allowed to have distinct shapes from either branch
                new_shape = tuple(
                    s_t if s_t == s_f else None
                    for s_t, s_f in zip(
                        input_t.type.shape, input_f.type.shape, strict=True
                    )
                )
                # TODO FIXME: The presence of this keyword is a strong
                # assumption.  Find something that's guaranteed by the/a
                # confirmed interface.
                output_var_t = input_t.type.clone(shape=new_shape)()  # type: ignore[attr-defined]
                output_var_f = input_f.type.clone(shape=new_shape)()  # type: ignore[attr-defined]
            else:
                output_var_t = input_t.type()
                output_var_f = input_f.type()

            input_t_ = output_var_f.type.filter_variable(input_t)
            input_f_ = output_var_t.type.filter_variable(input_f)

            new_inputs_true_branch.append(input_t_)
            new_inputs_false_branch.append(input_f_)
            output_vars.append(output_var_t)

        return Apply(
            self,
            [condition, *new_inputs_true_branch, *new_inputs_false_branch],
            output_vars,
        )

    def pushforward(self, inputs, outputs, eval_points):
        return self(inputs[0], *eval_points[1:], return_list=True)

    def pullback(self, ins, outputs, grads):
        condition = ins[0]
        inputs_true_branch = ins[1:][: self.n_outs]
        inputs_false_branch = ins[1:][self.n_outs :]

        if self.name is not None:
            nw_name_t = self.name + "_grad_t"
            nw_name_f = self.name + "_grad_f"
        else:
            nw_name_t = None
            nw_name_f = None

        if_true_op = IfElse(n_outs=self.n_outs, as_view=self.as_view, name=nw_name_t)
        if_false_op = IfElse(n_outs=self.n_outs, as_view=self.as_view, name=nw_name_f)

        # The `grads` can have different dtypes than the `inputs`.
        # Since input true/false entries must have the same dtypes, we need to
        # cast the zeros to the corresponding `grads` dtypes and not the input
        # dtypes.
        inputs_true_grad = (
            [condition]
            + grads
            + [
                zeros_like(t, dtype=grads[i].dtype)
                for i, t in enumerate(inputs_true_branch)
            ]
        )
        inputs_false_grad = (
            [condition]
            + [
                zeros_like(f, dtype=grads[i].dtype)
                for i, f in enumerate(inputs_false_branch)
            ]
            + grads
        )

        # `condition` does affect the elements of the output so it is connected.
        # For the sake of making the gradient convenient we assume that
        # condition + epsilon always triggers the same branch as condition
        condition_grad = condition.zeros_like(dtype=config.floatX)

        return [
            condition_grad,
            *if_true_op(*inputs_true_grad, return_list=True),
            *if_false_op(*inputs_false_grad, return_list=True),
        ]

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl=None):
        cond = node.inputs[0]
        input_true_branch = node.inputs[1:][: self.n_outs]
        inputs_false_branch = node.inputs[1:][self.n_outs :]
        outputs = node.outputs

        def thunk():
            if not compute_map[cond][0]:
                return [0]
            else:
                truthval = storage_map[cond][0]
                if truthval != 0:
                    ls = [
                        idx + 1
                        for idx in range(self.n_outs)
                        if not compute_map[input_true_branch[idx]][0]
                    ]
                    if len(ls) > 0:
                        return ls
                    else:
                        # zip strict not specified because we are in a hot loop
                        for out, t in zip(outputs, input_true_branch):
                            compute_map[out][0] = 1
                            val = storage_map[t][0]
                            if self.as_view:
                                storage_map[out][0] = val
                            # Work around broken numpy deepcopy
                            elif isinstance(val, np.ndarray | np.memmap):
                                storage_map[out][0] = val.copy()
                            else:
                                storage_map[out][0] = deepcopy(val)
                        return []
                else:
                    ls = [
                        1 + idx + self.n_outs
                        for idx in range(self.n_outs)
                        if not compute_map[inputs_false_branch[idx]][0]
                    ]
                    if len(ls) > 0:
                        return ls
                    else:
                        # zip strict not specified because we are in a hot loop
                        for out, f in zip(outputs, inputs_false_branch):
                            compute_map[out][0] = 1
                            # can't view both outputs unless destroyhandler
                            # improves
                            # Work around broken numpy deepcopy
                            val = storage_map[f][0]
                            if isinstance(val, np.ndarray | np.memmap):
                                storage_map[out][0] = val.copy()
                            else:
                                storage_map[out][0] = deepcopy(val)
                        return []

        thunk.lazy = True
        thunk.inputs = [storage_map[v] for v in node.inputs]
        thunk.outputs = [storage_map[v] for v in node.outputs]
        return thunk


def ifelse(
    condition: "TensorLike",
    then_branch: Any | Sequence[Any],
    else_branch: Any | Sequence[Any],
    name: str | None = None,
) -> Variable | Sequence[Variable]:
    """Construct a graph for an ``if`` statement.

    Parameters
    ----------
    condition
        `condition` should be a tensor scalar representing the condition.
        If it evaluates to ``0`` it corresponds to ``False``, anything else
        stands for ``True``.

    then_branch
        A single variable or a list of variables that the
        function should return as the output if `condition` evaluates to
        true. The number of variables should match those in the
        `else_branch`, as well as the dtypes and numbers of dimensions of each
        tensor.

    else_branch
        A single variable or a list of variables that the function should
        return as the output if `condition` evaluates to false. The number of
        variables should match those in `then_branch`, as well as the dtypes
        and numbers of dimensions of each tensor.

    Returns
    -------
        A sequence of variables or a single variable, depending on the
        nature of `then_branch` and `else_branch`.  More exactly, if
        `then_branch` and `else_branch` is are single variables, then
        the return variable will also be a single variable; otherwise, it will
        be a sequence. The value returned correspond to either the values in
        the `then_branch` or in the `else_branch` depending on the value of
        `condition`.
    """

    rval_type = None
    if isinstance(then_branch, list | tuple):
        rval_type = type(then_branch)
    else:
        then_branch = [then_branch]

    if not isinstance(else_branch, list | tuple):
        else_branch = [else_branch]

    if len(then_branch) != len(else_branch):
        raise ValueError(
            "The number of values on the `then` branch "
            "must match the `else` branch: got "
            f"{len(then_branch)} for `then`, and "
            f"{len(else_branch)} for `else`."
        )

    new_ifelse = IfElse(n_outs=len(then_branch), as_view=False, name=name)

    ins = [condition, *then_branch, *else_branch]
    rval = cast(list[Variable], new_ifelse(*ins, return_list=True))

    if rval_type is None:
        return rval[0]
    elif rval_type is list:
        return list(rval)
    else:
        return tuple(rval)
