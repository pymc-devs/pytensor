from itertools import chain
from typing import Optional, Sequence, Tuple

from pytensor.compile import rebuild_collect_shared
from pytensor.graph.basic import Constant, Variable, clone
from pytensor.graph.fg import FunctionGraph
from pytensor.scalar.basic import ScalarInnerGraphOp, as_scalar


class ScalarLoop(ScalarInnerGraphOp):
    """Scalar Op that encapsulates a scalar loop operation.

    This Op can be used for the gradient of other Scalar Ops.
    It is much more restricted than `Scan` in that the entire inner graph
    must be composed of Scalar operations, and all inputs and outputs must be ScalarVariables.

    The pseudocode of the computation performed by this Op looks like the following:

    ```python
    def scalar_for_loop(fn, n_steps, init, update, constant):
        for i in range(n_steps):
            state = fn(*state, *constant)
        return state
    ```

    When an until condition is present it behaves like this:

    ```python
    def scalar_while_loop(fn, n_steps, init, update, constant):
        # If n_steps <= 0, we skip the loop altogether.
        # This does not count as a "failure"
        done = True

        for i in range(n_steps):
            *state, done = fn(*state, *constant)
            if done:
                break

        return *state, done
    ```

    """

    init_param: Tuple[str, ...] = (
        "init",
        "update",
        "constant",
        "until",
    )

    def __init__(
        self,
        init: Sequence[Variable],
        update: Sequence[Variable],
        constant: Optional[Sequence[Variable]] = None,
        until: Optional[Variable] = None,
        name="ScalarLoop",
    ):
        if constant is None:
            constant = []
        if not len(init) == len(update):
            raise ValueError("An update must be given for each init variable")
        if until:
            inputs, outputs = clone([*init, *constant], [*update, until])
        else:
            inputs, outputs = clone([*init, *constant], update)

        self.is_while = bool(until)
        self.inputs, self.outputs = self._cleanup_graph(inputs, outputs)
        self._validate_updates(self.inputs, self.outputs)

        self.inputs_type = tuple(input.type for input in self.inputs)
        self.outputs_type = tuple(output.type for output in self.outputs)
        self.nin = len(self.inputs) + 1  # n_steps is not part of the inner graph
        self.nout = len(self.outputs)
        self.name = name

        super().__init__()

    def output_types(self, input_types):
        return self.outputs_type

    def _validate_updates(
        self, inputs: Sequence[Variable], outputs: Sequence[Variable]
    ) -> None:
        init = inputs
        update: Sequence[Variable]
        if self.is_while:
            *update, until = outputs
            if not until.type.dtype == "bool":
                raise TypeError(
                    f"Until condition must be boolean, got {until}({until.type.dtype})"
                )
        else:
            update = outputs
        for i, u in zip(init, update):
            if i.type != u.type:
                raise TypeError(
                    "Init and update types must be the same: "
                    f"{i}({i.type}) != {u}({u.type})"
                )
        if set(init) & set(update):
            raise ValueError(
                "Some inputs and outputs are the same variable. "
                "If you want to return an output as a lagged input, wrap it in an identity Op."
            )

    @property
    def fgraph(self):
        if hasattr(self, "_fgraph"):
            return self._fgraph
        # fgraph cannot be a property of the base class because it messes up with C caching.
        # We also need a `FunctionGraph(clone=True)` (default) according to an old comment
        fgraph = FunctionGraph(self.inputs, self.outputs)
        self._fgraph = fgraph
        return self._fgraph

    def clone(self):
        if self.is_while:
            *update, until = self.outputs
        else:
            update, until = self.outputs, None
        init = self.inputs[: len(update)]
        constant = self.inputs[len(update) :]
        return ScalarLoop(
            init=init,
            update=update,
            constant=constant,
            until=until,
            name=self.name,
        )

    @property
    def fn(self):
        raise NotImplementedError

    def make_new_inplace(self, output_types_preference=None, name=None):
        """
        This op.__init__ fct don't have the same parameter as other scalar op.
        This break the insert_inplace_optimizer optimization.
        This fct allow fix patch this.

        """
        d = {k: getattr(self, k) for k in self.init_param}
        out = self.__class__(**d)
        if name:
            out.name = name
        else:
            name = out.name
        super(ScalarLoop, out).__init__(output_types_preference, name)
        return out

    def make_node(self, n_steps, *inputs):
        assert len(inputs) == self.nin - 1

        n_steps = as_scalar(n_steps)
        if not n_steps.type.dtype.startswith("int"):
            raise TypeError(
                "The first variable of ScalarLoop (n_steps) must be of integer type. "
                f"Got {n_steps.type.dtype}",
            )

        if self.inputs_type == tuple([i.type for i in inputs]):
            return super().make_node(n_steps, *inputs)
        else:
            # Make a new op with the right input types.
            res = rebuild_collect_shared(
                self.outputs,
                replace=dict(zip(self.inputs, inputs)),
                rebuild_strict=False,
            )
            if self.is_while:
                *cloned_update, cloned_until = res[1]
            else:
                cloned_update, cloned_until = res[1], None
            cloned_inputs = [res[2][0][i] for i in inputs]
            cloned_init = cloned_inputs[: len(cloned_update)]
            cloned_constant = cloned_inputs[len(cloned_update) :]
            # This will fail if the cloned init have a different dtype than the cloned_update
            op = ScalarLoop(
                init=cloned_init,
                update=cloned_update,
                constant=cloned_constant,
                until=cloned_until,
                name=self.name,
            )
            node = op.make_node(n_steps, *inputs)
            return node

    def perform(self, node, inputs, output_storage):
        n_steps, *inputs = inputs
        n_update = len(self.outputs) - (1 if self.is_while else 0)
        carry, constant = inputs[:n_update], inputs[n_update:]
        inner_fn = self.py_perform_fn

        if self.is_while:
            until = True
            for i in range(n_steps):
                *carry, until = inner_fn(*carry, *constant)
                if until:
                    break
            carry.append(until)

        else:
            if n_steps < 0:
                raise ValueError("ScalarLoop does not have a termination condition.")
            for i in range(n_steps):
                carry = inner_fn(*carry, *constant)

        for storage, out_val in zip(output_storage, carry):
            storage[0] = out_val

    @property
    def c_code_template(self):
        from pytensor.link.c.interface import CLinkerType

        if hasattr(self, "_c_code"):
            return self._c_code

        fgraph = self.fgraph

        # The first input is `n_steps` so we skip it in the mapping dictionary
        n_update = len(self.outputs) - (1 if self.is_while else 0)
        carry_subd = {
            c: f"%(i{int(i)})s" for i, c in enumerate(fgraph.inputs[:n_update], start=1)
        }
        constant_subd = {
            c: f"%(i{int(i)})s"
            for i, c in enumerate(fgraph.inputs[n_update:], start=n_update + 1)
        }
        update_subd = {
            u: f"%(o{int(i)})s" for i, u in enumerate(fgraph.outputs[:n_update])
        }
        until_subd = {u: "until" for u in fgraph.outputs[n_update:]}
        subd = {**carry_subd, **constant_subd, **update_subd, **until_subd}

        for var in fgraph.variables:
            if var.owner is None:
                if var not in self.fgraph.inputs:
                    # This is an orphan
                    if isinstance(var, Constant) and isinstance(var.type, CLinkerType):
                        subd[var] = var.type.c_literal(var.data)
                    else:
                        raise ValueError(
                            "All orphans in the fgraph to ScalarLoop must"
                            " be Constant, CLinkerType instances."
                        )
            elif any(i.dtype == "float16" for i in var.owner.inputs) or any(
                o.dtype == "float16" for o in var.owner.outputs
            ):
                # flag for elemwise ops to check.
                self.inner_float16 = True

        _c_code = "{\n"
        if self.is_while:
            _c_code += "bool until = 1;\n\n"

        # Copy carried inputs
        for i, (var, name) in enumerate(carry_subd.items()):
            copy_var_name = f"{name}_copy{i}"
            _c_code += f"{var.type.dtype_specs()[1]} {copy_var_name} = {name};\n"
            carry_subd[var] = copy_var_name
            subd[var] = copy_var_name

        # _c_code += 'printf("inputs=[");'
        # for i in range(1, len(fgraph.inputs)):
        #     _c_code += f'printf("%%.16g, ", %(i{i})s);'
        # _c_code += 'printf("]\\n");\n'

        _c_code += "\nfor(%(n_steps_dtype)s i = 0; i < %(n_steps)s; i++){\n"

        self.nodenames = [
            f"%(nodename)s_subnode{int(j)}" for j, n in enumerate(fgraph.toposort())
        ]

        i = 0
        for j, node in enumerate(fgraph.toposort()):
            for output in node.outputs:
                if output not in subd:
                    i += 1
                    name = f"V%(id)s_tmp{int(i)}"
                    subd[output] = name
                    _c_code += f"{output.type.dtype_specs()[1]} {name};\n"
            s = node.op.c_code(
                node,
                self.nodenames[j],
                # Any node that depended on `init` will depend on `update` instead
                # The initial value of `update` was set to `init` before the loop
                [subd[input] for input in node.inputs],
                [subd[output] for output in node.outputs],
                dict(fail="%(fail)s", id=f"%(id)s_{int(j)}"),
            )
            _c_code += s
            _c_code += "\n"

        # Set the carry variables to the output variables
        _c_code += "\n"
        for init, update in zip(carry_subd.values(), update_subd.values()):
            _c_code += f"{init} = {update};\n"

        # _c_code += 'printf("%%ld\\n", i);\n'
        # for carry in range(1, 10):
        #     _c_code += f'printf("\\t %%.g\\n", i, %(i{carry})s_copy{carry-1});\n'

        if self.is_while:
            _c_code += "\nif(until){break;}\n"

        # End of the loop
        _c_code += "}\n"

        # Output until flag
        if self.is_while:
            _c_code += f"%(o{len(fgraph.outputs)-1})s = until;\n"

        _c_code += "}\n"

        self._c_code = _c_code

        return self._c_code

    def c_code(self, node, nodename, inames, onames, sub):
        d = dict(
            chain(
                zip((f"i{int(i)}" for i in range(len(inames))), inames),
                zip((f"o{int(i)}" for i in range(len(onames))), onames),
            ),
            **sub,
        )
        d["nodename"] = nodename
        if "id" not in sub:
            # The use of a dummy id is safe as the code is in a separate block.
            # It won't generate conflicting variable name.
            d["id"] = "_DUMMY_ID_"

        # When called inside Elemwise we don't have access to the dtype
        # via the usual `f"dtype_{inames[i]}"` variable
        d["n_steps"] = inames[0]
        d["n_steps_dtype"] = "npy_" + node.inputs[0].dtype

        res = self.c_code_template % d
        # print(res)
        return res

    def c_code_cache_version_outer(self):
        return (2,)
