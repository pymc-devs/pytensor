from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.graph.traversal import (
    apply_ancestors,
    toposort,
    toposort_with_orderings,
    variable_ancestors,
)
from pytensor.graph.type import Type


class MyType(Type):
    def __init__(self, thingy):
        self.thingy = thingy

    def filter(self, *args, **kwargs):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self.thingy == other.thingy

    def __hash__(self):
        return hash(self.thingy)


def MyVariable(thingy):
    return Variable(MyType(thingy), owner=None, name=f"v{thingy}")


class _MyOp(Op):
    __props__ = ()

    def make_node(self, *inputs):
        outputs = [Variable(MyType(sum(i.type.thingy for i in inputs)), owner=None)]
        return Apply(self, list(inputs), outputs)

    def perform(self, *args, **kwargs):
        raise NotImplementedError()


_my_op = _MyOp()


class Traversal:
    """Benchmark graph traversal operations on a deep graph."""

    params = [
        "variable_ancestors",
        "variable_ancestors_with_blockers",
        "apply_ancestors",
        "apply_ancestors_with_blockers",
        "toposort",
        "toposort_with_blockers",
        "toposort_with_orderings",
        "toposort_with_orderings_and_blockers",
    ]
    param_names = ["func_name"]

    def setup(self, func_name):
        r1 = MyVariable(1)
        out = r1
        for _ in range(50):
            out = _my_op(out, out)
        self.out = out

        blocker = out.clone()
        funcs = {
            "variable_ancestors": lambda: all(variable_ancestors([self.out])),
            "variable_ancestors_with_blockers": lambda: all(
                variable_ancestors([self.out], blockers=[blocker])
            ),
            "apply_ancestors": lambda: all(apply_ancestors([self.out])),
            "apply_ancestors_with_blockers": lambda: all(
                apply_ancestors([self.out], blockers=[blocker])
            ),
            "toposort": lambda: all(toposort([self.out])),
            "toposort_with_blockers": lambda: all(
                toposort([self.out], blockers=[blocker])
            ),
            "toposort_with_orderings": lambda: all(
                toposort_with_orderings([self.out], orderings={self.out.owner: []})
            ),
            "toposort_with_orderings_and_blockers": lambda: all(
                toposort_with_orderings(
                    [self.out],
                    blockers=[blocker],
                    orderings={self.out.owner: []},
                )
            ),
        }
        self.func = funcs[func_name]

    def time_traversal(self, func_name):
        self.func()
