import copy
import math

import pytest

from pytensor.assumptions import (
    AssumptionFeature,
    AssumptionKey,
    register_assumption,
    register_matrix_property_rules,
)
from pytensor.assumptions import core as _assumptions_core
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op


@pytest.fixture(autouse=True)
def _snapshot_assumption_registries():
    """Restore module-global assumption registries after each test."""
    infer_snapshot = copy.deepcopy(_assumptions_core.ASSUMPTION_INFER_REGISTRY)
    implies_snapshot = copy.deepcopy(_assumptions_core.IMPLIES)
    key_snapshot = dict(_assumptions_core.KEY_REGISTRY)
    universal_snapshot = list(_assumptions_core.UNIVERSAL_RULES)
    client_snapshot = dict(_assumptions_core.CLIENT_INFER_REGISTRY)
    client_keys_snapshot = set(_assumptions_core.CLIENT_INFER_KEYS)
    try:
        yield
    finally:
        _assumptions_core.ASSUMPTION_INFER_REGISTRY.clear()
        _assumptions_core.ASSUMPTION_INFER_REGISTRY.update(infer_snapshot)
        _assumptions_core.IMPLIES.clear()
        _assumptions_core.IMPLIES.update(implies_snapshot)
        _assumptions_core.KEY_REGISTRY.clear()
        _assumptions_core.KEY_REGISTRY.update(key_snapshot)
        _assumptions_core.UNIVERSAL_RULES[:] = universal_snapshot
        _assumptions_core.CLIENT_INFER_REGISTRY.clear()
        _assumptions_core.CLIENT_INFER_REGISTRY.update(client_snapshot)
        _assumptions_core.CLIENT_INFER_KEYS.clear()
        _assumptions_core.CLIENT_INFER_KEYS.update(client_keys_snapshot)


def make_fgraph(*outputs, **kwargs):
    fg = FunctionGraph(outputs=outputs, clone=False, **kwargs)
    af = AssumptionFeature()
    fg.attach_feature(af)
    return fg, af


class RankBound(AssumptionKey):
    value_type = float
    unknown = math.inf

    def join(self, left, right):
        return min(left, right)

    def is_conflict(self, value):
        return False


class RankBoundedOp(Op):
    __props__ = ("bound",)

    def __init__(self, bound):
        self.bound = bound

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0]


@pytest.fixture
def rank():
    key = RankBound("rank_bound")
    register_matrix_property_rules(key)

    @register_assumption(key, RankBoundedOp)
    def _rank_bounded(key, op, feature, fgraph, node, input_states):
        return [op.bound]

    return key
