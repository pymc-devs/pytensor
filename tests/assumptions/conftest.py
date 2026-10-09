import copy

import pytest

from pytensor.assumptions import AssumptionFeature
from pytensor.assumptions import core as _assumptions_core
from pytensor.graph.fg import FunctionGraph


@pytest.fixture(autouse=True)
def _snapshot_assumption_registries():
    """Restore module-global assumption registries after each test."""
    infer_snapshot = copy.deepcopy(_assumptions_core.ASSUMPTION_INFER_REGISTRY)
    implies_snapshot = copy.deepcopy(_assumptions_core.IMPLIES)
    key_snapshot = dict(_assumptions_core.KEY_REGISTRY)
    universal_snapshot = list(_assumptions_core.UNIVERSAL_RULES)
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


def make_fgraph(*outputs, **kwargs):
    fg = FunctionGraph(outputs=outputs, clone=False, **kwargs)
    af = AssumptionFeature()
    fg.attach_feature(af)
    return fg, af
