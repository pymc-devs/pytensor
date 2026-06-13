"""Run the legacy scan test suite against the new scan constructor.

This is a temporary local harness to measure feature parity: an adapter maps
the legacy `pytensor.scan` signature to `pytensor.loop.basic.scan`, and the
legacy tests are re-exported with the adapter patched in. Tests that use
features the new constructor does not support yet are skipped by the adapter.

Run with:
    LOOP_LEGACY_COMPAT=1 pytest tests/loop/test_legacy_compat.py
"""

import os

import pytest

from pytensor.loop.basic import scan as loop_scan
from pytensor.scan.utils import until


if not os.environ.get("LOOP_LEGACY_COMPAT"):
    pytest.skip(
        "Set LOOP_LEGACY_COMPAT=1 to run the legacy scan suite against the new constructor",
        allow_module_level=True,
    )

import tests.scan.test_basic as legacy_test_basic


def scan_compat(
    fn,
    sequences=None,
    outputs_info=None,
    non_sequences=None,
    n_steps=None,
    truncate_gradient=-1,
    go_backwards=False,
    mode=None,
    name=None,
    profile=False,
    allow_gc=None,
    strict=False,
    return_list=False,
    return_updates=True,
):
    if truncate_gradient != -1:
        pytest.skip("truncate_gradient not supported by new scan constructor")

    if outputs_info is None:
        init_states = None
    else:
        if not isinstance(outputs_info, list | tuple):
            outputs_info = [outputs_info]
        init_states = []
        for info in outputs_info:
            if isinstance(info, dict):
                taps = info.get("taps", [-1])
                if taps != [-1]:
                    pytest.skip(
                        f"taps {taps} not supported by new scan constructor yet"
                    )
                init_states.append(info["initial"])
            else:
                init_states.append(info)

    if go_backwards and sequences is not None:
        if not isinstance(sequences, list | tuple):
            sequences = [sequences]
        sequences = [s[::-1] for s in sequences]

    def fn_compat(*args):
        res = fn(*args)
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
            outputs, updates = res
            if updates:
                pytest.skip("updates dict not supported by new scan constructor")
            res = outputs
        if isinstance(res, dict):
            pytest.skip("updates dict not supported by new scan constructor")
        return res

    traces = loop_scan(
        fn=fn_compat,
        init_states=init_states,
        sequences=sequences,
        non_sequences=non_sequences,
        n_steps=n_steps,
    )

    if return_list and not isinstance(traces, list):
        traces = [traces]
    if return_updates:
        return traces, {}
    return traces


# Patch the module global so all legacy tests build graphs with the new constructor
legacy_test_basic.scan = scan_compat
legacy_test_basic.until = until

# Re-export every test so pytest collects it from this module
for _name in dir(legacy_test_basic):
    if _name.startswith(("test_", "Test")):
        globals()[_name] = getattr(legacy_test_basic, _name)
