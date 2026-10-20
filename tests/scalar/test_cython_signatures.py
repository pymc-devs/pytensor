import subprocess
import sys

import numpy as np
import pytest

from pytensor.scalar.cython_special import Signature, _CythonFunctionSpec


@pytest.mark.parametrize(
    "sig, expected_result, expected_args",
    [
        (b"double(double)", np.float64, [np.float64]),
        (b"float(unsigned int)", np.float32, [np.uintc]),
        (b"unsigned char(unsigned short foo)", np.ubyte, [np.ushort]),
        (
            b"unsigned char(unsigned short foo, double bar)",
            np.ubyte,
            [np.ushort, np.float64],
        ),
    ],
)
def test_parse_signature(sig, expected_result, expected_args):
    actual = Signature.from_c_types(sig)
    assert actual.res_dtype == expected_result
    assert actual.arg_dtypes == expected_args


@pytest.mark.parametrize(
    "have, want, should_provide",
    [
        (b"double(int)", b"float(int)", True),
        (b"float(int)", b"double(int)", False),
        (b"double(unsigned short)", b"double(unsigned char)", True),
        (b"double(unsigned char)", b"double(short)", False),
        (b"short(double)", b"int(double)", True),
        (b"int(double)", b"short(double)", False),
        (b"float(double, int)", b"float(double, short)", True),
    ],
)
def test_signature_provides(have, want, should_provide):
    have = Signature.from_c_types(have)
    want = Signature.from_c_types(want)
    provides = have.provides(want.res_dtype, want.arg_dtypes)
    assert provides == should_provide


@pytest.mark.parametrize(
    "raw, has_dispatch",
    [
        (b"double (long, double)", False),
        (b"double (long long, double, int __pyx_skip_dispatch)", True),
    ],
)
def test_signature_preserves_c_abi(raw, has_dispatch):
    signature = Signature.from_c_types(raw)
    assert signature.c_signature == raw.decode()
    assert signature.arg_c_types[0] == ("long long" if has_dispatch else "long")
    spec = _CythonFunctionSpec(signature, "export", "module")
    assert spec.has_pyx_skip_dispatch() is has_dispatch


def test_lazy_cython_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pytensor.scalar.cython_special; "
            "assert 'scipy' not in sys.modules; "
            "assert 'numba' not in sys.modules",
        ],
        check=True,
    )
