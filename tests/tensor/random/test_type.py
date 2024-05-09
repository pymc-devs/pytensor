import pickle

import numpy as np
import pytest

from pytensor import shared
from pytensor.compile.ops import ViewOp
from pytensor.tensor.random.type import (
    RandomGeneratorType,
    random_generator_type,
)


# @pytest.mark.skipif(
#     not config.cxx, reason="G++ not available, so we need to skip this test."
# )
def test_view_op_c_code():
    # TODO: It might be good to make sure that the registered C code works
    # (even though it's basically copy-paste from other registered `Op`s).
    # from pytensor.compile.ops import view_op
    # from pytensor.link.c.basic import CLinker
    # rng_var = random_state_type()
    # rng_view = view_op(rng_var)
    # function(
    #     [rng_var],
    #     rng_view,
    #     mode=Mode(optimizer=None, linker=CLinker()),
    # )
    assert ViewOp.c_code_and_version[RandomGeneratorType]


class TestRandomGeneratorType:
    def test_pickle(self):
        rng_r = random_generator_type()

        rng_pkl = pickle.dumps(rng_r)
        rng_unpkl = pickle.loads(rng_pkl)

        assert rng_r != rng_unpkl
        assert rng_r.type == rng_unpkl.type
        assert hash(rng_r.type) == hash(rng_unpkl.type)

    def test_repr(self):
        assert repr(random_generator_type) == "RandomGeneratorType"

    def test_filter(self):
        rng_type = random_generator_type

        rng = np.random.default_rng()
        assert rng_type.filter(rng) is rng

        with pytest.raises(TypeError):
            rng_type.filter(1)

        rng_dict = rng.__getstate__()

        assert rng_type.is_valid_value(rng_dict) is False
        assert rng_type.is_valid_value(rng_dict, strict=False)

        rng_dict["state"] = {}

        assert rng_type.is_valid_value(rng_dict, strict=False) is False

        rng_dict = {}
        assert rng_type.is_valid_value(rng_dict, strict=False) is False

    def test_values_eq(self):
        rng_type = random_generator_type
        bg_1 = np.random.PCG64()
        bg_2 = np.random.Philox()
        bg_3 = np.random.MT19937()
        bg_4 = np.random.SFC64()

        bitgen_a = np.random.Generator(bg_1)
        bitgen_b = np.random.Generator(bg_1)
        assert rng_type.values_eq(bitgen_a, bitgen_b)

        bitgen_c = np.random.Generator(bg_2)
        bitgen_d = np.random.Generator(bg_2)
        assert rng_type.values_eq(bitgen_c, bitgen_d)

        bitgen_e = np.random.Generator(bg_3)
        bitgen_f = np.random.Generator(bg_3)
        assert rng_type.values_eq(bitgen_e, bitgen_f)

        bitgen_g = np.random.Generator(bg_4)
        bitgen_h = np.random.Generator(bg_4)
        assert rng_type.values_eq(bitgen_g, bitgen_h)

        assert rng_type.is_valid_value(bitgen_a, strict=True)
        assert rng_type.is_valid_value(bitgen_b.__getstate__(), strict=False)
        assert rng_type.is_valid_value(bitgen_c, strict=True)
        assert rng_type.is_valid_value(bitgen_d.__getstate__(), strict=False)
        assert rng_type.is_valid_value(bitgen_e, strict=True)
        assert rng_type.is_valid_value(bitgen_f.__getstate__(), strict=False)
        assert rng_type.is_valid_value(bitgen_g, strict=True)
        assert rng_type.is_valid_value(bitgen_h.__getstate__(), strict=False)

    def test_may_share_memory(self):
        bg_a = np.random.PCG64()
        bg_b = np.random.PCG64()
        rng_a = np.random.Generator(bg_a)
        rng_b = np.random.Generator(bg_b)

        rng_var_a = shared(rng_a, borrow=True)
        rng_var_b = shared(rng_b, borrow=True)

        assert (
            random_generator_type.may_share_memory(
                rng_var_a.get_value(borrow=True), rng_var_b.get_value(borrow=True)
            )
            is False
        )

        rng_c = np.random.Generator(bg_b)
        rng_var_c = shared(rng_c, borrow=True)

        assert (
            random_generator_type.may_share_memory(
                rng_var_b.get_value(borrow=True), rng_var_c.get_value(borrow=True)
            )
            is True
        )
