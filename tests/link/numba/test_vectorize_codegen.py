from llvmlite import ir
from numba import types

from pytensor.link.numba.dispatch.vectorize_codegen import compute_itershape


class Mock32BitContext:
    """Minimal Numba context whose ``intp`` LLVM representation is i32."""

    class CallConv:
        @staticmethod
        def return_user_exc(builder, exc, args):
            pass

    call_conv = CallConv()

    @staticmethod
    def get_constant(typ, value):
        assert typ is types.intp
        return ir.Constant(ir.IntType(32), value)


def test_compute_itershape_uses_target_intp_width():
    module = ir.Module()
    function_type = ir.FunctionType(ir.VoidType(), ())
    function = ir.Function(module, function_type, "compute_itershape")
    builder = ir.IRBuilder(function.append_basic_block("entry"))

    one_i32 = ir.Constant(ir.IntType(32), 1)
    shape = compute_itershape(
        Mock32BitContext(),
        builder,
        in_shapes=[[one_i32]],
        broadcast_pattern=((True,),),
        size=None,
    )
    builder.ret_void()

    assert shape == [one_i32]
    assert "icmp ne i32" in str(module)
