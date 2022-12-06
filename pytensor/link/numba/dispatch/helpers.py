from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic


def tuple_mapper(item_map_func):
    @intrinsic
    def map_tuple(typingctx, *input_tuples):
        signatures = [
            typingctx.resolve_function_type(item_map_func, args, {})
            for args in zip(*[in_type.types for in_type in input_tuples], strict=True)
        ]

        output_type = types.Tuple([sig.return_type for sig in signatures])
        signature = output_type(types.StarArgTuple(input_tuples))

        def codegen(context, builder, signature, args):
            (input_tuples,) = args
            input_values = []
            for val in cgutils.unpack_tuple(builder, input_tuples):
                input_values.append(cgutils.unpack_tuple(builder, val))

            mapped_values = []
            for values, sig in zip(zip(*input_values), signatures, strict=True):
                func = context.compile_subroutine(builder, item_map_func, sig)
                output = context.call_internal(builder, func.fndesc, sig, values)
                mapped_values.append(output)

            return context.make_tuple(builder, output_type, mapped_values)

        return signature, codegen

    return map_tuple


@njit
def check_broadcasting(array, bcs, shape):
    assert array.ndim == len(shape)
    for bc, array_length, length in zip(bcs, array.shape, shape):
        if bc:
            assert array_length == 1
        else:
            assert array_length == length
