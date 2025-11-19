import numba

import pytensor.link.numba.dispatch.basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.tensor.type_other import SliceType
from pytensor.typed_list import (
    Append,
    Count,
    Extend,
    GetItem,
    Index,
    Insert,
    Length,
    MakeList,
    Remove,
    Reverse,
)


def numba_all_equal(x, y):
    raise NotImplementedError


@numba.extending.overload(numba_all_equal)
def list_all_equal(x, y):
    all_equal = None

    if isinstance(x, numba.types.List) and isinstance(y, numba.types.List):

        def all_equal(x, y):
            if len(x) != len(y):
                return False
            for xi, yi in zip(x, y):
                if not numba_all_equal(xi, yi):
                    return False
            return True

    if isinstance(x, numba.types.Array) and isinstance(y, numba.types.Array):

        def all_equal(x, y):
            return (x == y).all()

    if isinstance(x, numba.types.Number) and isinstance(y.numba.types.Number):

        def all_equal(x, y):
            return x == y

    return all_equal


@numba.extending.overload(numba_deepcopy)
def numba_deepcopy_list(x):
    if isinstance(x, numba.types.List):

        def deepcopy_list(x):
            return [numba_deepcopy(xi) for xi in x]

        return deepcopy_list


@register_funcify_default_op_cache_key(GetItem)
def numba_funcify_list_get_item(op, node, **kwargs):
    if isinstance(node.inputs[1].type, SliceType):

        @numba_basic.numba_njit
        def list_get_item_slice(x, index):
            return x[index]

        return list_get_item_slice

    else:

        @numba_basic.numba_njit
        def list_get_item_index(x, index):
            return x[index.item()]

        return list_get_item_index


@register_funcify_default_op_cache_key(Reverse)
def numba_funcify_list_reverse(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_reverse(x):
        if inplace:
            z = x.copy()
        else:
            z = x

        z.reverse()
        return z

    return list_reverse


@register_funcify_default_op_cache_key(Append)
def numba_funcify_list_append(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_append(x, to_append):
        if inplace:
            z = x.copy()
        else:
            z = x

        z.append(numba_deepcopy(to_append))
        return z

    return list_append


@register_funcify_default_op_cache_key(Extend)
def numba_funcify_list_extend(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_extend(x, to_append):
        if inplace:
            z = x.copy()
        else:
            z = x

        z.extend(numba_deepcopy(to_append))
        return z

    return list_extend


@register_funcify_default_op_cache_key(Insert)
def numba_funcify_list_insert(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_insert(x, index, to_insert):
        if inplace:
            z = x.copy()
        else:
            z = x
        z.insert(index.item(), numba_deepcopy(to_insert))
        return z

    return list_insert


@register_funcify_default_op_cache_key(Remove)
def numba_funcify_list_remove(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_remove(x, to_remove):
        if inplace:
            z = x.copy()
        else:
            z = x

        index_to_remove = -1
        for i, zi in enumerate(z):
            if numba_all_equal(zi, to_remove):
                index_to_remove = i
                break
        if index_to_remove == -1:
            raise ValueError("list.remove(x): x not in list")
        z.pop(index_to_remove)
        return z

    return list_remove


@register_funcify_default_op_cache_key(Index)
def numba_funcify_list_index(op, node, **kwargs):
    @numba_basic.numba_njit
    def list_index(x, elem):
        for idx, xi in enumerate(x):
            if numba_all_equal(xi, elem):
                break
        return idx

    return list_index


@register_funcify_default_op_cache_key(Count)
def numba_funcify_list_count(op, node, **kwargs):
    @numba_basic.numba_njit
    def list_count(x, elem):
        c = 0
        for xi in x:
            if numba_all_equal(xi, elem):
                c += 1
        return c

    return list_count


@register_funcify_default_op_cache_key(Length)
def numba_funcify_list_length(op, node, **kwargs):
    @numba_basic.numba_njit
    def list_length(x):
        return len(x)

    return list_length


@register_funcify_default_op_cache_key(MakeList)
def numba_funcify_make_list(op, node, **kwargs):
    @numba_basic.numba_njit
    def make_list(*args):
        return [numba_deepcopy(arg) for arg in args]

    return make_list
