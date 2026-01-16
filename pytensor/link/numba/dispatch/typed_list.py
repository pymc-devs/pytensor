import numba
import numpy as np
from numba.types import Array, Boolean, List, Number

import pytensor.link.numba.dispatch.basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    default_hash_key_from_props,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.link.numba.dispatch.sparse.variable import CSCMatrixType, CSRMatrixType
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
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            return False
        return (x == y).all()
    if isinstance(x, list) or isinstance(y, list):
        if not (isinstance(x, list) and isinstance(y, list)):
            return False
        if len(x) != len(y):
            return False
        return all(numba_all_equal(xi, yi) for xi, yi in zip(x, y))
    return x == y


@numba.extending.overload(numba_all_equal)
def list_all_equal(x, y):
    all_equal = None

    if isinstance(x, List) and isinstance(y, List):

        def all_equal(x, y):
            if len(x) != len(y):
                return False
            for xi, yi in zip(x, y):
                if not numba_all_equal(xi, yi):
                    return False
            return True

    if (isinstance(x, Array) and x.ndim > 0) and (isinstance(y, Array) and y.ndim > 0):
        # (x == y).all() fails for 0d arrays
        def all_equal(x, y):
            return (x == y).all()

    if (isinstance(x, Number | Boolean) or (isinstance(x, Array) and x.ndim == 0)) and (
        isinstance(y, Number | Boolean) or (isinstance(y, Array) and y.ndim == 0)
    ):

        def all_equal(x, y):
            return x == y

    if (isinstance(x, CSRMatrixType) and isinstance(y, CSRMatrixType)) or (
        isinstance(x, CSCMatrixType) and isinstance(y, CSCMatrixType)
    ):

        def all_equal(x, y):
            return (
                x.shape == y.shape
                and (x.data == y.data).all()
                and (x.indptr == y.indptr).all()
                and (x.indices == y.indices).all()
            )

    return all_equal


@numba.extending.overload(numba_deepcopy)
def numba_deepcopy_list(x):
    if isinstance(x, List):

        def deepcopy_list(x):
            return [numba_deepcopy(xi) for xi in x]

        return deepcopy_list


def cache_key_if_not_inplace(op, inplace: bool):
    if inplace:
        # NUMBA is misbehaving with wrapped inplace ListType operations
        # which happens when we cache it in PyTensor
        # https://github.com/numba/numba/issues/10356
        return None
    else:
        return default_hash_key_from_props(op)


@register_funcify_default_op_cache_key(MakeList)
def numba_funcify_make_list(op, node, **kwargs):
    @numba_basic.numba_njit
    def make_list(*args):
        return [numba_deepcopy(arg) for arg in args]

    return make_list


@register_funcify_default_op_cache_key(Length)
def numba_funcify_list_length(op, node, **kwargs):
    @numba_basic.numba_njit
    def list_length(x):
        return np.array(len(x), dtype=np.int64)

    return list_length


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


@register_funcify_and_cache_key(Reverse)
def numba_funcify_list_reverse(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_reverse(x):
        if inplace:
            z = x
        else:
            z = numba_deepcopy(x)
        z.reverse()
        return z

    return list_reverse, cache_key_if_not_inplace(op, inplace)


@register_funcify_and_cache_key(Append)
def numba_funcify_list_append(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_append(x, to_append):
        if inplace:
            z = x
        else:
            z = numba_deepcopy(x)
        z.append(numba_deepcopy(to_append))
        return z

    return list_append, cache_key_if_not_inplace(op, inplace)


@register_funcify_and_cache_key(Extend)
def numba_funcify_list_extend(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_extend(x, to_append):
        if inplace:
            z = x
        else:
            z = numba_deepcopy(x)
        z.extend(numba_deepcopy(to_append))
        return z

    return list_extend, cache_key_if_not_inplace(op, inplace)


@register_funcify_and_cache_key(Insert)
def numba_funcify_list_insert(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_insert(x, index, to_insert):
        if inplace:
            z = x
        else:
            z = numba_deepcopy(x)
        z.insert(index.item(), numba_deepcopy(to_insert))
        return z

    return list_insert, cache_key_if_not_inplace(op, inplace)


@register_funcify_default_op_cache_key(Index)
def numba_funcify_list_index(op, node, **kwargs):
    @numba_basic.numba_njit
    def list_index(x, elem):
        for idx, xi in enumerate(x):
            if numba_all_equal(xi, elem):
                break
        return np.array(idx, dtype=np.int64)

    return list_index


@register_funcify_default_op_cache_key(Count)
def numba_funcify_list_count(op, node, **kwargs):
    @numba_basic.numba_njit
    def list_count(x, elem):
        c = 0
        for xi in x:
            if numba_all_equal(xi, elem):
                c += 1
        return np.array(c, dtype=np.int64)

    return list_count


@register_funcify_and_cache_key(Remove)
def numba_funcify_list_remove(op, node, **kwargs):
    inplace = op.inplace

    @numba_basic.numba_njit
    def list_remove(x, to_remove):
        if inplace:
            z = x
        else:
            z = numba_deepcopy(x)
        index_to_remove = -1
        for i, zi in enumerate(z):
            if numba_all_equal(zi, to_remove):
                index_to_remove = i
                break
        if index_to_remove == -1:
            raise ValueError("list.remove(x): x not in list")
        z.pop(index_to_remove)
        return z

    return list_remove, cache_key_if_not_inplace(op, inplace)
