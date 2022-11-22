.. _libdoc_typed_list:

===============================
:mod:`typed_list` -- Typed List
===============================

.. note::

    This has been added in release 0.7.

.. note::

    This works, but is not well integrated with the rest of PyTensor. If
    speed is important, it is probably better to pad to a dense
    tensor.

This is a type that represents a list in PyTensor. All elements must have
the same PyTensor type. Here is an example:

>>> import pytensor.typed_list
>>> tl = pytensor.typed_list.TypedListType(pytensor.tensor.fvector)()
>>> v = pytensor.tensor.fvector()
>>> o = pytensor.typed_list.append(tl, v)
>>> f = pytensor.function([tl, v], o)
>>> f([[1, 2, 3], [4, 5]], [2])
[array([ 1.,  2.,  3.], dtype=float32), array([ 4.,  5.], dtype=float32), array([ 2.], dtype=float32)]

A second example with Scan. Scan doesn't yet have direct support of
TypedList, so you can only use it as non_sequences (not in sequences or
as outputs):

>>> import pytensor.typed_list
>>> a = pytensor.typed_list.TypedListType(pytensor.tensor.fvector)()
>>> l = pytensor.typed_list.length(a)
>>> s, _ = pytensor.scan(fn=lambda i, tl: tl[i].sum(),
...                    non_sequences=[a],
...                    sequences=[pytensor.tensor.arange(l, dtype='int64')])
>>> f = pytensor.function([a], s)
>>> f([[1, 2, 3], [4, 5]])
array([ 6.,  9.], dtype=float32)

.. automodule:: pytensor.typed_list.basic
    :members:
