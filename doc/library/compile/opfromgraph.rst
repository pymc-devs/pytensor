:orphan:

.. _opfromgraph:

=============
`OpFromGraph`
=============

This page describes :class:`pytensor.compile.builders.OpFromGraph
<pytensor.compile.builders.OpFromGraph>`, an `Op` constructor that allows one to
encapsulate an PyTensor graph in a single `Op`.

This can be used to encapsulate some functionality in one block. It is
useful to scale PyTensor compilation for regular bigger graphs when we
reuse that encapsulated functionality with different inputs many
times. Due to this encapsulation, it can make PyTensor's compilation phase
faster for graphs with many nodes.

Using this for small graphs is not recommended as it disables
rewrites between what is inside the encapsulation and outside of it.

.. note:

    This was not used widely up to now. If you have any
    questions/comments do not hesitate to contact us on the mailing list.



.. autoclass:: pytensor.compile.builders.OpFromGraph
