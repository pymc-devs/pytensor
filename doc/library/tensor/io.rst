===================================================================
:mod:`tensor.io` --  Tensor IO Ops
===================================================================

.. module:: tensor.io
   :platform: Unix, Windows
   :synopsis: Tensor IO Ops
.. moduleauthor:: LISA

File operation
==============

- Load from disk with the function :func:`load <pytensor.tensor.io.load>` and its associated op :class:`LoadFromDisk <pytensor.tensor.io.LoadFromDisk>`

MPI operation
=============
- Non-blocking transfer: :func:`isend <pytensor.tensor.io.isend>` and :func:`irecv <pytensor.tensor.io.irecv>`.
- Blocking transfer: :func:`send <pytensor.tensor.io.send>` and :func:`recv <pytensor.tensor.io.recv>`

Details
=======

.. automodule:: pytensor.tensor.io
    :members:
