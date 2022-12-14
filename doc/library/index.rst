
.. _libdoc:
.. _Library documentation:

=================
API Documentation
=================

This documentation covers PyTensor module-wise.  This is suited to finding the
Types and Ops that you can use to build and compile expression graphs.

Modules
=======

.. toctree::
   :maxdepth: 1

   compile/index
   config
   d3viz/index
   graph/index
   gradient
   misc/pkl_utils
   printing
   sandbox/index
   scalar/index
   scan
   sparse/index
   sparse/sandbox
   tensor/index
   typed_list

.. module:: pytensor
   :platform: Unix, Windows
   :synopsis: PyTensor top-level import
.. moduleauthor:: LISA

There are also some top-level imports that you might find more convenient:

Graph
=====

.. function:: shared(...)

   Alias for :func:`pytensor.compile.sharedvalue.shared`

.. function:: function(...)

   Alias for :func:`pytensor.compile.function.function`

.. autofunction:: pytensor.clone_replace(...)

   Alias for :func:`pytensor.graph.basic.clone_replace`

Control flow
============

.. autofunction:: pytensor.scan(...)

   Alias for :func:`pytensor.scan.basic.scan`

Convert to Variable
====================

.. autofunction:: pytensor.as_symbolic(...)

Debug
=====

.. autofunction:: pytensor.dprint(...)

   Alias for :func:`pytensor.printing.debugprint`

