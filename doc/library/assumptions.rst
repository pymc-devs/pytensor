.. _libdoc_assumptions:

==============================================================================
:mod:`assumptions` -- Structural Assumptions and Assumption-Driven Rewrites
==============================================================================

.. module:: pytensor.assumptions
   :platform: Unix, Windows
   :synopsis: Track structural properties of tensors and let rewrites exploit them

The :mod:`pytensor.assumptions` module records structural facts about symbolic
tensors -- that a matrix is diagonal, triangular, symmetric, positive-definite --
so that graph rewrites can replace an expensive operation with a cheaper
specialized one without inserting runtime checks.

Facts are attached to ``(variable, property)`` pairs inside a
:class:`~pytensor.graph.fg.FunctionGraph`, inference is lazy and cached, and an
answer of *unknown* is both common and legitimate.

For a worked introduction, see :doc:`the assumptions gallery notebook
</gallery/rewrites/assumptions>`.

Declaring assumptions
=====================

.. autofunction:: pytensor.assumptions.assume

.. autoclass:: pytensor.assumptions.SpecifyAssumptions

Inspecting assumptions
======================

.. autofunction:: pytensor.assumptions.check_assumption

.. autoclass:: pytensor.assumptions.AssumptionFeature
   :members: get, check

.. autoclass:: pytensor.assumptions.FactState

.. autoclass:: pytensor.assumptions.ConflictingAssumptionsError

.. autofunction:: pytensor.assumptions.summarize_assumptions

.. autofunction:: pytensor.assumptions.assumption_tags

Properties
==========

Each property is an :class:`AssumptionKey`. The built-in keys are
``DIAGONAL``, ``LOWER_TRIANGULAR``, ``UPPER_TRIANGULAR``, ``SYMMETRIC``,
``POSITIVE_DEFINITE``, ``ORTHOGONAL``, ``SELECTION``, ``PERMUTATION``, and
``UNIQUE_INDICES``. ``MATRIX_KEYS`` holds the eight that describe a matrix;
``ALL_KEYS`` is a live view of every registered key, including those added by
downstream libraries.

.. autoclass:: pytensor.assumptions.AssumptionKey
   :members: assume, holds

Defining a new property
=======================

Constructing an :class:`AssumptionKey` registers it, after which
:func:`assume` accepts it by name and ``debugprint(print_assumptions=True)``
reports it. The functions below say how the new property behaves.

.. autofunction:: pytensor.assumptions.register_assumption

.. autofunction:: pytensor.assumptions.register_matrix_property_rules

.. autofunction:: pytensor.assumptions.register_universal_assumption

.. autofunction:: pytensor.assumptions.register_implies

.. autofunction:: pytensor.assumptions.register_constant_inference
