
.. _libdoc_graph_fgraph:

================================================
:mod:`fg` -- Graph Container [doc TODO]
================================================

.. module:: pytensor.graph.fg
   :platform: Unix, Windows
   :synopsis: PyTensor Internals
.. moduleauthor:: LISA


.. _fgraph:

FunctionGraph
-------------

.. autoclass:: pytensor.graph.fg.FunctionGraph
    :members:

    ***TODO***

    .. note:: FunctionGraph(inputs, outputs) clones the inputs by
        default. To avoid this behavior, add the parameter
        clone=False. This is needed as we do not want cached constants
        in fgraph.

.. _libdoc_graph_fgraphfeature:

.. _fgraphfeature:

FunctionGraph Features
----------------------

.. autoclass:: pytensor.graph.features.Feature
    :members:

.. _libdoc_graph_fgraphfeaturelist:

FunctionGraph Feature List
^^^^^^^^^^^^^^^^^^^^^^^^^^
* ReplaceValidate
* DestroyHandler
