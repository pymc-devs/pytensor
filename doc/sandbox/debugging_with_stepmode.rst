
.. _sandbox_debugging_step_mode:

Debugging with a customized so-called StepMode
==============================================

One convenient trick I've found for debugging my programs that are running with pytensor is to
use what I call a 'StepMode'.  There is no such StepMode in the standard library because the
purpose of it is to hack it to investigate what your own particular program is doing.


.. code-block:: python

    from pytensor.link import WrapLinkerMany
    from pytensor.configdefaults import config
    from pytensor.compile.mode import (Mode, register_mode, predefined_modes, predefined_linkers,
            predefined_optimizers)

    class StepMode(Mode):
        def __init__(self, linker=None, optimizer='default'):
            if linker is None:
                linker = config.linker
            if optimizer is 'default':
                optimizer = config.optimizer
            def blah(i, node, th):
                # This function will be run for each node in your compiled program.
                # here you can inspect all the values as they are computed,
                # ... you can even change them !

                # 'i' is the execution position in the serialized graph
                # node is the symbolic Apply instance
                # th is a callable thing that will compute the node.

                print i, node, len(th.inputs)

                # the symbolic inputs of the node are in node.inputs
                # the j'th non-symbolic input of the node is in th.inputs[j][0]

                th() # call the function to actually 'run' the graph

                # the symbolic outputs of the node are in node.outputs
                # the j'th non-symbolic output of the node is in th.outputs[j][0]

                print type(th.outputs[0][0])

                if i == 39:
                    print 'this node is weird...', th.outputs[0][0]


            self.provided_linker = linker
            self.provided_optimizer = optimizer
            if isinstance(linker, basestring) or linker is None:
                linker = predefined_linkers[linker]

            self.linker = WrapLinkerMany([linker], [blah])

            if isinstance(optimizer, basestring) or optimizer is None:
                optimizer = predefined_optimizers[optimizer]
            self._optimizer = optimizer



The way to use it is like this:

.. code-block:: python

    fn = function(inputs, outputs, mode=StepMode())

When you call fn, your function in the stepmode will be called for each node in the compiled
program.  You can print out some or all of the values, you can change them in mid-execution.
You can see where bizarre values are first occurring in your computations.  It's a very
powerful way to understand your program's execution.

Remember, if you give names your variables then printing nodes will give you a better idea of
where in the calculations you are.
