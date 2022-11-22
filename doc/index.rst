
Welcome
=======

PyTensor is a Python library that allows you to define, optimize/rewrite, and
evaluate mathematical expressions involving multi-dimensional arrays
efficiently.

Some of PyTensor's features are:

* **Tight integration with NumPy**
  - Use `numpy.ndarray` in PyTensor-compiled functions
* **Efficient symbolic differentiation**
  - PyTensor efficiently computes your derivatives for functions with one or many inputs
* **Speed and stability optimizations**
  - Get the right answer for ``log(1 + x)`` even when ``x`` is near zero
* **Dynamic C/JAX/Numba code generation**
  - Evaluate expressions faster

PyTensor is based on `Theano`_, which has been powering large-scale computationally
intensive scientific investigations since 2007.


.. warning::

   Much of the documentation hasn't been updated and is simply the old Theano documentation.

Download
========

PyTensor is `available on PyPI`_, and can be installed via ``pip install PyTensor``.

Those interested in bleeding-edge features should obtain the latest development
version, available via::

    git clone git://github.com/pymc-devs/pytensor.git

You can then place the checkout directory on your ``$PYTHONPATH`` or use
``python setup.py develop`` to install a ``.pth`` into your ``site-packages``
directory, so that when you pull updates via Git, they will be
automatically reflected the "installed" version. For more information about
installation and configuration, see :ref:`installing PyTensor <install>`.

.. _available on PyPI: http://pypi.python.org/pypi/pytensor
.. _Related Projects: https://github.com/pymc-devs/pytensor/wiki/Related-projects

Documentation
=============

Roughly in order of what you'll want to check out:

* :ref:`install` -- How to install PyTensor.
* :ref:`introduction` -- What is PyTensor?
* :ref:`tutorial` -- Learn the basics.
* :ref:`troubleshooting` -- Tips and tricks for common debugging.
* :ref:`libdoc` -- PyTensor's functionality, module by module.
* :ref:`faq` -- A set of commonly asked questions.
* :ref:`optimizations` -- Guide to PyTensor's graph optimizations.
* :ref:`extending` -- Learn to add a Type, Op, or graph optimization.
* :ref:`dev_start_guide` -- How to contribute code to PyTensor.
* :ref:`internal` -- How to maintain PyTensor and more...
* :ref:`acknowledgement` -- What we took from other projects.
* `Related Projects`_ -- link to other projects that implement new functionalities on top of PyTensor


.. _pytensor-community:

Community
=========

* Visit `pytensor-users`_ to discuss the general use of PyTensor with developers and other users
* We use `GitHub issues <http://github.com/pymc-devs/pytensor/issues>`__ to
  keep track of issues and `GitHub Discussions <https://github.com/pymc-devs/pytensor/discussions>`__ to discuss feature
  additions and design changes

.. toctree::
   :maxdepth: 1
   :hidden:

   introduction
   user_guide
   API <library/index>
   Contributing <dev_start_guide>

.. _Theano: https://github.com/Theano/Theano
.. _pytensor-users: https://gitter.im/pymc-devs/pytensor
