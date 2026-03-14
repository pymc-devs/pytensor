.. _install:

Installing PyTensor
===================

The latest release of PyTensor can be installed from Pypi using `pip`:

.. code-block:: bash

    pip install pytensor


Or via conda-forge:

.. code-block:: bash

    conda install -c conda-forge pytensor


The current development branch of PyTensor can be installed from GitHub using `pip`:


.. code-block:: bash

    pip install git+https://github.com/pymc-devs/pytensor


To use the Numba and JAX backend you will need to install these libraries in addition to PyTensor. Please refer to `Numba's installation instructions <https://numba.readthedocs.io/en/stable/user/installing.html>`__ and `JAX's installation instructions  <https://github.com/google/jax#installation>`__ respectively.


Installing on Pyodide (WebAssembly)
-----------------------------------

PyTensor can be used in browser-based Python environments via `Pyodide <https://pyodide.org/>`__, such as `JupyterLite <https://jupyterlite.readthedocs.io/>`__.

PyPI does not yet accept WebAssembly wheels, so you'll need to install from a GitHub Release:

.. code-block:: python

    import micropip
    micropip.install("https://github.com/pymc-devs/pytensor/releases/download/vVERSION/pytensor-VERSION-py312-py312-pyodide_2024_0_wasm32.whl")

Replace ``VERSION`` with the desired release version (e.g., ``2.26.4``).

.. note::

   The Numba backend is not available on Pyodide/wasm32. PyTensor will use other backends automatically.
