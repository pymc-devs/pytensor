Benchmarking
============

PyTensor uses `airspeed velocity (ASV) <https://asv.readthedocs.io/>`_ for
performance benchmarking. Benchmarks are stored in the ``benchmarks/`` directory
and track performance across commits over time.

A `dashboard <https://pymc-devs.github.io/pytensor/>`_ is automatically updated
on each push to ``main``.


Quick start
-----------

Install ASV::

    pip install asv virtualenv

Or with the benchmark extra::

    pip install -e ".[benchmark]"


Running benchmarks
------------------

Run all benchmarks against your current working tree::

    asv run --python=same --quick

The ``--python=same`` flag uses your current Python environment instead of
creating a new virtual environment. The ``--quick`` flag runs each benchmark
only once for a fast (but noisier) result.

For more accurate results, drop ``--quick``::

    asv run --python=same

Run a specific benchmark module or class::

    asv run --python=same --bench bench_compile
    asv run --python=same --bench "bench_elemwise.CAReduce"

Run benchmarks matching a pattern::

    asv run --python=same --bench ".*Numba.*"


Comparing branches
------------------

Compare the current branch against ``main``::

    asv continuous --python=same main HEAD

This runs benchmarks on both commits and reports any regressions or
improvements. Use ``--factor`` to set the threshold for flagging changes::

    asv continuous --python=same --factor 1.1 main HEAD

This flags benchmarks that changed by more than 10%.


Viewing results
---------------

Generate the HTML dashboard and open it in a browser::

    asv publish
    asv preview

This starts a local web server (typically at ``http://127.0.0.1:8080``) where
you can explore benchmark results interactively.


Profiling
---------

Profile a specific benchmark to identify bottlenecks::

    asv profile bench_compile.RadonModelCall.time_call --python=same

This runs the benchmark under cProfile and displays the results.


Writing benchmarks
------------------

Benchmarks live in ``benchmarks/`` as Python files prefixed with ``bench_``.
Each file contains classes with:

- A ``setup()`` method for initialization (compilation, data generation).
  This is **not** timed.
- Methods prefixed with ``time_`` that contain **only** the code to benchmark.
- ``params`` and ``param_names`` class attributes for parametrization.

Example::

    import numpy as np
    import pytensor
    import pytensor.tensor as pt

    class MyBenchmark:
        params = [[10, 100, 1000]]
        param_names = ["size"]

        def setup(self, size):
            x = pt.vector("x", shape=(size,))
            self.fn = pytensor.function([x], pt.exp(x), trust_input=True)
            self.x_val = np.random.normal(size=size)
            self.fn(self.x_val)  # warmup / JIT compile

        def time_exp(self, size):
            self.fn(self.x_val)

For benchmarks that require optional backends (Numba, JAX), raise
``NotImplementedError`` in ``setup()`` if the backend is not available::

    def setup(self, ...):
        try:
            import numba  # noqa: F401
        except ImportError:
            raise NotImplementedError("Numba not available")

See the `ASV documentation <https://asv.readthedocs.io/en/stable/writing_benchmarks.html>`_
for more details on writing benchmarks.


CI integration
--------------

Benchmarks run automatically in GitHub Actions:

- **On push to main**: Full benchmark suite runs and results are published to
  the dashboard on GitHub Pages. Historical results are stored on the
  ``asv-results`` branch.

- **On pull requests**: ``asv continuous`` compares benchmarks between ``main``
  and the PR head. Results are posted as a PR comment, flagging any benchmarks
  that regressed by more than 10%.
