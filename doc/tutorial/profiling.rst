
.. _tut_profiling:

===========================
Profiling PyTensor function
===========================

.. note::

    This method replaces the old `ProfileMode`. Do not use `ProfileMode`
    anymore.

Besides checking for errors, another important task is to profile your code in
terms of speed and/or memory usage.  You can profile your functions using either
of the following two options:

1. Use the PyTensor flag :attr:`config.profile` to enable profiling.
    - To enable the memory profiler use the PyTensor flag:
      :attr:`config.profile_memory` in addition to :attr:`config.profile`.
    - Moreover, to enable the profiling of PyTensor rewrite phases,
      use the PyTensor flag: :attr:`config.profile_optimizer` in addition
      to :attr:`config.profile`.
    - You can also use the PyTensor flags :attr:`profiling__n_apply`,
      :attr:`profiling__n_ops` and :attr:`profiling__min_memory_size`
      to modify the quantity of information printed.

2. Pass the argument :attr:`profile=True` to the function :func:`pytensor.function
   <function.function>` and then call :attr:`f.profile.summary()` for a single
   function.

   - Use this option when you want to profile not all the
     functions but only one or more specific function(s).
   - You can also combine the profile results of many functions:

      .. doctest::
          :hide:

          profile = pytensor.compile.ProfileStats()
          f = pytensor.function(..., profile=profile)  # doctest: +SKIP
          g = pytensor.function(..., profile=profile)  # doctest: +SKIP
          ...  # doctest: +SKIP
          profile.summary()



The profiler will output one profile per PyTensor function and produce a profile
that is the sum of the printed profiles. Each profile contains four sections:
global info, class info, `Op`\s info and `Apply` node info.

In the global section, the "Message" is the name of the PyTensor
function. :func:`pytensor.function` has an optional parameter ``name`` that
defaults to ``None``. Change it to something else to help profile many
PyTensor functions. In that section, we also see the number of times the
function was called (1) and the total time spent in all those
calls. The time spent in :meth:`Function.vm.__call__` and in thunks is useful
to understand PyTensor's overhead.

Also, we see the time spent in the two parts of the compilation process:
rewriting (i.e. modifying the graph to make it more stable/faster) and the
linking (i.e. compile C code and make the Python callable returned by
:func:`pytensor.function`).

The class, `Op`\s and `Apply` nodes sections have the same information: i.e.
information about the `Apply` nodes that ran. The `Op`\s section takes the
information from the `Apply` section and merges it with the `Apply` nodes that have
exactly the same `Op`. If two `Apply` nodes in the graph have two `Op`\s that
compare equal, they will be merged. Some `Op`\s, like `Elemwise`, will not
compare equal if their parameters differ, so the class section will merge more
`Apply` nodes than the `Op`\s section.

Note that the profile also shows which `Op`\s were run with C
implementation.

Developers wishing to optimize the performance of their graph should
focus on the worst offending `Op`\s and `Apply` nodes--either by optimizing
an implementation, providing a missing C implementation, or by writing
a graph rewrite that eliminates the offending `Op` altogether.

Here is some example output when PyTensor's rewrites are disabled. With all
rewrites enabled, there would be only one `Op` remaining in the graph.

To run the example:

    PYTENSOR_FLAGS=optimizer_excluding=fusion:inplace,profile=True python doc/tutorial/profiling_example.py

The output:

.. literalinclude:: profiling_example_out.prof

.. _numba_native_profiling:

Native sampling with Numba
==========================

Compile with ``profile=True``, warm up the function, and record representative
calls with ``f.profile.sample()``. ``summary()`` shows native hotspots and
``dprint()`` places their costs on the optimized graph:

.. literalinclude:: profiling_numba_example.py
    :language: python

The context records only the calls you make. Increase or decrease the loop
count for your function's speed; compilation must finish before recording.

The native section of ``summary()`` and the graph from this example are shown
below. Other function/compilation statistics and repeated graph metadata are
omitted; long symbol names are abbreviated. Measurements vary between runs
and machines.

.. literalinclude:: profiling_numba_example_out.txt
    :language: text

``self`` includes library work associated with one node, excluding computation
of its inputs. ``shared`` has multiple candidate nodes: its percentage appears
on each candidate, but totals count each sample once. Unassigned work stays in
the denominator. These are sampled event costs, not Op wall times. JIT address
coverage describes where instructions execute; library instructions can still
belong to graph nodes. ``graph0/3`` identifies topological node 3 of the first
graph. Nested locations append inner-graph indices.

Costs follow inner graphs, including fused scalar expressions, Scan, and
OpFromGraph, when compiled symbols or call sites identify their nodes.
Remaining loop or wrapper costs stay on the enclosing Op.

With Rich installed, ``pytensor.dprint(f, file="rich")`` as a notebook cell's
last expression displays a heat-colored tree. In a script, render it with
``rich.print()``. Colors follow self plus shared cost relative to the hottest
node. ``print_profile=False`` hides profiling information.

Setup
-----

The default Linux x86-64 backend requires perf, ``objdump``, a C compiler
(``cc``), and libdw/libelf development headers and libraries. The small native
unwinder is compiled once into PyTensor's compilation cache. Keep
``numba__cache=True`` (the default). From a checkout, install the Python
requirements and check access to the default event:

.. code-block:: bash

    python -m pip install -e '.[numba]'
    perf --version
    perf stat --all-user --event cycles:u -- python -c "sum(range(1_000_000))"
    python doc/tutorial/profiling_numba_example.py

Install perf and native development packages through your system's package
manager. Recording must permit attaching to the current process and collecting
user-space stacks; kernel or container policy can restrict this access. An
available alternative event can be selected with ``event="cpu-clock:u"``.
Defaults are ``cycles:u`` at 499 Hz, with 8 KiB user-stack snapshots.

The built-in frame-pointer backend needs no unwinder helper but can recover
less library work. Import ``LinuxPerfBackend`` from
``pytensor.link.numba.profiling`` and pass ``backend=LinuxPerfBackend()`` to
select it. Other platforms/tools can provide a backend using the
:ref:`developer interface <native_sampling_backends>`; no macOS backend is
bundled.

Saved results
-------------

By default, reports are written to a new temporary directory whose location is
printed in the summary. ``output="native_profile"`` selects a new directory
explicitly. Decoded samples, graph/source dumps, native objects, and annotated
``hot_*.asm`` files are retained. ``n_hot_functions=0`` disables assembly output.

The DWARF backend removes raw ``perf.data`` after successful decoding. To retain
it, pass ``backend=LinuxDwarfBackend(keep_raw=True)`` after importing that class
from ``pytensor.link.numba.profiling``. Saved decoded results can be rebuilt
without the raw recording:

.. code-block:: bash

    python -m pytensor.link.numba.profiling native_profile

Inlining, incomplete stacks, and worker threads can still leave costs
unassigned. ``call_graph=False`` collects flat native samples only. Keep Numba
cache files until decoding completes. ``f.profile.reset()`` clears in-memory
results without deleting saved files.
