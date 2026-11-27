Native profiling backends
=========================

The Numba profiler shares graph attribution, self/shared accounting, and display
across native tools. Backends own object parsing, recording, stack recovery, and
assembly annotation. The Linux x86-64 default uses perf and libdwfl, with the
register/frame and call-site analysis isolated in ``_profiling_dwarf.py``.
``_profiling_unwind.c`` reads only the captured stack bytes during unwinding.
Its shared library is built once in the PyTensor compilation cache.

Graph locations descend through named inner graphs and scalar/core wrappers,
using the same graph boundaries as ``dprint``. Attribution prefers inner nodes
over their containers. Library-call candidates are restricted to the recovered
caller's graph region; reused inner implementations retain all candidate
locations. Samples without finer evidence remain on their enclosing node.

.. _native_sampling_backends:

Adapting another native tool
----------------------------

Pass an instance implementing
``pytensor.link.numba.profiling.NativeSamplingBackend`` to the same context:

.. code-block:: python

    # native_tool_backend is supplied by an adapter for your platform/tool.
    with f.profile.sample(backend=native_tool_backend, n_hot_functions=0):
        f(values)

The built-in backends are ``linux-perf-dwarf`` (the Linux x86-64 default) and
``linux-perf`` (frame pointers). There is currently no bundled macOS/ARM
backend. An adapter can use another object format and profiling tool without
changing graph attribution, ``summary()``, or ``dprint()``. Subclassing
is not required; an object with the following interface is sufficient:

.. list-table:: Backend contract
    :header-rows: 1
    :widths: 24 76

    * - Member
      - Responsibility
    * - ``name``, ``default_event``, ``call_graph``
      - Stable backend identifier, default event name, and caller-collection label.
    * - ``validate(event=..., n_hot_functions=..., call_graph=...)``
      - Check this backend's platform, tool access, and supported options before
        creating output or entering the workload.
    * - ``functions(blob)``
      - Read the backend's native object format. Return records with ``name``,
        ``size``, ``offset``, and ``section``. Names must resolve through Numba's
        code library; sizes and offsets describe the defining object function.
    * - ``external_symbols(blob)``
      - Return imported symbol names used for library-call candidate matching.
    * - ``recorder(output, frequency, event, call_graph)``
      - Return an object with ``start()``, ``stop()``, and ``close()``. Start and
        stop delimit the caller's workload; close releases resources on success
        or failure. The recorder must not call or warm the profiled function.
    * - ``decode(output, call_graph=...)``
      - Convert the tool's recording to the normalized files described below.
        Symbol/node metadata is already saved in ``output``. Return optional
        recording metadata as a dictionary, or ``None``.
    * - ``annotate(report, output, n_hot_functions)``
      - Optionally add assembly artifacts and report metadata. Reject unsupported
        assembly requests in ``validate``; zero disables assembly output.

``samples.txt`` contains one sample per nonempty line:

.. code-block:: text

    40 7fff0100 library_function (library_name)

The fields are a nonnegative integer event weight, a normalized absolute
instruction address in hexadecimal (without ``0x``), a symbol name, and a
parenthesized library identifier. Weights can represent event periods or one
per sample; use one event and consistent units throughout the recording.

When callers are requested, ``callchains.txt`` contains matching blocks in the
same sample order, with the same weights:

.. code-block:: text

    40
        7fff0100 library_function (library_name)
        7fff0205 jit_caller ([jit])

Separate blocks with blank lines. The first frame must match the sampled
instruction address; following frames contain caller return addresses, with
architecture-specific pointer normalization handled by the backend. The core
looks up the instruction immediately before each return address. Emit a
leaf-only block when a stack cannot be recovered. Broken or missing stacks
must not remove flat samples. Adapters do not need to emit or read perf's
binary format. Normalize symbol names consistently between the object reader
and decoded samples so library imports can be matched to stack frames.

The core saves a backend's identity in ``report.json`` and does not import
adapter code from saved files. To rebuild a custom backend's report, provide
the matching instance explicitly:

.. code-block:: python

    from pytensor.link.numba.profiling import rebuild_report

    rebuild_report("native_profile", backend=native_tool_backend)

This contract is new. Architecture-specific object parsing, unwinding, and
instruction analysis belong in the adapter; the common layer handles graph
locations, candidate accounting, self/shared costs, and rendering.

An optional ``call_sites.json`` maps caller return addresses (decimal JSON keys)
to verified imported function names. It lets a backend recover library callers
whose internal symbol names differ from their imports. The common attribution
layer still uses the compiled candidate sets, including opaque callers; it does
not match Op names against library names.

The DWARF backend uses the recorded user registers and stack snapshots, retaining
the first JIT caller even when it has no unwind table. For supported leaf-frame
layouts, it verifies the stack adjustment, saved registers, and return instruction
before resuming unwinding. Reaching imported call targets are tracked through
control flow; unknown or conflicting targets stay unassigned. Unsupported frame
layouts stay visible as native hotspots. No graph code is instrumented.
