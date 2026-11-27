"""Deciding how a graph runs, before any code is generated.

The planner decides for every node what runs it (generated code, a Rust kernel, a numba
kernel, or generated code in a module of its own), how its function is called, and
where the values live. Code generation then follows these decisions and makes none of
its own. For example, an Elemwise of the program without a Rust kernel becomes a `Step`
whose implementation is an `OwnModule`, with the `Signature` of its node function; the
memory plan puts its output into the program's frame if its size is fixed.

* `planner`: the decisions (`Planner`), for the program's graph and for the inner graphs
  generated code runs;
* `plan`: what the planner hands to code generation (`Step`, `PlannedGraph`);
* `implementation`: what runs a node (`Implementation`), what it declares about its
  outputs, and how it writes the node function's body;
* `signature`: how a node's function is called (`Signature`, `node_outputs`);
* `memory`: the buffers the values of a planned graph live in, how long each is
  needed, and where the buffers of a fixed size go in the graph's frames.
"""

# Planning asks the registry about ops, and the ops' implementations import modules of
# this package. Importing the registry first, which registers them all, lets them find
# these modules complete whichever module is imported first.
import pytensor.link.llvm.registry
