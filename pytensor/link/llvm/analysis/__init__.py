"""Facts about graphs that code generation and planning both use, no LLVM code:

* `layout`: what is known about the strides of arrays (`Layout`, `LayoutFeature`);
* `work`: how much work a node does (`llvm_work`), for decisions like running a loop
  in parallel;
* `memory`: the buffers the values of a planned graph live in, how long each is
  needed, and where the buffers of a fixed size go in the graph's frames.
"""
