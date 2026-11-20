"""Facts about graphs that code generation and planning both use, no LLVM code:

* `layout`: what is known about the strides of arrays (`Layout`, `LayoutFeature`);
* `work`: how much work a node does (`llvm_work`), for decisions like running a loop
  in parallel;
* `liveness`: when the values of a planned graph are needed, and where its temporaries
  go in an arena.
"""
