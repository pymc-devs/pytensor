"""Facts about graphs that code generation and planning both use, no LLVM code:

* `layout`: the layout of every value of a graph (`graph_layouts`, `LayoutFeature`);
* `work`: how much work a node does (`llvm_work`), for decisions like running a loop
  in parallel.
"""
