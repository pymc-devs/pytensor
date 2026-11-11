"""Planning a program (DESIGN.md, section 8), in stages:

1. `flatten`: the graph becomes *items* (nodes, units, subprograms) that read and
   write *values* (`values.Values`); scalars are grouped into units.
2. `resolve`: how each node is run (generated code, a Rust kernel, a numba kernel, a
   subprogram), and what that says about its outputs.
3. `memory`: where each value lives and when it is released.

`program.Planner` runs them, generates and compiles the code, and builds the program.
"""
