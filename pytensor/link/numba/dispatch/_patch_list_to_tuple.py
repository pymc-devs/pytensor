"""Patch ``numba.core.interpreter.peep_hole_list_to_tuple`` to emit one
``build_tuple`` instead of a chain of tuple concatenations.

Numba turns >30-item calls and tuple displays (CPython's
``STACK_USE_GUIDELINE`` bytecode) into an IR tuple of every prefix length,
which types and lowers as O(n^2) LLVM IR — generated fgraph functions make
such calls routinely. Fixed upstream in `numba#10782
<https://github.com/numba/numba/pull/10782>`_; imported for its side effect,
drop this module once a numba release ships the fix.
"""

import operator
from collections import Counter

from numba.core import interpreter, ir


_peep_hole_list_to_tuple_orig = interpreter.peep_hole_list_to_tuple


def _collapse_tuple_chains(func_ir):
    # Number of *uses* of each variable (definition sites excluded)
    uses = Counter()
    for blk in func_ir.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                vars_ = (
                    stmt.value.list_vars()
                    if isinstance(stmt.value, ir.Expr)
                    else [stmt.value]
                    if isinstance(stmt.value, ir.Var)
                    else []
                )
            else:
                vars_ = stmt.list_vars()
            for var in vars_:
                uses[var.name] += 1

    defs = func_ir._definitions
    for blk in func_ir.blocks.values():
        # var name -> (body index, items) of single-def build_tuple assignments
        tuples: dict = {}
        body = blk.body
        changed = False
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ir.Assign):
                continue
            if isinstance(stmt.value, ir.Var):
                # Forward the tuple through single-use aliases: numba's
                # CALL_FUNCTION_EX peephole requires a call's vararg to be
                # defined by a build_tuple directly when kwargs are present
                name = stmt.value.name
                if name in tuples and uses[name] == 1:
                    t_idx, t_items = tuples.pop(name)
                    new_expr = ir.Expr.build_tuple(t_items, stmt.loc)
                    defs[name].clear()
                    defs[stmt.target.name].remove(stmt.value)
                    defs[stmt.target.name].append(new_expr)
                    stmt = ir.Assign(new_expr, stmt.target, stmt.loc)
                    body[t_idx] = None
                    body[idx] = stmt
                    changed = True
                    if len(defs[stmt.target.name]) == 1:
                        tuples[stmt.target.name] = (idx, list(t_items))
                continue
            if not isinstance(stmt.value, ir.Expr):
                continue
            expr = stmt.value
            if (
                expr.op == "binop"
                and expr.fn is operator.add
                and expr.lhs.name in tuples
                and expr.rhs.name in tuples
                and uses[expr.lhs.name] == 1
                and uses[expr.rhs.name] == 1
            ):
                l_idx, l_items = tuples.pop(expr.lhs.name)
                r_idx, r_items = tuples.pop(expr.rhs.name)
                new_expr = ir.Expr.build_tuple(l_items + r_items, expr.loc)
                defs[expr.lhs.name].clear()
                defs[expr.rhs.name].clear()
                defs[stmt.target.name].remove(expr)
                defs[stmt.target.name].append(new_expr)
                stmt = ir.Assign(new_expr, stmt.target, stmt.loc)
                body[l_idx] = None
                body[r_idx] = None
                body[idx] = stmt
                expr = new_expr
                changed = True
            if expr.op == "build_tuple" and len(defs.get(stmt.target.name, ())) == 1:
                tuples[stmt.target.name] = (idx, list(expr.items))
            elif (
                expr.op == "call"
                and expr.vararg is not None
                and expr.varkwarg is None
                and not expr.args
                and expr.vararg.name in tuples
                and uses[expr.vararg.name] == 1
            ):
                # Inline the tuple back into direct call arguments: a wide
                # vararg call round-trips every argument through one wide
                # tuple, whose LLVM lowering is O(n^2) in the item count
                t_idx, t_items = tuples.pop(expr.vararg.name)
                new_call = ir.Expr.call(
                    expr.func, t_items, expr.kws, expr.loc, target=expr.target
                )
                defs[expr.vararg.name].clear()
                defs[stmt.target.name].remove(expr)
                defs[stmt.target.name].append(new_call)
                body[t_idx] = None
                body[idx] = ir.Assign(new_call, stmt.target, stmt.loc)
                changed = True
        if changed:
            new_body = [s for s in body if s is not None]
            body.clear()
            body.extend(new_body)
    return func_ir


def _peep_hole_list_to_tuple_flat(func_ir):
    return _collapse_tuple_chains(_peep_hole_list_to_tuple_orig(func_ir))


# ``Interpreter.interpret`` resolves the peephole through the module at call time
interpreter.peep_hole_list_to_tuple = _peep_hole_list_to_tuple_flat
