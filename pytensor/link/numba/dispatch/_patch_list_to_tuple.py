"""Backport of numba's fixed ``peep_hole_list_to_tuple`` (numba/numba#10782).

CPython emits ``BUILD_LIST`` + ``LIST_APPEND`` per item + ``LIST_TO_TUPLE`` for
any call or tuple display past 30 items (``STACK_USE_GUIDELINE``). Before the
fix numba turned each appended item into a one-element tuple joined to an
accumulator by a binary add, so the IR held a tuple of every prefix length and
both it and the LLVM lowered from it were quadratic in the item count.
Generated fgraph functions hit this for every wide node.

``_peep_hole_list_to_tuple`` below is copied verbatim from numba ``master``
(merged 2026-08-27, only the leading underscore added) so the behaviour matches
what numba will ship, including runs of appends around starred unpacking. It is
installed only on numba < 0.68, which is where the fix ships; delete this module
once our minimum numba is at least that. The file is excluded from ruff in pyproject.toml
so it stays diffable against upstream.
"""

import operator

from numba import version_info as numba_version_info
from numba.core import interpreter, ir


def _peep_hole_list_to_tuple(func_ir):
    """
    This peephole rewrites a bytecode sequence new to Python 3.9 that looks
    like e.g.:

    def foo(a):
        return (*a,)

    41          0 BUILD_LIST               0
                2 LOAD_FAST                0 (a)
                4 LIST_EXTEND              1
                6 LIST_TO_TUPLE
                8 RETURN_VAL

    essentially, the unpacking of tuples is written as a list which is appended
    to/extended and then "magicked" into a tuple by the new LIST_TO_TUPLE
    opcode.

    This peephole repeatedly analyses the bytecode in a block looking for a
    window between a `LIST_TO_TUPLE` and `BUILD_LIST` and...

    1. Turns the BUILD_LIST into a BUILD_TUPLE
    2. Sets an accumulator's initial value as the target of the BUILD_TUPLE
    3. Searches for 'extend' on the original list and turns these into binary
       additions on the accumulator.
    4. Searches for 'append' on the original list, collecting runs of them
       into a single `BUILD_TUPLE` which is then appended via binary addition
       to the accumulator (an append-only window becomes the result itself).
    5. Assigns the accumulator to the variable that exits the peephole and the
       rest of the block/code refers to as the result of the unpack operation.
    6. Patches up

    Step 4 coalesces because CPython emits this bytecode for every call with
    more than 30 arguments, and a `BUILD_TUPLE` per item leaves a tuple of
    every prefix length, i.e. IR (and LLVM lowered from it) quadratic in the
    item count. For `f(x[0], x[1], ..., x[30])` the emitted IR used to be::

        $14build_list.2 = build_tuple(items=[])
        $20binary_subscr.5 = getitem(value=x, index=$const18.4.1)
        $24list_append.6_var = build_tuple(items=[$20binary_subscr.5])
        $24list_append.7 = $14build_list.2 + $24list_append.6_var
        $30binary_subscr.10 = getitem(value=x, index=$const28.9.2)
        $34list_append.11_var = build_tuple(items=[$30binary_subscr.10])
        $34list_append.12 = $24list_append.7 + $34list_append.11_var
        ...                    # 29 more tuples, of widths 3, 4, ..., 31
        $326call_intrinsic_1.158 = $324list_append.157
        $328call_function_ex.159 = call $4load_global.0(
            *$326call_intrinsic_1.158, vararg=$326call_intrinsic_1.158)

    and is now a single tuple, built once and passed straight to the call::

        $20binary_subscr.5 = getitem(value=x, index=$const18.4.1)
        $30binary_subscr.10 = getitem(value=x, index=$const28.9.2)
        ...
        $326call_intrinsic_1.158 = build_tuple(items=[$20binary_subscr.5,
            $30binary_subscr.10, ..., $320binary_subscr.155])
        $328call_function_ex.159 = call $4load_global.0(
            *$326call_intrinsic_1.158, vararg=$326call_intrinsic_1.158)
    """
    _DEBUG = False

    # For all blocks
    for offset, blk in func_ir.blocks.items():
        # keep doing the peephole rewrite until nothing is left that matches
        while True:
            # first try and find a matching region
            # i.e. BUILD_LIST...<stuff>...LIST_TO_TUPLE
            def find_postive_region():
                found = False
                for idx in reversed(range(len(blk.body))):
                    stmt = blk.body[idx]
                    if isinstance(stmt, ir.Assign):
                        value = stmt.value
                        if (isinstance(value, ir.Expr) and
                                value.op == 'list_to_tuple'):
                            target_list = value.info[0]
                            found = True
                            bt = (idx, stmt)
                    if found:
                        if isinstance(stmt, ir.Assign):
                            if stmt.target.name == target_list:
                                region = (bt, (idx, stmt))
                                return region

            region = find_postive_region()
            # if there's a peep hole region then do something with it
            if region is not None:
                peep_hole = blk.body[region[1][0] : region[0][0]]
                if _DEBUG:
                    print("\nWINDOW:")
                    for x in peep_hole:
                        print(x)
                    print("")

                appends = []
                extends = []
                init = region[1][1]
                const_list = init.target.name
                # Walk through the peep_hole and find things that are being
                # "extend"ed and "append"ed to the BUILD_LIST
                for x in peep_hole:
                    if isinstance(x, ir.Assign):
                        if isinstance(x.value, ir.Expr):
                            expr = x.value
                            if (expr.op == 'getattr' and
                                    expr.value.name == const_list):
                                # it's not strictly necessary to split out
                                # extends and appends, but it helps with
                                # debugging to do so!
                                if expr.attr == 'extend':
                                    extends.append(x.target.name)
                                elif expr.attr == 'append':
                                    appends.append(x.target.name)
                                else:
                                    assert 0
                # go back through the peep hole build new IR based on it.
                new_hole = []

                def append_and_fix(x):
                    """ Adds to the new_hole and fixes up definitions"""
                    new_hole.append(x)
                    if x.target.name in func_ir._definitions:
                        # if there's already a definition, drop it, should only
                        # be 1 as the way cpython emits the sequence for
                        # `list_to_tuple` should ensure this.
                        assert len(func_ir._definitions[x.target.name]) == 1
                        func_ir._definitions[x.target.name].clear()
                    func_ir._definitions[x.target.name].append(x.value)

                the_build_list = init.target

                # Buffer appended items and emit them as one build_tuple: a
                # binary add per item makes the IR, and the LLVM lowered from
                # it, quadratic in the number of items. Only an `extend` (which
                # reads the accumulator) forces a flush; item values are
                # computed by statements that stay in place, so buffering past
                # them preserves evaluation order.
                pending_appends = []

                def flush_pending_appends(acc, loc, target=None):
                    """Emit pending appends as one build_tuple. When `target`
                    is given, an append-only window is written to it as a
                    plain build_tuple (which the CALL_FUNCTION_EX peephole
                    requires for calls with kwargs); any other shape keeps
                    the assignment of the accumulator to `target`."""
                    if pending_appends:
                        items = list(pending_appends)
                        pending_appends.clear()
                        scope = items[0].scope
                        tup = ir.Expr.build_tuple(items, loc)
                        if acc is the_build_list and not init.value.items:
                            # accumulator is still the empty initial list
                            if target is not None:
                                append_and_fix(ir.Assign(tup, target, loc))
                                return target
                            acc = scope.redefine("$_list_append_tuple",
                                                 loc=loc)
                            append_and_fix(ir.Assign(tup, acc, loc))
                            return acc
                        tup_var = scope.redefine("$_list_append_tuple",
                                                 loc=loc)
                        append_and_fix(ir.Assign(tup, tup_var, loc))
                        acc_var = scope.redefine("$_list_append_acc", loc=loc)
                        append_and_fix(
                            ir.Assign(
                                ir.Expr.binop(fn=operator.add, lhs=acc,
                                              rhs=tup_var, loc=loc),
                                acc_var, loc,
                            )
                        )
                        acc = acc_var
                    if target is not None:
                        append_and_fix(ir.Assign(acc, target, loc))
                        return target
                    return acc

                # Do the transform on the peep hole
                if _DEBUG:
                    print("\nBLOCK:")
                    blk.dump()

                # This section basically accumulates list appends and extends
                # as binop(+) on tuples, it drops all the getattr() for extend
                # and append as they are now dead and replaced with binop(+).
                # It also switches out the build_list for a build_tuple and then
                # ensures everything is wired up and defined ok.
                t2l_agn = region[0][1]
                acc = the_build_list
                for x in peep_hole:
                    if isinstance(x, ir.Assign):
                        if isinstance(x.value, ir.Expr):
                            expr = x.value
                            if expr.op == 'getattr':
                                if (x.target.name in extends or
                                        x.target.name in appends):
                                    # drop definition, it's being wholesale
                                    # replaced.
                                    func_ir._definitions.pop(x.target.name)
                                    continue
                                else:
                                    # a getattr on something we're not
                                    # interested in
                                    new_hole.append(x)
                            elif expr.op == 'call':
                                fname = expr.func.name
                                if fname in extends or fname in appends:
                                    arg = expr.args[0]
                                    if (fname in appends
                                            and isinstance(arg, ir.Var)):
                                        pending_appends.append(arg)
                                        func_ir._definitions.pop(x.target.name,
                                                                 None)
                                        continue
                                    acc = flush_pending_appends(acc, x.loc)
                                    if isinstance(arg, ir.Var):
                                        tmp_name = "%s_var_%s" % (fname,
                                                                  arg.name)
                                        if fname in appends:
                                            bt = ir.Expr.build_tuple([arg,],
                                                                     expr.loc)
                                        else:
                                            # Extend as tuple
                                            gv_tuple = ir.Global(
                                                name="tuple", value=tuple,
                                                loc=expr.loc,
                                            )
                                            tuple_var = arg.scope.redefine(
                                                "$_list_extend_gv_tuple",
                                                loc=expr.loc,
                                            )
                                            new_hole.append(
                                                ir.Assign(
                                                    target=tuple_var,
                                                    value=gv_tuple,
                                                    loc=expr.loc,
                                                ),
                                            )
                                            bt = ir.Expr.call(
                                                tuple_var, (arg,), (),
                                                loc=expr.loc,
                                            )
                                        var = ir.Var(arg.scope, tmp_name,
                                                     expr.loc)
                                        asgn = ir.Assign(bt, var, expr.loc)
                                        append_and_fix(asgn)
                                        arg = var

                                    # this needs to be a binary add
                                    new = ir.Expr.binop(fn=operator.add,
                                                        lhs=acc,
                                                        rhs=arg,
                                                        loc=x.loc)
                                    asgn = ir.Assign(new, x.target, expr.loc)
                                    append_and_fix(asgn)
                                    acc = asgn.target
                                else:
                                    # there could be a call in the unpack, like
                                    # *(a, x.append(y))
                                    new_hole.append(x)
                            elif (expr.op == 'build_list' and
                                    x.target.name == const_list):
                                new = ir.Expr.build_tuple(expr.items, expr.loc)
                                asgn = ir.Assign(new, x.target, expr.loc)
                                # Not a temporary any more
                                append_and_fix(asgn)
                            else:
                                new_hole.append(x)
                        else:
                            new_hole.append(x)

                    else:
                        # stick everything else in as-is
                        new_hole.append(x)
                # Finally write the result back into the original build list
                # as everything refers to it. Flushing straight into it keeps
                # an all-append window a plain build_tuple, which the
                # CALL_FUNCTION_EX peephole requires for calls with kwargs.
                flush_pending_appends(acc, the_build_list.loc,
                                      target=t2l_agn.target)
                if _DEBUG:
                    print("\nNEW HOLE:")
                    for x in new_hole:
                        print(x)

                # and then update the block body with the modified region
                cpy = blk.body[:]
                head = cpy[:region[1][0]]
                tail = blk.body[region[0][0] + 1:]
                tmp = head + new_hole + tail
                blk.body.clear()
                blk.body.extend(tmp)

                if _DEBUG:
                    print("\nDUMP post hole:")
                    blk.dump()

            else:
                # else escape
                break

    return func_ir


# Merged after 0.67.0 was tagged, so it ships in 0.68
if numba_version_info.short < (0, 68):
    # ``Interpreter.interpret`` resolves the peephole through the module at call time
    interpreter.peep_hole_list_to_tuple = _peep_hole_list_to_tuple
