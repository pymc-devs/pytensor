from collections.abc import Sequence
from textwrap import dedent, indent

from pytensor.configdefaults import config


def make_declare(loop_orders, dtypes, sub, compute_stride_jump=True):
    """
    Produce code to declare all necessary variables.

    """
    decl = ""
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes, strict=True)):
        var = sub[f"lv{i}"]  # input name corresponding to ith loop variable
        # we declare an iteration variable
        # and an integer for the number of dimensions
        decl += f"{dtype}* {var}_iter;\n"
        for j, value in enumerate(loop_order):
            if value != "x":
                # If the dimension is not broadcasted, we declare
                # the number of elements in that dimension,
                # the stride in that dimension,
                # and the jump from an iteration to the next
                decl += f"npy_intp {var}_n{value};\nnpy_intp {var}_stride{value};\n"
                if compute_stride_jump:
                    decl += f"npy_intp {var}_jump{value}_{j};\n"

            elif compute_stride_jump:
                # if the dimension is broadcasted, we only need
                # the jump (arbitrary length and stride = 0)
                decl += f"npy_intp {var}_jump{value}_{j};\n"

    return decl


def make_checks(loop_orders, dtypes, sub, compute_stride_jump=True):
    init = ""
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes, strict=True)):
        var = sub[f"lv{i}"]
        # List of dimensions of var that are not broadcasted
        nonx = [x for x in loop_order if x != "x"]
        if nonx:
            # If there are dimensions that are not broadcasted
            # this is a check that the number of dimensions of the
            # tensor is as expected.
            min_nd = max(nonx) + 1
            init += dedent(
                f"""
                if (PyArray_NDIM({var}) < {min_nd}) {{
                    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                    {indent(sub["fail"], " " * 12)}
                }}
                """
            )

        # In loop j, adjust represents the difference of values of the
        # data pointer between the beginning and the end of the
        # execution of loop j+1 (the loop inside the current one). It
        # is equal to the stride in loop j+1 times the length of loop
        # j+1, or 0 for the inner-most loop.
        adjust = "0"

        # We go from the inner loop to the outer loop
        for j, index in reversed(list(enumerate(loop_order))):
            if index != "x":
                # Initialize the variables associated to the jth loop
                # jump = stride - adjust
                jump = f"({var}_stride{index}) - ({adjust})"
                init += f"{var}_n{index} = PyArray_DIMS({var})[{index}];\n"
                init += f"{var}_stride{index} = PyArray_STRIDES({var})[{index}] / sizeof({dtype});\n"
                if compute_stride_jump:
                    init += f"{var}_jump{index}_{j} = {jump};\n"
                adjust = f"{var}_n{index}*{var}_stride{index}"

            elif compute_stride_jump:
                jump = f"-({adjust})"
                init += f"{var}_jump{index}_{j} = {jump};\n"
                adjust = "0"
    check = ""

    # This loop builds multiple if conditions to verify that the
    # dimensions of the inputs match, and the first one that is true
    # raises an informative error message

    runtime_broadcast_error_msg = (
        "Runtime broadcasting not allowed. "
        "One input had a distinct dimension length of 1, but was not marked as broadcastable: "
        "(input[%i].shape[%i] = %lld, input[%i].shape[%i] = %lld). "
        "If broadcasting was intended, use `specify_broadcastable` on the relevant input."
    )

    for matches in zip(*loop_orders, strict=True):
        to_compare = [(j, x) for j, x in enumerate(matches) if x != "x"]

        # elements of to_compare are pairs ( input_variable_idx, input_variable_dim_idx )
        if len(to_compare) < 2:
            continue

        j0, x0 = to_compare[0]
        for j, x in to_compare[1:]:
            check += dedent(
                f"""
                if ({sub[f"lv{j0}"]}_n{x0} != {sub[f"lv{j}"]}_n{x})
                {{
                    if ({sub[f"lv{j0}"]}_n{x0} == 1 || {sub[f"lv{j}"]}_n{x} == 1)
                    {{
                        PyErr_Format(PyExc_ValueError, "{runtime_broadcast_error_msg}",
                       {j0},
                       {x0},
                       (long long int) {sub[f"lv{j0}"]}_n{x0},
                       {j},
                       {x},
                       (long long int) {sub[f"lv{j}"]}_n{x}
                        );
                    }} else {{
                        PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%i].shape[%i] = %lld, input[%i].shape[%i] = %lld)",
                           {j0},
                           {x0},
                           (long long int) {sub[f"lv{j0}"]}_n{x0},
                           {j},
                           {x},
                           (long long int) {sub[f"lv{j}"]}_n{x}
                        );
                    }}
                    {sub["fail"]}
                }}
            """
            )

    return init + check


def compute_output_dims_lengths(array_name: str, loop_orders, sub) -> str:
    """Create c_code to compute the output dimensions of an Elemwise operation.

    The code returned by this function populates the array `array_name`, but does not
    initialize it.

    Note: We could specialize C code even further with the known static output shapes
    """
    dims_c_code = ""
    for i, candidates in enumerate(zip(*loop_orders, strict=True)):
        # Borrow the length of the first non-broadcastable input dimension
        for j, candidate in enumerate(candidates):
            if candidate != "x":
                var = sub[f"lv{j}"]
                dims_c_code += f"{array_name}[{i}] = {var}_n{candidate};\n"
                break
        # If none is non-broadcastable, the output dimension has a length of 1
        else:  # no-break
            dims_c_code += f"{array_name}[{i}] = 1;\n"

    return dims_c_code


def make_alloc(loop_orders, dtype, sub, fortran="0"):
    """Generate C code to allocate outputs.

    Parameters
    ----------
    fortran : str
        A string included in the generated code. If it
        evaluate to non-zero, an ndarray in fortran order will be
        created, otherwise it will be c order.

    """
    type = dtype.upper()
    if type.startswith("PYTENSOR_COMPLEX"):
        type = type.replace("PYTENSOR_COMPLEX", "NPY_COMPLEX")
    nd = len(loop_orders[0])
    init_dims = compute_output_dims_lengths("dims", loop_orders, sub)
    olv = sub["olv"]
    fail = sub["fail"]

    # TODO: it would be interesting to allocate the output in such a
    # way that its contiguous dimensions match one of the input's
    # contiguous dimensions, or the dimension with the smallest
    # stride. Right now, it is allocated to be C_CONTIGUOUS.
    return dedent(
        f"""
        {{
            npy_intp dims[{nd}];
            {init_dims}
            if (!{olv}) {{
                {olv} = (PyArrayObject*)PyArray_EMPTY({nd},
                                                      dims,
                                                      {type},
                                                      {fortran});
            }}
            else {{
                PyArray_Dims new_dims;
                new_dims.len = {nd};
                new_dims.ptr = dims;
                PyObject* success = PyArray_Resize({olv}, &new_dims, 0, NPY_CORDER);
                if (!success) {{
                    // If we can't resize the ndarray we have we can allocate a new one.
                    PyErr_Clear();
                    Py_XDECREF({olv});
                    {olv} = (PyArrayObject*)PyArray_EMPTY({nd}, dims, {type}, 0);
                }} else {{
                    Py_DECREF(success);
                }}
            }}
            if (!{olv}) {{
                {fail}
            }}
        }}
        """
    )


def make_loop(
    loop_orders: list[tuple[int | str, ...]],
    dtypes: list,
    loop_tasks: list,
    sub: dict[str, str],
    openmp: bool = False,
):
    """
    Make a nested loop over several arrays and associate specific code
    to each level of nesting.

    Parameters
    ----------
    loop_orders : list of N tuples of length M
        Each value of each tuple can be either the index of a dimension to
        loop over or the letter 'x' which means there is no looping to be done
        over that variable at that point (in other words we broadcast
        over that dimension). If an entry is an integer, it will become
        an alias of the entry of that rank.
    loop_tasks : list of M+1 pieces of code
        The ith loop_task is a pair of strings, the first
        string is code to be executed before the ith loop starts, the second
        one contains code to be executed just before going to the next element
        of the ith dimension.
        The last element of loop_tasks is a single string, containing code
        to be executed at the very end.
    sub : dictionary
        Maps 'lv#' to a suitable variable name.
        The 'lvi' variable corresponds to the ith element of loop_orders.

    """

    def loop_over(preloop, code, indices, i):
        iterv = f"ITER_{i}"
        update = ""
        suitable_n = "1"
        for j, index in enumerate(indices):
            var = sub[f"lv{j}"]
            dtype = dtypes[j]
            update += f"{dtype} &{var}_i = * ( {var}_iter + {iterv} * {var}_jump{index}_{i} );\n"

            if index != "x":
                suitable_n = f"{var}_n{index}"
        if openmp:
            openmp_elemwise_minsize = config.openmp_elemwise_minsize
            forloop = f"""#pragma omp parallel for if( {suitable_n} >={openmp_elemwise_minsize})\n"""
        else:
            forloop = ""
        forloop += f"""for (npy_intp {iterv} = 0; {iterv}<{suitable_n}; {iterv}++)"""
        return f"""
        {preloop}
        {forloop} {{
            {update}
            {code}
        }}
        """

    preloops: dict[int, str] = {}
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes, strict=True)):
        for j, index in enumerate(loop_order):
            if index != "x":
                preloops.setdefault(j, "")
                preloops[j] += (
                    f"%(lv{i})s_iter = ({dtype}*)(PyArray_DATA(%(lv{i})s));\n"
                ) % sub
                break
        else:  # all broadcastable
            preloops.setdefault(0, "")
            preloops[0] += (
                f"%(lv{i})s_iter = ({dtype}*)(PyArray_DATA(%(lv{i})s));\n"
            ) % sub

    s = ""

    tasks_indices = zip(loop_tasks[:-1], zip(*loop_orders, strict=True), strict=True)
    for i, ((pre_task, task), indices) in reversed(list(enumerate(tasks_indices))):
        s = loop_over(preloops.get(i, "") + pre_task, s + task, indices, i)

    s += loop_tasks[-1]
    return f"{{{s}}}"


def make_reordered_loop(
    init_loop_orders, olv_index, dtypes, inner_task, sub, openmp=None
):
    """A bit like make_loop, but when only the inner-most loop executes code.

    All the loops will be reordered so that the loops over the output tensor
    are executed with memory access as contiguous as possible.
    For instance, if the output tensor is c_contiguous, the inner-most loop
    will be on its rows; if it's f_contiguous, it will be on its columns.

    The output tensor's index among the loop variables is indicated by olv_index.

    """

    # Number of variables
    nvars = len(init_loop_orders)
    # Number of loops (dimensionality of the variables)
    nnested = len(init_loop_orders[0])

    # This is the var from which we'll get the loop order
    ovar = sub[f"lv{olv_index}"]

    # The loops are ordered by (decreasing) absolute values of ovar's strides.
    # The first element of each pair is the absolute value of the stride
    # The second element correspond to the index in the initial loop order
    order_loops = f"""
    std::vector< std::pair<int, npy_intp> > {ovar}_loops({nnested});
    std::vector< std::pair<int, npy_intp> >::iterator {ovar}_loops_it = {ovar}_loops.begin();
    """

    # Fill the loop vector with the appropriate <stride, index> pairs
    for i, index in enumerate(init_loop_orders[olv_index]):
        if index != "x":
            order_loops += f"""
            {ovar}_loops_it->first = abs(PyArray_STRIDES({ovar})[{index}]);
            """
        else:
            # Stride is 0 when dimension is broadcastable
            order_loops += f"""
            {ovar}_loops_it->first = 0;
            """

        order_loops += f"""
        {ovar}_loops_it->second = {i};
        ++{ovar}_loops_it;
        """

    # We sort in decreasing order so that the outermost loop (loop 0)
    # has the largest stride, and the innermost loop (nnested - 1) has
    # the smallest stride.
    order_loops += f"""
    // rbegin and rend are reversed iterators, so this sorts in decreasing order
    std::sort({ovar}_loops.rbegin(), {ovar}_loops.rend());
    """

    # Get the (sorted) total number of iterations of each loop
    declare_totals = f"npy_intp init_totals[{nnested}];\n"
    declare_totals += compute_output_dims_lengths("init_totals", init_loop_orders, sub)

    # Sort totals to match the new order that was computed by sorting
    # the loop vector. One integer variable per loop is declared.
    declare_totals += f"""
    {ovar}_loops_it = {ovar}_loops.begin();
    """

    for i in range(nnested):
        declare_totals += f"""
        npy_intp TOTAL_{i} = init_totals[{ovar}_loops_it->second];
        ++{ovar}_loops_it;
        """

    # Get sorted strides
    # Get strides in the initial order
    def get_loop_strides(loop_order, i):
        """
        Returns a list containing a C expression representing the
        stride for each dimension of the ith variable, in the
        specified loop_order.

        """
        var = sub[f"lv{i}"]
        r = []
        for index in loop_order:
            # Note: the stride variable is not declared for broadcasted variables
            if index != "x":
                r.append(f"{var}_stride{index}")
            else:
                r.append("0")
        return r

    # We declare the initial strides as a 2D array, nvars x nnested
    strides = ", \n".join(
        ", ".join(get_loop_strides(lo, i))
        for i, lo in enumerate(init_loop_orders)
        if len(lo) > 0
    )

    declare_strides = f"""
    npy_intp init_strides[{nvars}][{nnested}] = {{
        {strides}
    }};"""

    # Declare (sorted) stride and for each variable
    # we iterate from innermost loop to outermost loop
    declare_strides += f"""
    std::vector< std::pair<int, npy_intp> >::reverse_iterator {ovar}_loops_rit;
    """

    for i in range(nvars):
        var = sub[f"lv{i}"]
        declare_strides += f"""
        {ovar}_loops_rit = {ovar}_loops.rbegin();"""
        for j in reversed(range(nnested)):
            declare_strides += f"""
            npy_intp {var}_stride_l{j} = init_strides[{i}][{ovar}_loops_rit->second];
            ++{ovar}_loops_rit;
            """

    declare_iter = ""
    for i, dtype in enumerate(dtypes):
        var = sub[f"lv{i}"]
        declare_iter += f"{var}_iter = ({dtype}*)(PyArray_DATA({var}));\n"

    pointer_update = ""
    for j, dtype in enumerate(dtypes):
        var = sub[f"lv{j}"]
        pointer_update += f"{dtype} &{var}_i = * ( {var}_iter"
        for i in reversed(range(nnested)):
            iterv = f"ITER_{i}"
            pointer_update += f"+{var}_stride_l{i}*{iterv}"
        pointer_update += ");\n"

    loop = inner_task
    for i in reversed(range(nnested)):
        iterv = f"ITER_{i}"
        total = f"TOTAL_{i}"
        update = ""
        forloop = ""
        # The pointers are defined only in the most inner loop
        if i == nnested - 1:
            update = pointer_update
        if i == 0:
            if openmp:
                openmp_elemwise_minsize = config.openmp_elemwise_minsize
                forloop += f"""#pragma omp parallel for if( {total} >={openmp_elemwise_minsize})\n"""
        forloop += f"for(npy_intp {iterv} = 0; {iterv}<{total}; {iterv}++)"

        loop = f"""
        {forloop}
        {{ // begin loop {i}
            {update}
            {loop}
        }} // end loop {i}
        """

    code = "\n".join((order_loops, declare_totals, declare_strides, declare_iter, loop))
    return f"{{\n{code}\n}}\n"


##################
#   DimShuffle   #
##################

#################
#   Broadcast   #
#################


################
#   CAReduce   #
################


def make_complete_loop_careduce(
    inp_var: str,
    acc_var: str,
    inp_dtype: str,
    acc_dtype: str,
    initial_value: str,
    inner_task: str,
    fail_code,
) -> str:
    """Generate C code for a complete reduction loop.

    The generated code for a float64 input variable `inp` and accumulation variable `acc` looks like:

    .. code-block:: C
        {
            NpyIter* iter;
            NpyIter_IterNextFunc *iternext;
            char** data_ptr;
            npy_intp* stride_ptr,* innersize_ptr;

            // Special case for empty inputs
            if (PyArray_SIZE(inp) == 0) {
                npy_float64 acc_i = *(npy_float64*)(PyArray_DATA(acc));
                acc_i = 0;
            }else{
                iter = NpyIter_New(inp,
                                   NPY_ITER_READONLY| NPY_ITER_EXTERNAL_LOOP| NPY_ITER_REFS_OK,
                                   NPY_KEEPORDER,
                                   NPY_NO_CASTING,
                                   NULL);
                iternext = NpyIter_GetIterNext(iter, NULL);
                if (iternext == NULL) {
                    NpyIter_Deallocate(iter);
                    { fail }
                }
                data_ptr = NpyIter_GetDataPtrArray(iter);
                stride_ptr = NpyIter_GetInnerStrideArray(iter);
                innersize_ptr = NpyIter_GetInnerLoopSizePtr(iter);

                npy_float64 acc_i;
                acc_i = 0;
                do {
                    char* data = *data_ptr;
                    npy_intp stride = *stride_ptr;
                    npy_intp count = *innersize_ptr;

                    while(count--) {
                        npy_float64 inp_i = *((npy_float64*)data);
                        acc_i = acc_i + inp_i;
                        data += stride;
                    }

                } while(iternext(iter));
                NpyIter_Deallocate(iter);

                *(npy_float64*)(PyArray_DATA(acc)) = acc_i;
            }
        }
    """
    return dedent(
        f"""
        {{
            NpyIter* iter;
            NpyIter_IterNextFunc *iternext;
            char** data_ptr;
            npy_intp* stride_ptr,* innersize_ptr;

            // Special case for empty inputs
            if (PyArray_SIZE({inp_var}) == 0) {{
                {acc_dtype} &{acc_var}_i = *({acc_dtype}*)(PyArray_DATA({acc_var}));
                {initial_value}
            }}else{{
                iter = NpyIter_New({inp_var},
                                   NPY_ITER_READONLY| NPY_ITER_EXTERNAL_LOOP| NPY_ITER_REFS_OK,
                                   NPY_KEEPORDER,
                                   NPY_NO_CASTING,
                                   NULL);

                iternext = NpyIter_GetIterNext(iter, NULL);
                if (iternext == NULL) {{
                    NpyIter_Deallocate(iter);
                    {fail_code}
                }}

                data_ptr = NpyIter_GetDataPtrArray(iter);
                stride_ptr = NpyIter_GetInnerStrideArray(iter);
                innersize_ptr = NpyIter_GetInnerLoopSizePtr(iter);

                {acc_dtype} {acc_var}_i;
                {initial_value}

                do {{
                    char* data = *data_ptr;
                    npy_intp stride = *stride_ptr;
                    npy_intp count = *innersize_ptr;

                    while(count--) {{
                        {inp_dtype} {inp_var}_i = *(({inp_dtype}*)data);
                        {inner_task}
                        data += stride;
                    }}
                }} while(iternext(iter));

                NpyIter_Deallocate(iter);
                *({acc_dtype}*)(PyArray_DATA({acc_var})) = {acc_var}_i;
            }}
        }}
        """
    )


def make_reordered_loop_careduce(
    inp_var: str,
    acc_var: str,
    inp_dtype: str,
    acc_dtype: str,
    inp_ndim: int,
    reduction_axes: Sequence[int],
    initial_value: str,
    inner_task: str,
) -> str:
    """Generate C code for a partial reduction loop, reordering for optimal memory access of the input variable.

    The generated code for a sum along the last axis of a 2D float64 input variable `inp`
    in an accumulation variable `acc` looks like:

    .. code-block:: C
        {
            // Special case for empty inputs
            if (PyArray_SIZE(inp) == 0) {
                acc_iter = (npy_float64*)(PyArray_DATA(acc));
                int_n =  PyArray_SIZE(acc);
                for(npy_intp i = 0; i < n; i++)
                {
                    npy_float64 &acc_i = acc_iter[i];
                    acc_i = 0;
                }
            } else {
            std::vector< std::pair<int, npy_intp> > loops(2);
            std::vector< std::pair<int, npy_intp> >::iterator loops_it = loops.begin();

            loops_it->first = abs(PyArray_STRIDES(inp)[0]);
            loops_it->second = 0;
            ++loops_it;
            loops_it->first = abs(PyArray_STRIDES(inp)[1]);
            loops_it->second = 1;
            ++loops_it;
            std::sort(loops.rbegin(), loops.rend());

            npy_intp dim_lengths[2] = {inp_n0, inp_n1};
            npy_intp inp_strides[2] = {inp_stride0, inp_stride1};
            npy_intp acc_strides[2] = {acc_stride0, 0};
            bool reduction_axes[2] = {0, 1};

            loops_it = loops.begin();
            npy_intp dim_length_0 = dim_lengths[loops_it->second];
            bool is_reduction_axis_0 = reduction_axes[loops_it->second];
            npy_intp inp_stride_0 = inp_strides[loops_it->second];
            npy_intp acc_stride_0 = acc_strides[loops_it->second];
            ++loops_it;
            npy_intp dim_length_1 = dim_lengths[loops_it->second];
            bool is_reduction_axis_1 = reduction_axes[loops_it->second];
            npy_intp inp_stride_1 = inp_strides[loops_it->second];
            npy_intp acc_stride_1 = acc_strides[loops_it->second];
            ++loops_it;

            inp_iter = (npy_float64*)(PyArray_DATA(inp));
            acc_iter = (npy_float64*)(PyArray_DATA(acc));

            for(npy_intp iter_0 = 0; iter_0<dim_length_0; iter_0++){
                for(npy_intp iter_1 = 0; iter_1<dim_length_1; iter_1++){
                    npy_float64 &inp_i = *(inp_iter + inp_stride_1*iter_1 + inp_stride_0*iter_0);
                    npy_float64 &acc_i = *(acc_iter + acc_stride_1*iter_1 + acc_stride_0*iter_0);

                    if((!is_reduction_axis_0 || iter_0 == 0) && (!is_reduction_axis_1 || iter_1 == 0))
                    {
                        acc_i = 0;
                    }
                    {acc_i = acc_i + inp_i;}
                }
            }
        }

    """

    empty_case = dedent(
        f"""
        // Special case for empty inputs
        if (PyArray_SIZE({inp_var}) == 0) {{
            {acc_var}_iter = ({acc_dtype}*)(PyArray_DATA({acc_var}));
            npy_intp n =  PyArray_SIZE({acc_var});
            for(npy_intp i = 0; i < n; i++)
            {{
                {acc_dtype} &{acc_var}_i = {acc_var}_iter[i];
                {initial_value}
            }}
        }} else {{
        """
    )

    # The loops are ordered by (decreasing) absolute values of inp_var's strides.
    # The first element of each pair is the absolute value of the stride
    # The second element correspond to the index in the initial loop order
    order_loops = dedent(
        f"""
        std::vector< std::pair<int, npy_intp> > loops({inp_ndim});
        std::vector< std::pair<int, npy_intp> >::iterator loops_it = loops.begin();
        """
    )

    # Fill the loop vector with the appropriate <stride, index> pairs
    for i in range(inp_ndim):
        order_loops += dedent(
            f"""
            loops_it->first = abs(PyArray_STRIDES({inp_var})[{i}]);
            loops_it->second = {i};
            ++loops_it;"""
        )

    # We sort in decreasing order so that the outermost loop (loop 0)
    # has the largest stride, and the innermost loop has the smallest stride.
    order_loops += "\nstd::sort(loops.rbegin(), loops.rend());\n"

    # Sort shape and strides to match the new order that was computed by sorting the loop vector.
    counter = iter(range(inp_ndim))
    unsorted_vars = dedent(
        f"""
        npy_intp dim_lengths[{inp_ndim}] = {{{",".join(f"{inp_var}_n{i}" for i in range(inp_ndim))}}};
        npy_intp inp_strides[{inp_ndim}] = {{{",".join(f"{inp_var}_stride{i}" for i in range(inp_ndim))}}};
        npy_intp acc_strides[{inp_ndim}] = {{{",".join("0" if i in reduction_axes else f"{acc_var}_stride{next(counter)}" for i in range(inp_ndim))}}};
        bool reduction_axes[{inp_ndim}] = {{{", ".join("1" if i in reduction_axes else "0" for i in range(inp_ndim))}}};\n
        """
    )

    sorted_vars = "loops_it = loops.begin();"
    for i in range(inp_ndim):
        sorted_vars += dedent(
            f"""
            npy_intp dim_length_{i} = dim_lengths[loops_it->second];
            bool is_reduction_axis_{i} = reduction_axes[loops_it->second];
            npy_intp {inp_var}_stride_{i} = inp_strides[loops_it->second];
            npy_intp {acc_var}_stride_{i} = acc_strides[loops_it->second];
            ++loops_it;
            """
        )

    declare_iter = dedent(
        f"""
        {inp_var}_iter = ({inp_dtype}*)(PyArray_DATA({inp_var}));
        {acc_var}_iter = ({acc_dtype}*)(PyArray_DATA({acc_var}));
        """
    )

    pointer_update = ""
    for var, dtype in ((inp_var, inp_dtype), (acc_var, acc_dtype)):
        pointer_update += f"{dtype} &{var}_i = *({var}_iter"
        for i in reversed(tuple(range(inp_ndim))):
            iter_var = f"iter_{i}"
            pointer_update += f" + {var}_stride_{i}*{iter_var}"
        pointer_update += ");\n"

    # Set initial value in first iteration of each output
    # This happens on the first iteration of every reduction axis
    initial_iteration = " && ".join(
        f"(!is_reduction_axis_{i} || iter_{i} == 0)" for i in range(inp_ndim)
    )
    set_initial_value = dedent(
        f"""
        if({initial_iteration})
        {{
            {initial_value}
        }}
        """
    )

    # We set do pointer_update, initial_value and inner task in inner loop
    loop = "\n\n".join((pointer_update, set_initial_value, f"{{{inner_task}}}"))

    # Create outer loops recursively
    for i in reversed(range(inp_ndim)):
        iter_var = f"iter_{i}"
        dim_length = f"dim_length_{i}"
        loop = dedent(
            f"""
            for(npy_intp {iter_var} = 0; {iter_var}<{dim_length}; {iter_var}++){{
                {loop}
            }}
            """
        )

    non_empty_case = "\n".join(
        (order_loops, unsorted_vars, sorted_vars, declare_iter, loop)
    )
    code = "\n".join((empty_case, non_empty_case, "}"))
    return f"{{\n{code}\n}}\n"
