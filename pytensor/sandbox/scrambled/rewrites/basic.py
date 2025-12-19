from egglog import String, birewrite, eq, i64, rewrite

from pytensor.sandbox.scrambled.basic import Int, IntTuple, Tensor, TensorTuple, egraph


@egraph.register
def int_rules(
    i: i64,
    j: i64,
):
    yield rewrite(Int(i) + Int(j)).to(Int(i + j))
    yield rewrite(Int(i) - Int(j)).to(Int(i - j))

    # TODO: Use booleans like normal people
    yield rewrite(Int(i) == Int(i)).to(Int(1))
    yield rewrite(Int(i) == Int(j)).to(Int(0), i != j)
    # # yield rewrite(Int(i) != Int(j)).to(Int(1), i != j)
    # # yield rewrite(Int(i) != Int(j)).to(Int(0), i == j)
    yield rewrite(Int(i) > Int(j)).to(Int(1), i > j)
    yield rewrite(Int(i) > Int(j)).to(Int(0), i <= j)
    yield rewrite(Int(i) >= Int(j)).to(Int(1), i >= j)
    yield rewrite(Int(i) >= Int(j)).to(Int(0), i < j)
    yield rewrite(Int(i) < Int(j)).to(Int(1), i < j)
    yield rewrite(Int(i) < Int(j)).to(Int(0), i >= j)
    yield rewrite(Int(i) <= Int(j)).to(Int(1), i <= j)
    yield rewrite(Int(i) <= Int(j)).to(Int(0), i > j)


def create_tuple_rules(TupleClass, tuple_var, x, y, i):
    # Concatenating to empty
    yield rewrite(TupleClass.empty() + TupleClass.empty()).to(TupleClass.empty())
    yield rewrite(TupleClass(x) + TupleClass.empty()).to(TupleClass(x))
    # yield rewrite(TupleClass(x)).to(TupleClass(x) + TupleClass.empty())
    # # Associativity
    # yield birewrite(TupleClass(x) + (TupleClass(y) + tuple_var)).to(
    #     (TupleClass(x) + TupleClass(y)) + tuple_var
    # )
    # Indexing
    yield rewrite(TupleClass(x)[0]).to(x)
    yield rewrite((TupleClass(x) + tuple_var)[0]).to(x)
    yield rewrite((TupleClass(x) + tuple_var)[i]).to(tuple_var[Int(i - 1)], i > 0)
    # Length
    yield rewrite(TupleClass.empty().length()).to(Int(0))
    yield rewrite(TupleClass(x).length()).to(Int(1))
    yield rewrite((TupleClass(x) + tuple_var).length()).to(Int(1) + tuple_var.length())
    # Insert
    yield rewrite(TupleClass.empty().insert(0, y)).to(TupleClass(y))
    yield rewrite(TupleClass(x).insert(0, y)).to(TupleClass(y) + TupleClass(x))
    yield rewrite(TupleClass(x).insert(1, y)).to(TupleClass(x) + TupleClass(y))
    yield rewrite((TupleClass(x) + tuple_var).insert(0, y)).to(
        TupleClass(y) + (TupleClass(x) + tuple_var)
    )

    yield rewrite((TupleClass(x) + tuple_var).insert(i, y)).to(
        TupleClass(x) + tuple_var.insert(i - 1, y),
        i > 0,
    )

    # Pop
    yield rewrite(TupleClass(x).pop(0)).to(TupleClass.empty())
    yield rewrite((TupleClass(x) + tuple_var).pop(0)).to(tuple_var)
    # Use pop(1) as a base case to avoid introducing IntTuple.empty() unless strictly necessary.
    # This ensures that IntTuple(i).insert(1, x).pop(1) == IntTuple(i)
    # Instead of IntTuple(i) + IntTuple.empty()
    yield rewrite((TupleClass(x) + TupleClass(y)).pop(1)).to(TupleClass(x))
    yield rewrite((TupleClass(x) + TupleClass(y) + tuple_var).pop(1)).to(
        TupleClass(x) + tuple_var
    )

    yield rewrite((TupleClass(x) + tuple_var).pop(i)).to(
        TupleClass(x) + tuple_var.pop(i - 1),
        i > 1,
    )


@egraph.register
def int_tuple_rules(
    int_tuple: IntTuple,
    x: Int,
    y: Int,
    i: i64,
    n: i64,
):
    yield from create_tuple_rules(IntTuple, int_tuple, x, y, i)

    yield rewrite(IntTuple.from_range(n, n)).to(IntTuple.empty())
    yield rewrite(IntTuple.from_range(i, n)).to(IntTuple(Int(i)), eq(i).to(n - 1))
    yield rewrite(IntTuple.from_range(i, n)).to(
        IntTuple(Int(i)) + IntTuple.from_range(i + 1, n), i < (n - 1)
    )


@egraph.register
def tensor_tuple_rules(
    tensor_tuple: TensorTuple,
    x: Tensor,
    y: Tensor,
    i: i64,
    int_tuple: IntTuple,
    sh: TensorTuple,
    static_dim: i64,
    static_sh: IntTuple,
    dim_name: String,
):
    yield from create_tuple_rules(TensorTuple, tensor_tuple, x, y, i)

    # Associativity
    yield birewrite(TensorTuple(x) + (TensorTuple(y) + tensor_tuple)).to(
        (TensorTuple(x) + TensorTuple(y)) + tensor_tuple
    )

    # TODO: Is it worth having two kinds?
    yield rewrite(TensorTuple.from_int_tuple(IntTuple.empty())).to(TensorTuple.empty())
    yield rewrite(TensorTuple.from_int_tuple(IntTuple(i))).to(
        TensorTuple(Tensor.constant(i))
    )
    yield rewrite(TensorTuple.from_int_tuple(IntTuple(i) + int_tuple)).to(
        TensorTuple(Tensor.constant(i)) + TensorTuple.from_int_tuple(int_tuple)
    )

    # If shape(x)[i] corresponds to a static dim, introduce a Tensor.constant with that value
    # Otherwise introduce shape(x)[i] there
    yield rewrite(TensorTuple.from_tensor_shape(sh, IntTuple.empty(), i)).to(
        TensorTuple.empty()
    )
    yield rewrite(TensorTuple.from_tensor_shape(sh, IntTuple(static_dim), i)).to(
        TensorTuple(Tensor.constant(static_dim))
    )
    yield rewrite(
        TensorTuple.from_tensor_shape(sh, IntTuple(static_dim) + static_sh, i)
    ).to(
        TensorTuple(Tensor.constant(static_dim))
        + TensorTuple.from_tensor_shape(sh, static_sh, i + 1)
    )
    yield rewrite(TensorTuple.from_tensor_shape(sh, IntTuple(Int.var(dim_name)), i)).to(
        TensorTuple(sh[i])
    )
    yield rewrite(
        TensorTuple.from_tensor_shape(sh, IntTuple(Int.var(dim_name)) + static_sh, i)
    ).to(TensorTuple(sh[i]) + TensorTuple.from_tensor_shape(sh, static_sh, i + 1))
