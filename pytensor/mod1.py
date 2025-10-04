import numba


@numba.extending.register_jitable(cache=True)
def foo(x):
    return x + 1
