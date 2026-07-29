import numpy as np

import pytensor
import pytensor.tensor as pt


def compare(output, inputs, values, *, mode="MLIR", rtol=1e-7, atol=0):
    reference = pytensor.function(inputs, output, mode="FAST_COMPILE")
    mlir = pytensor.function(inputs, output, mode=mode)
    expected = reference(*values)
    actual = mlir(*values)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return actual


if __name__ == "__main__":
    x = pt.vector("x")
    y = pt.vector("y")
    vector = compare(
        x + y * 2,
        [x, y],
        [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
    )
    multiply = compare(
        x * y,
        [x, y],
        [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
    )
    assert vector.dtype == np.dtype("float64")

    x32 = pt.vector("x32", dtype="float32")
    y32 = pt.vector("y32", dtype="float32")
    vector32 = compare(
        x32 + y32 * np.float32(2),
        [x32, y32],
        [
            np.array([1.0, 2.0, 3.0], dtype="float32"),
            np.array([4.0, 5.0, 6.0], dtype="float32"),
        ],
    )
    assert vector32.dtype == np.dtype("float32")

    matrix = pt.matrix("matrix")
    row = pt.vector("row")
    matrix_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    broadcast = compare(
        matrix + row,
        [matrix, row],
        [matrix_value, np.array([10.0, 20.0, 30.0])],
    )
    transpose = compare(matrix.T, [matrix], [matrix_value])
    assert broadcast.dtype == np.dtype("float64")
    assert transpose.dtype == np.dtype("float64")

    lhs = pt.matrix("lhs")
    rhs = pt.matrix("rhs")
    lhs_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rhs_value = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    dot = compare(pt.dot(lhs, rhs), [lhs, rhs], [lhs_value, rhs_value])
    assert dot.dtype == np.dtype("float64")

    lhs32 = pt.matrix("lhs32", dtype="float32")
    rhs32 = pt.matrix("rhs32", dtype="float32")
    dot32 = compare(
        pt.dot(lhs32, rhs32),
        [lhs32, rhs32],
        [lhs_value.astype("float32"), rhs_value.astype("float32")],
    )
    assert dot32.dtype == np.dtype("float32")
    import iree.runtime

    metal_devices = iree.runtime.get_driver("metal").query_available_devices()
    assert metal_devices
    try:
        pytensor.function([x], x + 1, mode="MLIR_METAL")
    except TypeError as error:
        metal_f64_error = str(error)
        assert metal_f64_error == "MLIR Metal only supports float32 graphs"
    else:
        raise AssertionError("MLIR_METAL must reject float64 graphs")


    metal_vector32 = compare(
        x32 + y32 * np.float32(2),
        [x32, y32],
        [
            np.array([1.0, 2.0, 3.0], dtype="float32"),
            np.array([4.0, 5.0, 6.0], dtype="float32"),
        ],
        mode="MLIR_METAL",
        rtol=1e-6,
        atol=1e-6,
    )
    metal_multiply32 = compare(
        x32 * y32,
        [x32, y32],
        [
            np.array([1.0, 2.0, 3.0], dtype="float32"),
            np.array([4.0, 5.0, 6.0], dtype="float32"),
        ],
        mode="MLIR_METAL",
        rtol=1e-6,
        atol=1e-6,
    )
    matrix32 = pt.matrix("matrix32", dtype="float32")
    row32 = pt.vector("row32", dtype="float32")
    matrix32_value = matrix_value.astype("float32")
    metal_broadcast32 = compare(
        matrix32 + row32,
        [matrix32, row32],
        [matrix32_value, np.array([10.0, 20.0, 30.0], dtype="float32")],
        mode="MLIR_METAL",
        rtol=1e-6,
        atol=1e-6,
    )
    metal_transpose32 = compare(
        matrix32.T,
        [matrix32],
        [matrix32_value],
        mode="MLIR_METAL",
        rtol=1e-6,
        atol=1e-6,
    )
    metal_dot32 = compare(
        pt.dot(lhs32, rhs32),
        [lhs32, rhs32],
        [lhs_value.astype("float32"), rhs_value.astype("float32")],
        mode="MLIR_METAL",
        rtol=1e-6,
        atol=1e-6,
    )
    assert all(
        value.dtype == np.dtype("float32")
        for value in (
            metal_vector32,
            metal_multiply32,
            metal_broadcast32,
            metal_transpose32,
            metal_dot32,
        )
    )


    print("vector_float64", vector)
    print("multiply_float64", multiply)
    print("vector_float32", vector32)
    print("broadcast_float64", broadcast)
    print("transpose_float64", transpose)
    print("dot_float64", dot)
    print("dot_float32", dot32)
    print("metal_devices", metal_devices)
    print("metal_float64_rejected", metal_f64_error)
    print("metal_float32_tolerances", "rtol=1e-6", "atol=1e-6")
    print("metal_vector_float32", metal_vector32)
    print("metal_multiply_float32", metal_multiply32)
    print("metal_broadcast_float32", metal_broadcast32)
    print("metal_transpose_float32", metal_transpose32)
    print("metal_dot_float32", metal_dot32)
