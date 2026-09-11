import numpy as np

import iree.compiler.tools
import iree.runtime


MLIR_SOURCE = r'''
module {
  func.func @add(%lhs: tensor<?xf64>, %rhs: tensor<?xf64>) -> tensor<?xf64> {
    %c0 = arith.constant 0 : index
    %size = tensor.dim %lhs, %c0 : tensor<?xf64>
    %init = tensor.empty(%size) : tensor<?xf64>
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%lhs, %rhs : tensor<?xf64>, tensor<?xf64>) outs(%init : tensor<?xf64>) {
    ^bb0(%lhs_value: f64, %rhs_value: f64, %out: f64):
      %sum = arith.addf %lhs_value, %rhs_value : f64
      linalg.yield %sum : f64
    } -> tensor<?xf64>
    return %result : tensor<?xf64>
  }
}
'''


if __name__ == "__main__":
    vmfb = iree.compiler.tools.compile_str(
        MLIR_SOURCE,
        target_backends=["llvm-cpu"],
        extra_args=["--iree-input-demote-f64-to-f32=false"],
    )
    module = iree.runtime.load_vm_flatbuffer(vmfb, backend="llvm-cpu")
    lhs = np.array([1.0, 2.0, 3.0], dtype="float64")
    rhs = np.array([4.0, 5.0, 6.0], dtype="float64")
    result = np.asarray(module.add(lhs, rhs))
    np.testing.assert_allclose(result, lhs + rhs)
    assert result.dtype == np.dtype("float64")
    print(result)
