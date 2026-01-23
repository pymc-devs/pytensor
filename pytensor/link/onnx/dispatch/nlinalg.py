"""ONNX conversion for linear algebra operations."""

from onnx import helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.tensor.blas import BatchedDot, Gemm
from pytensor.tensor.math import Dot


@onnx_funcify.register(Dot)
def onnx_funcify_Dot(op, node, get_var_name, **kwargs):
    """Convert Dot op to ONNX MatMul node.

    Dot performs matrix multiplication. ONNX MatMul handles:
    - Matrix @ Matrix
    - Vector @ Matrix (with implicit unsqueeze)
    - Batched operations
    """
    input_a = get_var_name(node.inputs[0])
    input_b = get_var_name(node.inputs[1])
    output_name = get_var_name(node.outputs[0])

    # ONNX MatMul handles most cases directly
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[input_a, input_b],
        outputs=[output_name],
        name=f"MatMul_{output_name}",
    )

    return matmul_node


@onnx_funcify.register(Gemm)
def onnx_funcify_Gemm(op, node, get_var_name, **kwargs):
    """Convert Gemm op to ONNX Gemm node.

    PyTensor Gemm: gemm(C, alpha, A, B, beta) = beta*C + alpha*dot(A, B)
    ONNX Gemm: Y = alpha * A' * B' + beta * C

    Where inputs are: [C, alpha, A, B, beta]
    Remap to ONNX: [A, B, C] with alpha and beta as attributes
    """
    from pytensor.graph import Constant

    # PyTensor inputs: [C, alpha, A, B, beta]
    input_c = get_var_name(node.inputs[0])
    alpha_var = node.inputs[1]
    input_a = get_var_name(node.inputs[2])
    input_b = get_var_name(node.inputs[3])
    beta_var = node.inputs[4]
    output_name = get_var_name(node.outputs[0])

    # Extract alpha and beta values (should be constants)
    if isinstance(alpha_var, Constant):
        alpha = float(alpha_var.data)
    else:
        alpha = 1.0

    if isinstance(beta_var, Constant):
        beta = float(beta_var.data)
    else:
        beta = 1.0

    # ONNX Gemm: Y = alpha * A @ B + beta * C
    gemm_node = helper.make_node(
        "Gemm",
        inputs=[input_a, input_b, input_c],
        outputs=[output_name],
        name=f"Gemm_{output_name}",
        alpha=alpha,
        beta=beta,
        transA=0,
        transB=0,
    )

    return gemm_node


@onnx_funcify.register(BatchedDot)
def onnx_funcify_BatchedDot(op, node, get_var_name, **kwargs):
    """Convert BatchedDot to ONNX MatMul.

    BatchedDot performs batched matrix multiplication.
    ONNX MatMul handles batching natively.
    """
    input_a = get_var_name(node.inputs[0])
    input_b = get_var_name(node.inputs[1])
    output_name = get_var_name(node.outputs[0])

    matmul_node = helper.make_node(
        "MatMul",
        inputs=[input_a, input_b],
        outputs=[output_name],
        name=f"MatMul_{output_name}",
    )

    return matmul_node
