"""ONNX conversion for elementwise operations."""

from onnx import helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.scalar import basic as scalar
from pytensor.tensor.elemwise import Elemwise

# ⭐ THE MAGIC MAPPING - All 20 Tier 1 operations in one dict!
SCALAR_OP_TO_ONNX = {
    # Arithmetic (Tier 1)
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    scalar.IntDiv: "Div",
    # Math (Tier 1)
    scalar.Abs: "Abs",
    scalar.Exp: "Exp",
    scalar.Log: "Log",
    scalar.Sqrt: "Sqrt",
    scalar.Pow: "Pow",
    scalar.Floor: "Floor",
    scalar.Ceil: "Ceil",
    scalar.RoundHalfToEven: "Round",
    scalar.RoundHalfAwayFromZero: "Round",
    # Min/Max (Tier 1)
    scalar.Maximum: "Max",
    scalar.Minimum: "Min",
}


@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node.

    This ONE function handles ALL 20 operations!

    Parameters
    ----------
    op : Elemwise
        The elementwise operation
    node : Apply
        The Apply node
    get_var_name : callable
        Function to get variable names
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    onnx.NodeProto
        ONNX node for the operation
    """
    scalar_op_type = type(op.scalar_op)

    if scalar_op_type not in SCALAR_OP_TO_ONNX:
        raise NotImplementedError(
            f"Elemwise scalar op not supported for ONNX export: {scalar_op_type.__name__}. "
            f"Supported operations: {list(SCALAR_OP_TO_ONNX.keys())}"
        )

    onnx_op_type = SCALAR_OP_TO_ONNX[scalar_op_type]

    # Get input and output names
    input_names = [get_var_name(inp) for inp in node.inputs]
    output_names = [get_var_name(out) for out in node.outputs]

    # Create ONNX node
    return helper.make_node(
        onnx_op_type,
        inputs=input_names,
        outputs=output_names,
        name=f"{onnx_op_type}_{output_names[0]}",
    )
