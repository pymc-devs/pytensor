"""ONNX conversion for elementwise operations."""

from onnx import helper

from pytensor.link.onnx.dispatch.basic import onnx_funcify
from pytensor.scalar import basic as scalar
from pytensor.scalar import math as scalar_math
from pytensor.tensor.elemwise import Elemwise

# ⭐ THE MAGIC MAPPING - Tier 1 + Tier 4-5 operations
SCALAR_OP_TO_ONNX = {
    # Arithmetic (Tier 1)
    scalar.Add: "Add",
    scalar.Mul: "Mul",
    scalar.Sub: "Sub",
    scalar.TrueDiv: "Div",
    scalar.Neg: "Neg",
    # Note: IntDiv handled specially in onnx_funcify_Elemwise as Div + Floor
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
    # Trigonometric (Tier 5)
    scalar.Sin: "Sin",
    scalar.Cos: "Cos",
    scalar.Tan: "Tan",
    scalar.ArcSin: "Asin",
    scalar.ArcCos: "Acos",
    scalar.ArcTan: "Atan",
    # Hyperbolic (Tier 5)
    scalar.Sinh: "Sinh",
    scalar.Cosh: "Cosh",
    scalar.Tanh: "Tanh",
    scalar.ArcSinh: "Asinh",
    scalar.ArcCosh: "Acosh",
    scalar.ArcTanh: "Atanh",
    # Comparison (Tier 5)
    scalar.LT: "Less",
    scalar.GT: "Greater",
    scalar.LE: "LessOrEqual",
    scalar.GE: "GreaterOrEqual",
    scalar.EQ: "Equal",
    # Note: NEQ is handled specially in onnx_funcify_Elemwise as Equal + Not
    # Logical (Tier 5)
    scalar.AND: "And",
    scalar.OR: "Or",
    scalar.XOR: "Xor",
    scalar.Invert: "Not",
    # Special (Tier 5)
    scalar_math.Sigmoid: "Sigmoid",
    scalar_math.Softplus: "Softplus",
    scalar_math.Erf: "Erf",
    # Note: Clip handled specially in onnx_funcify_Elemwise (requires scalar min/max)
    # Conditional
    scalar.Switch: "Where",
}


@onnx_funcify.register(Elemwise)
def onnx_funcify_Elemwise(op, node, get_var_name, **kwargs):
    """Convert Elemwise op to ONNX node.

    This ONE function handles ALL operations, including composed ones!

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
    onnx.NodeProto or list[onnx.NodeProto]
        ONNX node(s) for the operation
    """
    scalar_op_type = type(op.scalar_op)

    # Special handling for operations that need to be composed
    # Clip(x, min, max) - ONNX requires scalar min/max, but PyTensor may provide tensors
    if scalar_op_type == scalar.Clip:
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_name = get_var_name(node.outputs[0])

        # Input 0 is the array to clip, inputs 1 and 2 are min/max
        # ONNX Clip expects scalars for min/max, but PyTensor may have added dimensions
        # We need to squeeze them if they're not scalars
        x_name = input_names[0]
        min_name = input_names[1]
        max_name = input_names[2]

        # Create Squeeze nodes for min and max to ensure they're scalars
        # ONNX Squeeze with empty axes removes all dimensions of size 1
        min_scalar_name = f"{output_name}_min_scalar"
        min_squeeze = helper.make_node(
            "Squeeze",
            inputs=[min_name],
            outputs=[min_scalar_name],
            name=f"Squeeze_{min_scalar_name}",
        )

        max_scalar_name = f"{output_name}_max_scalar"
        max_squeeze = helper.make_node(
            "Squeeze",
            inputs=[max_name],
            outputs=[max_scalar_name],
            name=f"Squeeze_{max_scalar_name}",
        )

        # Clip with scalar min/max
        clip_node = helper.make_node(
            "Clip",
            inputs=[x_name, min_scalar_name, max_scalar_name],
            outputs=[output_name],
            name=f"Clip_{output_name}",
        )

        return [min_squeeze, max_squeeze, clip_node]

    # IntDiv(x, y) = Floor(Div(x, y))
    if scalar_op_type == scalar.IntDiv:
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_name = get_var_name(node.outputs[0])

        # Div(x, y)
        div_name = f"{output_name}_div"
        div_node = helper.make_node(
            "Div",
            inputs=input_names,
            outputs=[div_name],
            name=f"Div_{div_name}",
        )

        # Floor(Div(x, y))
        floor_node = helper.make_node(
            "Floor",
            inputs=[div_name],
            outputs=[output_name],
            name=f"Floor_{output_name}",
        )

        return [div_node, floor_node]

    # NEQ(x, y) = Not(Equal(x, y))
    if scalar_op_type == scalar.NEQ:
        input_names = [get_var_name(inp) for inp in node.inputs]
        output_name = get_var_name(node.outputs[0])

        # Equal(x, y)
        equal_name = f"{output_name}_equal"
        equal_node = helper.make_node(
            "Equal",
            inputs=input_names,
            outputs=[equal_name],
            name=f"Equal_{equal_name}",
        )

        # Not(Equal(x, y))
        not_node = helper.make_node(
            "Not",
            inputs=[equal_name],
            outputs=[output_name],
            name=f"Not_{output_name}",
        )

        return [equal_node, not_node]

    # Log1p(x) = Log(Add(x, 1))
    if scalar_op_type == scalar.Log1p:
        input_name = get_var_name(node.inputs[0])
        output_name = get_var_name(node.outputs[0])

        # Create constant 1
        one_name = f"{output_name}_one"
        one_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[one_name],
            value=helper.make_tensor("value", helper.TensorProto.FLOAT, [], [1.0]),
        )

        # Add(x, 1)
        add_name = f"{output_name}_add"
        add_node = helper.make_node(
            "Add",
            inputs=[input_name, one_name],
            outputs=[add_name],
            name=f"Add_{add_name}",
        )

        # Log(Add(x, 1))
        log_node = helper.make_node(
            "Log",
            inputs=[add_name],
            outputs=[output_name],
            name=f"Log_{output_name}",
        )

        return [one_node, add_node, log_node]

    # Expm1(x) = Sub(Exp(x), 1)
    if scalar_op_type == scalar.Expm1:
        input_name = get_var_name(node.inputs[0])
        output_name = get_var_name(node.outputs[0])

        # Exp(x)
        exp_name = f"{output_name}_exp"
        exp_node = helper.make_node(
            "Exp",
            inputs=[input_name],
            outputs=[exp_name],
            name=f"Exp_{exp_name}",
        )

        # Create constant 1
        one_name = f"{output_name}_one"
        one_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[one_name],
            value=helper.make_tensor("value", helper.TensorProto.FLOAT, [], [1.0]),
        )

        # Sub(Exp(x), 1)
        sub_node = helper.make_node(
            "Sub",
            inputs=[exp_name, one_name],
            outputs=[output_name],
            name=f"Sub_{output_name}",
        )

        return [exp_node, one_node, sub_node]

    # Standard operations
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
