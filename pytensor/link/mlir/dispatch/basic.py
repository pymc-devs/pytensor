from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from pytensor.compile.ops import DeepCopyOp
from pytensor.graph import Constant
from pytensor.graph.fg import AbstractFunctionGraph
from pytensor.scalar.basic import Add, Mul, Second, Sub
from pytensor.scalar.math import Sigmoid
from pytensor.tensor.blas import Dot22
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot


MLIR_DTYPES = {"float32": "f32", "float64": "f64"}


@dataclass(frozen=True)
class MLIRModule:
    text: str
    entrypoint: str = "main"
    output_count: int = 1
    runtime_validator: Callable[..., None] | None = None


def _typify_array(array):
    if array.dtype.name not in MLIR_DTYPES:
        raise TypeError(
            f"MLIR only supports float32 and float64 inputs, got {array.dtype}"
        )
    return array if array.ndim == 0 else np.ascontiguousarray(array)


@singledispatch
def mlir_typify(data, **kwargs):
    return _typify_array(np.asarray(data))


@mlir_typify.register(np.ndarray)
def mlir_typify_ndarray(data, **kwargs):
    return _typify_array(data)


@singledispatch
def mlir_funcify(op, node=None, emitter=None, **kwargs):
    raise NotImplementedError(f"No MLIR conversion for the given Op: {op}")


@mlir_funcify.register(AbstractFunctionGraph)
def mlir_funcify_FunctionGraph(fgraph, **kwargs):
    return _MLIREmitter(fgraph).emit()


@mlir_funcify.register(Elemwise)
def mlir_funcify_Elemwise(op, node, emitter, **kwargs):
    emitter.emit_elemwise(node)


@mlir_funcify.register(DimShuffle)
def mlir_funcify_DimShuffle(op, node, emitter, **kwargs):
    emitter.emit_dimshuffle(node)


@mlir_funcify.register(DeepCopyOp)
def mlir_funcify_DeepCopyOp(op, node, emitter, **kwargs):
    emitter.emit_deepcopy(node)


@mlir_funcify.register(Dot)
@mlir_funcify.register(Dot22)
def mlir_funcify_Dot22(op, node, emitter, **kwargs):
    emitter.emit_dot22(node)


class _MLIREmitter:
    def __init__(self, fgraph):
        self.fgraph = fgraph
        self.nodes = fgraph.toposort()
        self.body: list[str] = []
        self.values = {}
        self.counter = 0
        self.index_constants = {}

        for index, variable in enumerate(fgraph.inputs):
            self._tensor_type(variable)
            self.values[variable] = f"%arg{index}"

    def emit(self):
        for node in self.nodes:
            mlir_funcify(node.op, node=node, emitter=self)

        result_types = [self._tensor_type(output) for output in self.fgraph.outputs]
        result_values = [self._value(output) for output in self.fgraph.outputs]
        arguments = ", ".join(
            f"{self.values[variable]}: {self._tensor_type(variable)}"
            for variable in self.fgraph.inputs
        )
        results = ", ".join(result_types)
        signature = f"({arguments})"
        if result_types:
            result_signature = results if len(result_types) == 1 else f"({results})"
            signature = f"{signature} -> {result_signature}"

        if result_values:
            self.body.append(f"return {', '.join(result_values)} : {results}")
        else:
            self.body.append("return")

        body = "\n".join(f"    {line}" for line in self.body)
        return MLIRModule(
            "\n".join(
                (
                    "module @pytensor_mlir {",
                    f"  func.func @main{signature} {{",
                    body,
                    "  }",
                    "}",
                )
            ),
            output_count=len(result_values),
            runtime_validator=self.validate_runtime_inputs,
        )

    def emit_elemwise(self, node):
        if len(node.outputs) != 1:
            raise NotImplementedError("MLIR Elemwise lowering requires one output")

        output = node.outputs[0]
        output_type = self._tensor_type(output)
        rank = output.type.ndim
        dtype = self._element_dtype(output)
        if rank > 2:
            raise NotImplementedError(
                "MLIR Elemwise lowering supports tensors up to rank 2"
            )

        inputs = list(node.inputs)
        for variable in inputs:
            self._validate_elemwise_input(variable, output)

        nonconstant_inputs = [
            variable for variable in inputs if not isinstance(variable, Constant)
        ]
        if not nonconstant_inputs:
            raise NotImplementedError("MLIR Elemwise lowering requires a tensor input")

        init = self._empty_for_elemwise_output(output, inputs)
        input_maps = [
            self._elemwise_map(variable, output) for variable in nonconstant_inputs
        ]
        output_map = self._identity_map(rank)
        iterator_types = ", ".join('"parallel"' for _ in range(rank))
        input_names = ", ".join(
            self._value(variable) for variable in nonconstant_inputs
        )
        input_types = ", ".join(
            self._tensor_type(variable) for variable in nonconstant_inputs
        )
        result = self._fresh("value")
        maps = ", ".join((*input_maps, output_map))
        ins = f"ins({input_names} : {input_types}) "
        self.body.append(
            f"{result} = linalg.generic {{indexing_maps = [{maps}], "
            f"iterator_types = [{iterator_types}]}} {ins}"
            f"outs({init} : {output_type}) {{"
        )

        block_arguments = ", ".join(
            [
                *(
                    f"%input{index}: {dtype}"
                    for index in range(len(nonconstant_inputs))
                ),
                f"%output: {dtype}",
            ]
        )
        self.body.append(f"  ^bb0({block_arguments}):")
        block_values = iter(
            f"%input{index}" for index in range(len(nonconstant_inputs))
        )
        scalar_inputs = []
        for variable in inputs:
            if isinstance(variable, Constant):
                scalar_inputs.append(self._emit_constant(variable, dtype))
            else:
                scalar_inputs.append(next(block_values))
        value = self._emit_scalar_op(node.op.scalar_op, scalar_inputs, dtype)
        self.body.append(f"    linalg.yield {value} : {dtype}")
        self.body.append(f"}} -> {output_type}")
        self.values[output] = result

    def emit_dimshuffle(self, node):
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise NotImplementedError(
                "MLIR DimShuffle lowering requires one input and one output"
            )

        input_variable = node.inputs[0]
        output = node.outputs[0]
        input_type = self._tensor_type(input_variable)
        output_type = self._tensor_type(output)
        if input_variable.type.ndim > 2 or output.type.ndim > 2:
            raise NotImplementedError(
                "MLIR DimShuffle lowering supports tensors up to rank 2"
            )
        if self._element_dtype(input_variable) != self._element_dtype(output):
            raise TypeError("MLIR DimShuffle requires matching input and output dtypes")

        new_order = node.op.new_order
        init = self._empty_for_dimshuffle_output(output, input_variable, new_order)
        input_map = self._dimshuffle_map(input_variable, output, new_order)
        output_map = self._identity_map(output.type.ndim)
        iterator_types = ", ".join('"parallel"' for _ in range(output.type.ndim))
        result = self._fresh("value")
        dtype = self._element_dtype(output)
        self.body.append(
            f"{result} = linalg.generic {{indexing_maps = [{input_map}, {output_map}], "
            f"iterator_types = [{iterator_types}]}} "
            f"ins({self._value(input_variable)} : {input_type}) "
            f"outs({init} : {output_type}) {{"
        )
        self.body.append(f"  ^bb0(%input: {dtype}, %output: {dtype}):")
        self.body.append(f"    linalg.yield %input : {dtype}")
        self.body.append(f"}} -> {output_type}")
        self.values[output] = result

    def emit_deepcopy(self, node):
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise NotImplementedError(
                "MLIR DeepCopyOp lowering requires one input and one output"
            )

        input_variable = node.inputs[0]
        output = node.outputs[0]
        input_type = self._tensor_type(input_variable)
        output_type = self._tensor_type(output)
        if input_variable.type.ndim > 2 or output.type.ndim > 2:
            raise NotImplementedError(
                "MLIR DeepCopyOp lowering supports tensors up to rank 2"
            )
        dtype = self._element_dtype(output)
        if self._element_dtype(input_variable) != dtype:
            raise TypeError("MLIR DeepCopyOp requires matching input and output dtypes")

        init = self._empty_for_output(
            output,
            {
                axis: (input_variable, axis)
                for axis, size in enumerate(output.type.shape)
                if size is None
            },
        )
        input_map = self._identity_map(output.type.ndim)
        result = self._fresh("value")
        iterator_types = ", ".join('"parallel"' for _ in range(output.type.ndim))
        self.body.append(
            f"{result} = linalg.generic {{indexing_maps = [{input_map}, {input_map}], "
            f"iterator_types = [{iterator_types}]}} "
            f"ins({self._value(input_variable)} : {input_type}) "
            f"outs({init} : {output_type}) {{"
        )
        self.body.append(f"  ^bb0(%input: {dtype}, %output: {dtype}):")
        self.body.append(f"    linalg.yield %input : {dtype}")
        self.body.append(f"}} -> {output_type}")
        self.values[output] = result

    def emit_dot22(self, node):
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise NotImplementedError(
                "MLIR Dot22 lowering requires two inputs and one output"
            )

        lhs, rhs = node.inputs
        if isinstance(lhs, Constant) or isinstance(rhs, Constant):
            raise NotImplementedError("MLIR Dot lowering requires tensor inputs")
        output = node.outputs[0]
        lhs_type = self._tensor_type(lhs)
        rhs_type = self._tensor_type(rhs)
        output_type = self._tensor_type(output)
        if (lhs.type.ndim, rhs.type.ndim, output.type.ndim) != (2, 2, 2):
            raise NotImplementedError("MLIR Dot22 lowering requires rank-2 tensors")
        dtype = self._element_dtype(output)
        if self._element_dtype(lhs) != dtype or self._element_dtype(rhs) != dtype:
            raise TypeError("MLIR Dot22 requires matching input and output dtypes")

        init = self._empty_for_output(
            output,
            {0: (lhs, 0), 1: (rhs, 1)},
        )
        zero = self._fresh("zero")
        self.body.append(f"{zero} = arith.constant 0.0 : {dtype}")
        filled = self._fresh("filled")
        self.body.append(
            f"{filled} = linalg.fill ins({zero} : {dtype}) outs({init} : {output_type}) "
            f"-> {output_type}"
        )
        result = self._fresh("value")
        self.body.append(
            f"{result} = linalg.matmul "
            f"ins({self._value(lhs)}, {self._value(rhs)} : {lhs_type}, {rhs_type}) "
            f"outs({filled} : {output_type}) -> {output_type}"
        )
        self.values[output] = result

    def _emit_scalar_op(self, scalar_op, operands, dtype):
        if isinstance(scalar_op, Second):
            return operands[1]
        if isinstance(scalar_op, Sigmoid):
            negative = self._fresh("scalar")
            exponential = self._fresh("scalar")
            one = self._fresh("scalar")
            denominator = self._fresh("scalar")
            result = self._fresh("scalar")
            self.body.append(f"    {negative} = arith.negf {operands[0]} : {dtype}")
            self.body.append(f"    {exponential} = math.exp {negative} : {dtype}")
            self.body.append(f"    {one} = arith.constant 1.0 : {dtype}")
            self.body.append(
                f"    {denominator} = arith.addf {one}, {exponential} : {dtype}"
            )
            self.body.append(
                f"    {result} = arith.divf {one}, {denominator} : {dtype}"
            )
            return result
        if isinstance(scalar_op, Add):
            operation = "arith.addf"
        elif isinstance(scalar_op, Sub):
            operation = "arith.subf"
        elif isinstance(scalar_op, Mul):
            operation = "arith.mulf"
        else:
            raise NotImplementedError(
                f"MLIR Elemwise lowering does not support {scalar_op}"
            )

        value = operands[0]
        for operand in operands[1:]:
            result = self._fresh("scalar")
            self.body.append(f"    {result} = {operation} {value}, {operand} : {dtype}")
            value = result
        return value

    def _empty_for_elemwise_output(self, output, inputs):
        sources = {}
        output_rank = output.type.ndim
        for output_axis, output_size in enumerate(output.type.shape):
            if output_size is not None:
                continue
            for variable in inputs:
                if isinstance(variable, Constant):
                    continue
                input_rank = variable.type.ndim
                input_axis = output_axis - (output_rank - input_rank)
                if input_axis < 0 or variable.type.shape[input_axis] == 1:
                    continue
                sources[output_axis] = (variable, input_axis)
                break
            else:
                raise NotImplementedError(
                    "MLIR could not determine a dynamic Elemwise output dimension"
                )
        return self._empty_for_output(output, sources)

    def _empty_for_dimshuffle_output(self, output, input_variable, new_order):
        sources = {
            output_axis: (input_variable, input_axis)
            for output_axis, input_axis in enumerate(new_order)
            if isinstance(input_axis, int) and output.type.shape[output_axis] is None
        }
        return self._empty_for_output(output, sources)

    def _empty_for_output(self, output, dimension_sources):
        output_type = self._tensor_type(output)
        dynamic_dimensions = []
        for output_axis, output_size in enumerate(output.type.shape):
            if output_size is not None:
                continue
            try:
                source, source_axis = dimension_sources[output_axis]
            except KeyError as error:
                raise NotImplementedError(
                    "MLIR could not determine a dynamic output dimension"
                ) from error
            dimension = self._fresh("dim")
            self.body.append(
                f"{dimension} = tensor.dim {self._value(source)}, {self._index(source_axis)} "
                f": {self._tensor_type(source)}"
            )
            dynamic_dimensions.append(dimension)
        init = self._fresh("init")
        dimensions = ", ".join(dynamic_dimensions)
        self.body.append(f"{init} = tensor.empty({dimensions}) : {output_type}")
        return init

    def validate_runtime_inputs(self, *inputs):
        if len(inputs) != len(self.fgraph.inputs):
            raise ValueError("MLIR received an unexpected number of inputs")

        shapes = {}
        for variable, value in zip(self.fgraph.inputs, inputs, strict=True):
            shape = np.asarray(value).shape
            if len(shape) != variable.type.ndim:
                raise ValueError(
                    f"MLIR expected rank {variable.type.ndim} for {variable}"
                )
            for actual, expected in zip(shape, variable.type.shape, strict=True):
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"MLIR expected shape {variable.type.shape} for {variable}"
                    )
            shapes[variable] = shape

        for node in self.nodes:
            output = node.outputs[0]
            if isinstance(node.op, Elemwise):
                shape = self._runtime_elemwise_shape(node, shapes)
            elif isinstance(node.op, DimShuffle):
                input_shape = self._runtime_shape(node.inputs[0], shapes)
                shape = tuple(
                    1 if axis == "x" else input_shape[axis]
                    for axis in node.op.new_order
                )
            elif isinstance(node.op, DeepCopyOp):
                shape = self._runtime_shape(node.inputs[0], shapes)
            elif isinstance(node.op, (Dot, Dot22)):
                lhs_shape = self._runtime_shape(node.inputs[0], shapes)
                rhs_shape = self._runtime_shape(node.inputs[1], shapes)
                if lhs_shape[1] != rhs_shape[0]:
                    raise ValueError("MLIR Dot requires matching inner dimensions")
                shape = (lhs_shape[0], rhs_shape[1])
            else:
                raise NotImplementedError(f"No MLIR shape validation for {node.op}")
            self._validate_runtime_output_shape(output, shape)
            shapes[output] = shape

    def _runtime_elemwise_shape(self, node, shapes):
        output = node.outputs[0]
        output_shape = []
        for output_axis in range(output.type.ndim):
            dimensions = []
            for variable in node.inputs:
                input_shape = self._runtime_shape(variable, shapes)
                input_axis = output_axis - (output.type.ndim - len(input_shape))
                if input_axis < 0 or variable.type.shape[input_axis] == 1:
                    continue
                dimensions.append(input_shape[input_axis])
            if dimensions and any(
                dimension != dimensions[0] for dimension in dimensions[1:]
            ):
                raise ValueError(
                    "Runtime broadcasting not allowed: a dynamic dimension differs "
                    "from another input dimension."
                )
            output_shape.append(dimensions[0] if dimensions else 1)
        return tuple(output_shape)

    @staticmethod
    def _runtime_shape(variable, shapes):
        if isinstance(variable, Constant):
            return np.asarray(variable.data).shape
        return shapes[variable]

    @staticmethod
    def _validate_runtime_output_shape(variable, shape):
        for actual, expected in zip(shape, variable.type.shape, strict=True):
            if expected is not None and actual != expected:
                raise ValueError(f"MLIR computed invalid shape {shape} for {variable}")

    def _validate_elemwise_input(self, variable, output):
        if isinstance(variable, Constant):
            if np.asarray(variable.data).size != 1:
                raise NotImplementedError(
                    "MLIR Elemwise lowering only supports scalar or size-one constants"
                )
            return
        self._tensor_type(variable)
        if variable.type.ndim > output.type.ndim:
            raise NotImplementedError("MLIR Elemwise input rank exceeds output rank")
        if self._element_dtype(variable) != self._element_dtype(output):
            raise TypeError("MLIR Elemwise requires matching input and output dtypes")

    def _elemwise_map(self, variable, output):
        output_rank = output.type.ndim
        expressions = []
        for input_axis, input_size in enumerate(variable.type.shape):
            output_axis = output_rank - variable.type.ndim + input_axis
            expressions.append("0" if input_size == 1 else f"d{output_axis}")
        return self._affine_map(output_rank, expressions)

    def _dimshuffle_map(self, input_variable, output, new_order):
        expressions = []
        for input_axis in range(input_variable.type.ndim):
            try:
                output_axis = new_order.index(input_axis)
            except ValueError:
                expressions.append("0")
            else:
                expressions.append(f"d{output_axis}")
        return self._affine_map(output.type.ndim, expressions)

    @staticmethod
    def _affine_map(rank, expressions):
        dimensions = ", ".join(f"d{index}" for index in range(rank))
        return f"affine_map<({dimensions}) -> ({', '.join(expressions)})>"

    def _identity_map(self, rank):
        return self._affine_map(rank, [f"d{index}" for index in range(rank)])

    def _emit_constant(self, variable, dtype):
        data = np.asarray(variable.data)
        if data.size != 1:
            raise NotImplementedError("MLIR only supports size-one constants")
        value = float(data.reshape(()))
        if not np.isfinite(value):
            raise NotImplementedError("MLIR only supports finite constants")
        literal = repr(value)
        if "e" in literal:
            mantissa, exponent = literal.split("e")
            if "." not in mantissa:
                literal = f"{mantissa}.0e{exponent}"
        constant = self._fresh("constant")
        self.body.append(f"    {constant} = arith.constant {literal} : {dtype}")
        return constant

    def _tensor_type(self, variable):
        try:
            dtype = self._element_dtype(variable)
            shape = variable.type.shape
        except AttributeError as error:
            raise TypeError(
                f"MLIR only supports TensorType values, got {variable.type}"
            ) from error
        dimensions = "x".join("?" if size is None else str(size) for size in shape)
        return f"tensor<{dimensions + 'x' if dimensions else ''}{dtype}>"

    @staticmethod
    def _element_dtype(variable):
        try:
            return MLIR_DTYPES[variable.type.dtype]
        except KeyError as error:
            raise TypeError(
                f"MLIR only supports float32 and float64 tensors, got {variable.type.dtype}"
            ) from error

    def _value(self, variable):
        try:
            return self.values[variable]
        except KeyError as error:
            if isinstance(variable, Constant):
                raise NotImplementedError(
                    "MLIR does not support a constant graph output"
                ) from error
            raise ValueError(f"MLIR value was not emitted for {variable}") from error

    def _index(self, value):
        if value not in self.index_constants:
            constant = self._fresh("index")
            self.body.append(f"{constant} = arith.constant {value} : index")
            self.index_constants[value] = constant
        return self.index_constants[value]

    def _fresh(self, prefix):
        value = f"%{prefix}{self.counter}"
        self.counter += 1
        return value
