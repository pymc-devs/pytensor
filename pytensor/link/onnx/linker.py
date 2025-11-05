"""ONNX linker for PyTensor."""

import numpy as np
import onnx
import onnxruntime as ort

from pytensor.link.basic import JITLinker


class ONNXLinker(JITLinker):
    """A `Linker` that converts PyTensor graphs to ONNX models and executes them with ONNX Runtime.

    This linker:
    1. Converts the PyTensor FunctionGraph to an ONNX ModelProto
    2. Creates an ONNX Runtime InferenceSession
    3. Returns a function that executes the model via ONNX Runtime
    """

    def __init__(self, opset_version=18, *args, **kwargs):
        """Initialize the ONNX linker.

        Parameters
        ----------
        opset_version : int, default=18
            ONNX opset version to use for the model
        """
        super().__init__(*args, **kwargs)
        self.opset_version = opset_version
        self.onnx_model = None

    def fgraph_convert(self, fgraph, input_storage, storage_map, **kwargs):
        """Convert FunctionGraph to ONNX and create executable function.

        Parameters
        ----------
        fgraph : FunctionGraph
            The function graph to convert
        input_storage : list
            Storage for inputs
        storage_map : dict
            Mapping from variables to storage

        Returns
        -------
        callable
            Function that executes the ONNX model
        """
        from pytensor.link.onnx.dispatch import onnx_funcify

        # Convert the FunctionGraph to ONNX ModelProto
        self.onnx_model = onnx_funcify(
            fgraph,
            opset_version=self.opset_version,
            input_storage=input_storage,
            storage_map=storage_map,
            **kwargs,
        )

        # Create ONNX Runtime function
        return self._create_onnx_runtime_function(fgraph)

    def _create_onnx_runtime_function(self, fgraph):
        """Create a function that executes the ONNX model via ONNX Runtime.

        Parameters
        ----------
        fgraph : FunctionGraph
            The function graph (for input/output info)

        Returns
        -------
        callable
            Function that takes inputs and returns outputs
        """
        # Serialize the model to bytes
        model_bytes = self.onnx_model.SerializeToString()

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Error level only
        session = ort.InferenceSession(model_bytes, sess_options)

        # Get input and output names from the ONNX model
        input_names = [inp.name for inp in self.onnx_model.graph.input]
        output_names = [out.name for out in self.onnx_model.graph.output]

        def onnx_runtime_function(*args):
            """Execute the ONNX model with ONNX Runtime.

            Parameters
            ----------
            *args : array-like
                Input values matching the graph inputs

            Returns
            -------
            array or tuple of arrays
                Output values from the ONNX model
            """
            # Prepare inputs as numpy arrays
            input_dict = {}
            for name, arg in zip(input_names, args):
                # Ensure inputs are numpy arrays with correct dtype
                if not isinstance(arg, np.ndarray):
                    arg = np.array(arg)
                input_dict[name] = arg

            # Run the model
            outputs = session.run(output_names, input_dict)

            # Return outputs as tuple to match expected format
            # (even for single outputs, as the thunk expects to iterate)
            return tuple(outputs)

        return onnx_runtime_function

    def create_thunk_inputs(self, storage_map):
        """Create thunk inputs from storage map.

        For ONNX, we simply return the storage list for each input variable.

        Parameters
        ----------
        storage_map : dict
            Mapping from variables to storage

        Returns
        -------
        list
            List of storage lists for inputs
        """
        thunk_inputs = []
        for n in self.fgraph.inputs:
            thunk_inputs.append(storage_map[n])
        return thunk_inputs

    def jit_compile(self, fn):
        """JIT compile a converted FunctionGraph.

        For ONNX, there is no additional JIT compilation needed -
        the function returned by fgraph_convert already executes via ONNX Runtime.

        Parameters
        ----------
        fn : callable
            The function to compile

        Returns
        -------
        callable
            The same function (no additional compilation needed)
        """
        # No JIT compilation needed for ONNX - already compiled to ONNX Runtime
        return fn

    def export_to_file(self, filename):
        """Export the ONNX model to a file.

        Parameters
        ----------
        filename : str or Path
            Path to save the ONNX model

        Raises
        ------
        ValueError
            If no model has been created yet
        """
        if self.onnx_model is None:
            raise ValueError(
                "No ONNX model available. Compile a function first."
            )

        onnx.save(self.onnx_model, filename)
