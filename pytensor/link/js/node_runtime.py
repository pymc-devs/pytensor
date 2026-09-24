"""Call a generated JS graph through a persistent Node process and binary pipe."""

import base64
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from pytensor.link.js.dispatch.basic import js_typify


class NodeJSFunction:
    """A Python-callable PyTensor graph with dynamic shapes and a warm V8 context."""

    def __init__(self, program):
        if shutil.which("node") is None:
            raise RuntimeError("The JS backend needs Node.js on PATH")
        self.program = program
        self.source = program.source
        worker = Path(__file__).with_name("worker.mjs")
        self.process = subprocess.Popen(
            ["node", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        init = {
            "source": program.source,
            "constants": [
                {
                    "shape": list(v.shape),
                    "data": base64.b64encode(v.tobytes()).decode("ascii"),
                }
                for v in program.constants
            ],
            "inputs": [v.ndim for v in program.inputs],
            "outputs": [v.ndim for v in program.outputs],
        }
        encoded = json.dumps(init).encode("utf-8")
        self._send(struct.pack("<I", len(encoded)) + encoded)
        if self._read(1) != b"\x01":
            raise RuntimeError("Node JS evaluator failed to initialize")

    def _send(self, data):
        view = memoryview(data)
        while view:
            count = self.process.stdin.write(view)
            if count is None or count == 0:
                raise RuntimeError("Node JS evaluator closed the input pipe")
            view = view[count:]

    def _read(self, size):
        data = bytearray(size)
        view = memoryview(data)
        while view:
            count = self.process.stdout.readinto(view)
            if count is None or count == 0:
                detail = self.process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Node JS evaluator closed the output pipe: {detail}"
                )
            view = view[count:]
        return data

    def _request(self, command, payload):
        frame = bytearray(struct.pack("<BI", command, len(payload)))
        frame.extend(payload)
        self._send(frame)
        status, length = struct.unpack("<BI", self._read(5))
        result = self._read(length)
        if status != 1:
            detail = result.decode("utf-8", errors="replace")
            raise RuntimeError(f"JS evaluation failed: {detail}")
        return result

    def __call__(self, *args):
        payload = bytearray()
        for arg, var in zip(args, self.program.inputs, strict=True):
            array = js_typify(arg, var.type.dtype)
            payload.extend(struct.pack("<" + "i" * var.ndim, *array.shape))
            payload.extend(array.tobytes())
        result = self._request(1, payload)
        outputs = []
        offset = 0
        for var in self.program.outputs:
            shape = struct.unpack_from("<" + "i" * var.ndim, result, offset)
            offset += var.ndim * 4
            size = int(np.prod(shape)) if shape else 1
            values = np.frombuffer(result, dtype="float64", count=size, offset=offset)
            offset += size * 8
            outputs.append(values.reshape(shape).astype(var.type.dtype, copy=False))
        if offset != len(result):
            raise RuntimeError("JS evaluator returned trailing output bytes")
        return tuple(outputs)

    def repeat(self, n):
        """Return microseconds/evaluation for repeated calls inside V8."""
        if n < 1 or n > 2**32 - 1:
            raise ValueError("repeat count must be between 1 and 2**32-1")
        return struct.unpack("<d", self._request(2, struct.pack("<I", n)))[0]

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        self.process = None
        if process.poll() is None:
            try:
                process.stdin.write(struct.pack("<BI", 0, 0))
            except BrokenPipeError:
                pass
        process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=1)
        process.stdout.close()
        process.stderr.close()

    def __del__(self):
        self.close()
