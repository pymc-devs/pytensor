"""Compile a one-parameter PyMC model in PyTensor, then sample inside Node.

Run with the PyMC environment and the PyTensor JS backend checkout on PYTHONPATH.
The exported program is sent once; no Python call occurs within the sampler.
"""

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymc as pm

import pytensor


def sample_normal_mean(*, draws=10_000, tune=1_000, seed=12345):
    observed = np.array([1.2, 0.8, 1.1, 0.9], dtype="float64")
    with pm.Model() as model:
        pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=model["mu"], sigma=1.0, observed=observed)

    fn = pytensor.function(model.value_vars, model.logp(), mode="JS")
    try:
        program = fn.vm.jit_fn.program
        assert len(program.inputs) == len(program.outputs) == 1
        payload = {
            "program": {
                "source": program.source,
                "constants": [
                    {
                        "shape": list(value.shape),
                        "data": base64.b64encode(value.tobytes()).decode("ascii"),
                    }
                    for value in program.constants
                ],
                "inputs": [
                    {"ndim": variable.ndim, "dtype": variable.type.dtype}
                    for variable in program.inputs
                ],
                "outputs": [
                    {"ndim": variable.ndim, "dtype": variable.type.dtype}
                    for variable in program.outputs
                ],
            },
            "initial": float(model.initial_point()["mu"]),
            "draws": draws,
            "tune": tune,
            "width": 1.0,
            "maxSteps": 20,
            "seed": seed,
        }
    finally:
        fn.vm.jit_fn.close()

    completed = subprocess.run(
        ["node", str(Path(__file__).with_name("slice_sampler.mjs"))],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    result["posterior_mean"] = float(observed.sum() / (1 + len(observed)))
    result["posterior_sd"] = float(1 / np.sqrt(1 + len(observed)))
    return result


if __name__ == "__main__":
    result = sample_normal_mean()
    samples = np.asarray(result["samples"])
    sys.stdout.write(
        json.dumps(
            {
                "mean": float(samples.mean()),
                "sd": float(samples.std()),
                "expected_mean": result["posterior_mean"],
                "expected_sd": result["posterior_sd"],
                "evaluations": result["evaluations"],
                "sample_seconds": result["sampleSeconds"],
            },
            indent=2,
        )
        + "\n"
    )
