import subprocess
import sys


def test_import_time_pytensor_tensor(benchmark):
    def run():
        subprocess.run(
            [sys.executable, "-c", "import pytensor.tensor"],
            check=True,
            capture_output=True,
        )

    benchmark.pedantic(run, rounds=5, iterations=1)
