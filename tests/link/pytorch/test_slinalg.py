import numpy as np
import pytest

import pytensor
from pytensor.tensor import tensor
from pytensor.tensor.slinalg import cholesky


torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "cov_batch_shape", [(), (1000,), (4, 1000)], ids=lambda arg: f"cov:{arg}"
)
def test_batched_mvnormal_logp_and_dlogp(cov_batch_shape):
    rng = np.random.default_rng(sum(map(ord, "batched_mvnormal")))

    cov = tensor("cov", shape=(*cov_batch_shape, 10, 10))

    test_values = np.eye(cov.type.shape[-1]) * np.abs(rng.normal(size=cov.type.shape))

    chol_cov = cholesky(cov, lower=True, on_error="raise")

    fn = pytensor.function([cov], [chol_cov], mode="PYTORCH")
    assert np.all(np.isclose(fn(test_values), np.linalg.cholesky(test_values)))
