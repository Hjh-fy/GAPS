from __future__ import annotations

import pytest

from scripts.materialize_iotj_federated_h1_topology import (
    ALPHAS,
    FORBIDDEN_SERVER_KEYS,
    reject_forbidden_payload,
    ridge_coef,
)

import numpy as np


def test_protocol_alpha_grid_is_frozen() -> None:
    assert ALPHAS == (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@pytest.mark.parametrize("key", sorted(FORBIDDEN_SERVER_KEYS))
def test_server_rejects_raw_or_sample_level_fields(key: str) -> None:
    with pytest.raises(ValueError, match="forbidden server payload"):
        reject_forbidden_payload({"nested": [{key: [1, 2, 3]}]})


def test_intercept_is_not_regularized() -> None:
    a = np.eye(3)
    b = np.ones(3)
    coef = ridge_coef(a, b, 9.0)
    assert coef[0] == pytest.approx(1.0)
    assert np.allclose(coef[1:], 0.1)
