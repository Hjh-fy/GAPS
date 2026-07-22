from __future__ import annotations

import numpy as np
import pytest


def test_probability_signature_matching_is_bijective() -> None:
    from scripts.build_iotj_b5_c5_runtime_row_map import match_probability_signatures

    runtime = np.asarray([[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]])
    reference = runtime[[1, 0]] + 1e-6

    assert match_probability_signatures(runtime, reference) == [
        (0, 1, pytest.approx(1e-6)),
        (1, 0, pytest.approx(1e-6)),
    ]


@pytest.mark.parametrize(
    "reference, message",
    [
        (np.asarray([[0.8, 0.1, 0.05, 0.05]] * 2), "duplicate"),
        (np.asarray([[0.7, 0.2, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05]]), "within tolerance"),
    ],
)
def test_probability_signature_matching_fails_closed(reference: np.ndarray, message: str) -> None:
    from scripts.build_iotj_b5_c5_runtime_row_map import match_probability_signatures

    runtime = np.asarray([[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]])
    with pytest.raises(ValueError, match=message):
        match_probability_signatures(runtime, reference)
