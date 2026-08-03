from __future__ import annotations

import pytest


@pytest.mark.parametrize("server_round", [1, 2, 3, 4, 5])
def test_rounds_one_through_five_are_complete_fedavg_warmup(server_round: int) -> None:
    from gaps_flower.strategy import selective_aggregation_phase

    assert selective_aggregation_phase(server_round, warmup=5) == "fedavg_warmup"


@pytest.mark.parametrize("server_round", [6, 7, 25])
def test_round_six_onward_is_selective(server_round: int) -> None:
    from gaps_flower.strategy import selective_aggregation_phase

    assert selective_aggregation_phase(server_round, warmup=5) == "selective"


def test_invalid_selective_warmup_round_is_rejected() -> None:
    from gaps_flower.strategy import selective_aggregation_phase

    with pytest.raises(ValueError, match="server_round"):
        selective_aggregation_phase(0, warmup=5)
