from __future__ import annotations

from scripts import diagnose_iotj_a003_timing as diagnosis


def _event(event_type: str, round_idx: int, **payload):
    return {
        "event_type": event_type,
        "round": round_idx,
        "client_id": payload.pop("client_id", None),
        "monotonic_ns": payload.pop("monotonic_ns", round_idx * 1_000_000_000),
        "payload": payload,
    }


def test_build_round_rows_uses_parallel_client_critical_path_and_residual_waiting():
    server = [
        _event("fit_round_end", 1, fit_round_wall_ns=1000),
        _event(
            "server_aggregate_end",
            1,
            server_aggregate_fit_total_ns=300,
            server_da_total_ns=250,
            server_aggregate_non_da_ns=50,
        ),
    ]
    c1 = [
        _event("client_fit_start", 1, client_id="C1", monotonic_ns=100),
        _event("client_fit_end", 1, client_id="C1", monotonic_ns=400, client_fit_callback_ns=300),
        _event("client_train_end", 1, client_id="C1", client_train_core_ns=200),
    ]
    c2 = [
        _event("client_fit_start", 1, client_id="C2", monotonic_ns=100),
        _event("client_fit_end", 1, client_id="C2", monotonic_ns=600, client_fit_callback_ns=500),
        _event("client_train_end", 1, client_id="C2", client_train_core_ns=400),
    ]

    rows = diagnosis.build_round_rows(server, c1, c2)

    assert rows == [
        {
            "round": 1,
            "round_wall_s": 1e-6,
            "pi_client_train_core_s": 2e-7,
            "pi_client_fit_callback_s": 3e-7,
            "pc_client_train_core_s": 4e-7,
            "pc_client_fit_callback_s": 5e-7,
            "server_aggregate_total_s": 3e-7,
            "server_da_s": 2.5e-7,
            "server_non_da_s": 5e-8,
            "client_waiting_or_sync_residual_s": 2e-7,
        }
    ]
