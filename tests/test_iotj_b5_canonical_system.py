from __future__ import annotations


def test_maps_fitins_proxy_to_logical_client_for_round() -> None:
    from scripts.summarize_iotj_b5_canonical_system import map_fitins_to_clients

    server_events = [
        {
            "event_type": "flower_fitins_prepared",
            "round": 1,
            "client_id": "proxy-c1",
            "payload": {"proxy_id": "proxy-c1", "downlink_audit": {"application_message_bytes": 101}},
        },
        {
            "event_type": "flower_fitins_prepared",
            "round": 1,
            "client_id": "proxy-c2",
            "payload": {"proxy_id": "proxy-c2", "downlink_audit": {"application_message_bytes": 102}},
        },
        {
            "event_type": "flower_fitres_available",
            "round": 1,
            "client_id": "C1",
            "payload": {"proxy_id": "proxy-c1"},
        },
        {
            "event_type": "flower_fitres_available",
            "round": 1,
            "client_id": "C2",
            "payload": {"proxy_id": "proxy-c2"},
        },
    ]

    mapped = map_fitins_to_clients(server_events, round_idx=1)

    assert mapped["C1"]["payload"]["downlink_audit"]["application_message_bytes"] == 101
    assert mapped["C2"]["payload"]["downlink_audit"]["application_message_bytes"] == 102
