# C5/H8 Runtime Contract B5 v2 Mapping Audit

Status: `blocked` / invalid for parity evidence.

The v2 contract binds `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid/client_5`. Although it contains 1,360 windows, its `(filename, repeat_id)` multiplicities differ from the frozen HC90/HC95 reference by 53 rows in each direction. It must not be used for runtime parity, QC evidence, or deployment claims.

The frozen suite and H8 manifests instead bind `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`, whose C5 test metadata exactly matches both HC references on those keys. A non-overwriting v3 contract must bind that root before any parity execution.
