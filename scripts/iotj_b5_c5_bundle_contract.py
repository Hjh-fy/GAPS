"""Shared B5/C5 deployment-bundle role and fail-closed policy."""

RUNTIME_ASSET_KEYS = (
    "classifier",
    "r4_policy",
    "h23_reference",
    "qc_risk_policy",
    "qc_component_calibrator",
    "qc_feature_reference",
    "qc_risk_selection",
    "feature_schema",
    "class_map",
    "normalization",
)
PARITY_REFERENCE_KEY = "offline_reference_1360"
REQUIRED_KEYS = RUNTIME_ASSET_KEYS + (PARITY_REFERENCE_KEY,)
FORBIDDEN_TOKENS = ("c3", "c4", "r3ak16", "h8+c4", "p4")
