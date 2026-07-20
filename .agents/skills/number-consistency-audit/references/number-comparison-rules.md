# Number Comparison Rules

Compare only after matching experiment ID, metric name, client/gas/sample scope, aggregation, seed set, unit, direction, and calibration/QC/routing state. Preserve raw precision in the canonical source. A displayed value may differ only by declared rounding; recomputing from a rounded value is not valid. Percent and fraction forms require explicit conversion. Treat stale, swapped, sign-flipped, unit-shifted, and scope-shifted values as discrepancies, not rounding.
