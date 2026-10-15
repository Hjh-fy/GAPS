# P0-U Label Access Audit

## Verdict: PASS pending experiment audit

- U1/U2 training APIs receive `torch.Tensor` target features only; non-tensor batches fail closed.
- No target class-label parameter exists in either adaptation function.
- U1 target CE and target prototype anchor are unavailable; class-conditional CORAL, class MMD, same-class-phase MMD and pseudo labels are disabled.
- U2 pseudo labels originate only from frozen source-teacher argmax predictions at the predeclared threshold 0.90.
- Calibration truth was opened only after both 100-step training branches completed, for one post-hoc pseudo-label precision audit: coverage 0.965625, precision 0.355987.
- C5 sealed test was opened after both branches and used only for final evaluation.
- No early stopping, threshold selection, hyperparameter search, or checkpoint selection occurred.
