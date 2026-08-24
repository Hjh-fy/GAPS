# Label Access Audit

- Target calibration API receives exactly two tensors: C5 calibration x and class.
- CORAL+TCE, MMD+TCE, and DANN+TCE activate target CE with fixed weight 1.0.
- CC-CORAL, Class-MMD, and CC-DANN set target CE structurally inactive; class
  labels only construct true-class masks for the alignment objective.
- CC-DANN uses one shared discriminator and per-shared-class GRL/binary-BCE.
- Target phase, concentration, pseudo labels, and test labels are unavailable to
  adaptation. Test labels are opened only for final fixed-endpoint evaluation.
