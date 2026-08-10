# MME Implementation Feasibility

The original MME method uses a temperature-scaled cosine-similarity classifier and adversarially maximizes unlabeled conditional entropy with respect to the classifier while minimizing it with respect to the feature encoder. The frozen GAPS model exposes normalized features but retains a conventional biased linear classifier. Replacing that layer would change the registered architecture and its source endpoint semantics.

This Gate therefore uses **MME-compatible (existing linear head)**: gradient reversal applies the minimax entropy direction through the existing classifier, with the official default entropy weight 0.1, while retaining the canonical backbone, head, Adam 5e-4 optimizer, and exactly 100 optimizer updates. It is an algorithm-compatible comparator, not an exact reproduction of the ICCV implementation. No target-test value selected this design or coefficient.

Primary references:
- https://openaccess.thecvf.com/content_ICCV_2019/html/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.html
- https://github.com/VisionLearningGroup/SSDA_MME
