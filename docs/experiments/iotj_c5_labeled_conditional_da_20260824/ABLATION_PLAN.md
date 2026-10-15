# Ablation Plan

| Factor | Level | Target label use | Target CE | Held constants | Expected evidence |
|---|---|---|---:|---|---|
| Global alignment | CORAL+TCE | CE | 1.0 | source/checkpoint/split/steps/lr/seed | versus CORAL and Target-CE |
| Global alignment | MMD+TCE | CE | 1.0 | same | versus MMD and Target-CE |
| Global alignment | DANN+TCE | CE | 1.0 | same | versus DANN and Target-CE |
| Conditional alignment | CC-CORAL | class grouping only | 0.0 | same | versus CORAL |
| Conditional alignment | Class-MMD | class grouping only | 0.0 | same | versus MMD |
| Conditional alignment | CC-DANN | class grouping only | 0.0 | same | versus DANN |

Stopping rule: finish six fixed step100 endpoints, evaluate sealed C5 test
once, analyze, audit, and stop without tuning or method expansion.

