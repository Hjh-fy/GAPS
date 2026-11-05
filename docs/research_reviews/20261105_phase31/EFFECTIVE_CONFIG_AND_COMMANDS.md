# 真实CLI绑定与尚未闭合的有效配置

以下全部**BLOCKED / DO NOT RUN**，仅解析了原parser，未调用main、未启动服务器。没有伪造--seq-len、--initial-state、--server-batch-size或--eval-split none。SIX_EFFECTIVE_CONFIGS.json包含6组完整args/config/client命令及无法表达字段；B1没有通用合法执行入口，故server_argv=null，不造一个可运行结果。

公共配置：seed42、4类、8通道、num-clients6（容纳ID5，不表示6个训练客户端）、3个phase槽、ce_only、25轮、batch16客户端、LR5e-4；目标Day18 A4固定32窗。B3–B5 sourceCE=1（原代码隐含）、targetCE=1，DA warmup0保证25×100步；所有CORAL/global及class MMD/对抗/proto/consistency/residual/stage/legacy align系数显式0，相关开关关闭。B4/B5 EMA=.8、warmup3、min_scale=.3一致。

B3/B4语义唯一区别use_adapted_as_global；B4/B5唯一区别use_selective_agg。parser实际差分还包含run_name/output_dir这两个身份字段，均逐项保存，未藏其他训练差异。两边UPLOAD_PROTO_STATS=False相同并不满足“共同上传原型”的必要条件，所以生产机制对照仍BLOCKED。

原server实际batch_size32、SEQ_LEN100、新随机初始化（没有S*加载入口）、fraction_evaluate1/min_evaluate_clients=min_clients；实际DA loader另有batch32和源500封顶。client明确--eval-split calibration，数据包根本不含test_*；这避免默认test读取，但evaluate回调仍运行Calibration，不得隐去，也不得用日志选轮。若要求完全无evaluate回调，需要薄入口显式fraction_evaluate0，现有CLI不能假造该开关。


## P3-PILOT-C5-A4-B0-S42 — BLOCKED

```powershell
python -m gaps_flower.server_app --strategy fedavg --rounds 25 --min-clients 2 --output-dir UNRUN/P3-PILOT-C5-A4-B0-S42 --run-name P3-PILOT-C5-A4-B0-S42 --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3 --da-preset none --proto-ema-alpha 0.8 --use-selective-agg False --selective-warmup 3 --selective-min-scale 0.3 --use-proto-mmd False --use-domain-adapt False --use-adapted-as-global False --domain-adapt-steps 100 --domain-adapt-warmup 0 --strict-calibration-split True --da-use-coral False --da-use-mmd False --da-use-adversarial False --da-use-align-reg-legacy False --da-lambda-target-ce 0 --da-target-ce-label-smoothing 0 --da-target-ce-class-balanced False --da-server-opt-lr 0.0005 --da-lambda-coral 0 --da-lambda-global-mmd 0 --da-lambda-class-mmd 0 --da-lambda-proto-anchor 0 --da-lambda-adv 0 --da-lambda-proto 0 --da-lambda-consistency 0 --da-lambda-residual 0 --da-lambda-proto-mmd 0 --da-lambda-stage-mmd 0 --da-lambda-align-reg-legacy 0
```

```powershell
python -m gaps_flower.client_app --client-id 1 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```

```powershell
python -m gaps_flower.client_app --client-id 2 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```


## P3-PILOT-C5-A4-B1-S42 — BLOCKED

B1可复用的旧真实CLI模板（新ID会在原validate_entry_spec拒绝；SSTAR_PENDING和SPEC_PENDING为未生成输入，不可执行）：
```powershell
python -m gaps_flower.fedongas_strict_entry target --data-root C5_A4_DATA --initial-state SSTAR_PENDING --spec SPEC_PENDING --output UNRUN_B1
```
该旧入口要求random initialization语义且batch32/旧ID，不能把已训练S*伪标random绕过验证。新的目标独立微调应复用其循环和原Client，先绑定25×100次、batch16及optimizer重置/同序采样再复检。


## P3-PILOT-C5-A4-B2-S42 — BLOCKED

```powershell
python -m gaps_flower.server_app --strategy fedavg --rounds 25 --min-clients 3 --output-dir UNRUN/P3-PILOT-C5-A4-B2-S42 --run-name P3-PILOT-C5-A4-B2-S42 --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3 --da-preset none --proto-ema-alpha 0.8 --use-selective-agg False --selective-warmup 3 --selective-min-scale 0.3 --use-proto-mmd False --use-domain-adapt False --use-adapted-as-global False --domain-adapt-steps 100 --domain-adapt-warmup 0 --strict-calibration-split True --da-use-coral False --da-use-mmd False --da-use-adversarial False --da-use-align-reg-legacy False --da-lambda-target-ce 0 --da-target-ce-label-smoothing 0 --da-target-ce-class-balanced False --da-server-opt-lr 0.0005 --da-lambda-coral 0 --da-lambda-global-mmd 0 --da-lambda-class-mmd 0 --da-lambda-proto-anchor 0 --da-lambda-adv 0 --da-lambda-proto 0 --da-lambda-consistency 0 --da-lambda-residual 0 --da-lambda-proto-mmd 0 --da-lambda-stage-mmd 0 --da-lambda-align-reg-legacy 0
```

```powershell
python -m gaps_flower.client_app --client-id 1 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```

```powershell
python -m gaps_flower.client_app --client-id 2 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```

```powershell
python -m gaps_flower.client_app --client-id 5 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_5" --local-epochs 50 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```


## P3-PILOT-C5-A4-B3-S42 — BLOCKED

```powershell
python -m gaps_flower.server_app --strategy gaps --rounds 25 --min-clients 2 --output-dir UNRUN/P3-PILOT-C5-A4-B3-S42 --run-name P3-PILOT-C5-A4-B3-S42 --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3 --da-preset none --proto-ema-alpha 0.8 --use-selective-agg False --selective-warmup 3 --selective-min-scale 0.3 --use-proto-mmd False --use-domain-adapt True --use-adapted-as-global False --domain-adapt-steps 100 --domain-adapt-warmup 0 --strict-calibration-split True --da-use-coral False --da-use-mmd False --da-use-adversarial False --da-use-align-reg-legacy False --da-lambda-target-ce 1 --da-target-ce-label-smoothing 0 --da-target-ce-class-balanced False --da-server-opt-lr 0.0005 --da-lambda-coral 0 --da-lambda-global-mmd 0 --da-lambda-class-mmd 0 --da-lambda-proto-anchor 0 --da-lambda-adv 0 --da-lambda-proto 0 --da-lambda-consistency 0 --da-lambda-residual 0 --da-lambda-proto-mmd 0 --da-lambda-stage-mmd 0 --da-lambda-align-reg-legacy 0 --server-val-data "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1,RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --server-calib-data "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_5"
```

```powershell
python -m gaps_flower.client_app --client-id 1 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```

```powershell
python -m gaps_flower.client_app --client-id 2 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```


## P3-PILOT-C5-A4-B4-S42 — BLOCKED

```powershell
python -m gaps_flower.server_app --strategy gaps --rounds 25 --min-clients 2 --output-dir UNRUN/P3-PILOT-C5-A4-B4-S42 --run-name P3-PILOT-C5-A4-B4-S42 --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3 --da-preset none --proto-ema-alpha 0.8 --use-selective-agg False --selective-warmup 3 --selective-min-scale 0.3 --use-proto-mmd False --use-domain-adapt True --use-adapted-as-global True --domain-adapt-steps 100 --domain-adapt-warmup 0 --strict-calibration-split True --da-use-coral False --da-use-mmd False --da-use-adversarial False --da-use-align-reg-legacy False --da-lambda-target-ce 1 --da-target-ce-label-smoothing 0 --da-target-ce-class-balanced False --da-server-opt-lr 0.0005 --da-lambda-coral 0 --da-lambda-global-mmd 0 --da-lambda-class-mmd 0 --da-lambda-proto-anchor 0 --da-lambda-adv 0 --da-lambda-proto 0 --da-lambda-consistency 0 --da-lambda-residual 0 --da-lambda-proto-mmd 0 --da-lambda-stage-mmd 0 --da-lambda-align-reg-legacy 0 --server-val-data "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1,RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --server-calib-data "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_5"
```

```powershell
python -m gaps_flower.client_app --client-id 1 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```

```powershell
python -m gaps_flower.client_app --client-id 2 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```


## P3-PILOT-C5-A4-B5-S42 — BLOCKED

```powershell
python -m gaps_flower.server_app --strategy gaps --rounds 25 --min-clients 2 --output-dir UNRUN/P3-PILOT-C5-A4-B5-S42 --run-name P3-PILOT-C5-A4-B5-S42 --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3 --da-preset none --proto-ema-alpha 0.8 --use-selective-agg True --selective-warmup 3 --selective-min-scale 0.3 --use-proto-mmd False --use-domain-adapt True --use-adapted-as-global True --domain-adapt-steps 100 --domain-adapt-warmup 0 --strict-calibration-split True --da-use-coral False --da-use-mmd False --da-use-adversarial False --da-use-align-reg-legacy False --da-lambda-target-ce 1 --da-target-ce-label-smoothing 0 --da-target-ce-class-balanced False --da-server-opt-lr 0.0005 --da-lambda-coral 0 --da-lambda-global-mmd 0 --da-lambda-class-mmd 0 --da-lambda-proto-anchor 0 --da-lambda-adv 0 --da-lambda-proto 0 --da-lambda-consistency 0 --da-lambda-residual 0 --da-lambda-proto-mmd 0 --da-lambda-stage-mmd 0 --da-lambda-align-reg-legacy 0 --server-val-data "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1,RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --server-calib-data "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_5"
```

```powershell
python -m gaps_flower.client_app --client-id 1 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_1" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```

```powershell
python -m gaps_flower.client_app --client-id 2 --data-root "RESEARCH_WORKTREE\research\fedongas_phase31_20261105\data\client_2" --local-epochs 1 --batch-size 16 --eval-split calibration --profile ce_only --seed 42 --num-classes 4 --input-dim 8 --num-clients 6 --num-phases 3
```


## 一次源预训练模板（只登记，BLOCKED）
从B0真实server/client模板复制，run-name改P31-SSTAR-DAY7-S42-PLAN、两个源客户端保持不变、起点明确为random。它不是从S*续训；仍需薄入口把实际50×8输入合同/元数据写对并输出祖先合同。没有额外调参或新增源训练轮数。


## 既有专用入口复用范围
strict_entry已有initial-state与独立upload-stats；其whitelist固定旧7组batch32并要求random，实际执行新ID验证得到Unapproved strict experiment configuration，未开权重文件。matched_entry已有接收/上传指纹链，但限定旧M0–M3、两个源ID1/2与[2360,2360]，不满足840/1680或第三目标客户端。SOURCE snapshot自身缺少一个导入protocol，白名单元数据测试只借用了另一归档的原常量；未宣称全依赖runner通过。不能删除guard、篡改旧ID或借旧reference文件让新实验过门禁。
