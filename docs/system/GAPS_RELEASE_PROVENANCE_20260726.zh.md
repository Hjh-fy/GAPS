# GAPS IoT-J Release Provenance 收口

日期：2026-07-26
状态：`RELEASE_PROVENANCE_CLOSED_ARCHIVE_PENDING`

## 1. 本轮结论

本轮已经把最终系统的外部资产身份从散落的绝对路径和历史结果目录，收口为一个
版本化、可机器校验的 provenance 合同：

- 合同：`docs/system/iotj_release_provenance_manifest_20260726.json`
- 校验器：`scripts/verify_iotj_release_provenance.py`
- 合同 schema：`iotj.release_provenance.v1`
- 合同状态：`PROVENANCE_LOCKED`
- 资产记录：35 项，合计 37,309,036 bytes
- v4/v5/v5-QC2 三个 loader closure 的并集：24 项，合计 4,105,879 bytes

35 项资产均已按 `bytes + SHA256` 完成只读验证。验证过程没有导入 runtime、没有
反序列化 checkpoint、没有运行推理或评估，也没有改写任何冻结文件。

这完成的是 release provenance，不等于完成 release archive，也不等于证明
clean-checkout deployment。

## 2. 三条运行资产闭包

| Profile | 身份 | 当前 loader 的实际依赖 | 结论 |
|---|---|---|---|
| `runtime_v4_loader_closure` | formal C5 Runtime-v4 baseline | contract、manifest、10 个 bundle assets、1360 行 offline parity reference | 身份完整，但 audit evidence 与启动耦合 |
| `runtime_v5_core_loader_closure` | final Runtime-v5 regression core | contract、manifest、classifier、Federated H1、target Ridge、calibration lock | 身份完整；仍含绝对路径合同 |
| `runtime_v5_qc2_candidate_loader_closure` | `VALID_CANDIDATE_NOT_PROMOTED` | v5 core closure、QC bundle、policy、HC95/HC90 contracts | 身份完整；不是正式 final runtime |

`formal_release_audit_inventory` 是审计集合，不是部署分发 profile。它还包含正式
C5 test 输入、HC95/HC90 1360 行记录、offline references 和 parity/decision
记录。

## 3. 新发现的工程边界

### 3.1 Runtime-v4 启动依赖测试派生审计文件

`load_c5_h8_bundle()` 在构建 runtime 时会调用 `_verify_reference()`，因此即使在线
推理不消费 1360 行 offline parity reference，当前 loader 仍要求该文件存在且哈希
正确。

这意味着现有 v4 loader closure 不是纯部署最小集。后续若制作默认 release archive，
必须二选一：

1. 保留 loader 不变，明确将其发布为“audit-augmented v4 package”；或
2. 新增版本化、向后兼容的 portable binding/loader，将 parity reference 改为显式
   audit-only 依赖。

在作出该工程决定前，不应把正式 test 派生文件静默装入默认部署包。

### 3.2 现有 runtime contracts 不可直接跨路径恢复

v4、v5 core 和 v5 QC2 contracts/manifest 中均存在冻结的绝对路径。原始文件必须
保持字节不变，因此不能原地改成相对路径。将这些文件按相同相对目录复制到另一台
机器，并不会自动使 loader 可用。

后续 portable release 应生成独立的版本化 binding contracts，并同时保留：

- 原冻结 contract 的 SHA256；
- portable binding 的生成规则；
- portable binding 的新 SHA256；
- 原资产与 release 内副本的逐项双向映射。

portable binding 是工程派生物，不能替代或覆盖冻结证据合同。

## 4. 资产分类

| Evidence class | 含义 |
|---|---|
| `RUNTIME_ASSET` | 模型、Ridge heads、policy、bundle manifest 等运行资产 |
| `LOADER_COUPLED_AUDIT_ASSET` | 语义上是审计/lineage，但当前 loader 启动时强制校验 |
| `PROVENANCE_AUDIT_ASSET` | contract、lock、decision/parity receipt 等审计身份 |
| `FORMAL_TEST_MATERIAL` | 1360 行 test 输入、reference 或 test-bound records |

`contains_formal_test_material=true` 的记录不得进入未来默认 archive，除非 package
明确命名为 audit/reproduction package，并在发布边界中披露。

## 5. 校验命令

仅验证三条 loader closure：

```powershell
python scripts/verify_iotj_release_provenance.py `
  docs/system/iotj_release_provenance_manifest_20260726.json `
  --profile runtime_v4_loader_closure `
  --profile runtime_v5_core_loader_closure `
  --profile runtime_v5_qc2_candidate_loader_closure
```

验证完整审计 inventory：

```powershell
python scripts/verify_iotj_release_provenance.py `
  docs/system/iotj_release_provenance_manifest_20260726.json `
  --profile formal_release_audit_inventory
```

任一文件缺失、bytes 不符、SHA256 漂移、路径越界、未知 profile 或 manifest schema
变化都会返回非零退出码并标记 `FAIL_CLOSED`。

## 6. 冻结资产确认

原 Runtime-v4 六项冻结身份仍为：

| 资产 | SHA256 |
|---|---|
| v4 bundle manifest | `a2514bd74ba0a98334d146af218922ee84884a53b93b0d4c44414723abee73b5` |
| C5 test features | `7955cb70b24fa86ce109a52ca3b2231ad543b8ba8be0276781ffa03384143a82` |
| C5 test metadata | `9b48459f52698b11fad66c0a2c63c9ede22292555e4bcaa71125e1f7e90097bf` |
| C5 test phase labels | `a69f333c8418fa3bf94c599a2d684cd122b4a46df2ff405bced227b68fcdb8b5` |
| HC95 1360-row reference | `33d04439376852bb976d9a4ed5f09235107b296c5f839c75ed667fdecc598860` |
| HC90 1360-row reference | `6051e7787915e0163ffd815dc089626e751906474c858072c5c0520c615dccb3` |

本轮没有改写上述文件。

## 7. 尚未完成

release provenance 已收口，但下面三项仍是发布阻塞：

1. 生成不覆盖原文件、排除正式 test material 的 portable external-asset archive；
2. 增加 Runtime-v5 独立 inference CLI；
3. 在 fresh checkout + restored archive 上完成静态 import、bundle load、
   synthetic smoke，并生成 clean-checkout deployment receipt。

因此当前只能表述为：

```text
PAPER_EVIDENCE_READY
CODE_CONTRACT_READY
RELEASE_PROVENANCE_CLOSED
RELEASE_ARCHIVE_PENDING
CLEAN_CHECKOUT_DEPLOYMENT_NOT_COMPLETE
```
