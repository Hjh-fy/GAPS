# GAPS IoT-J 中文投稿候选稿 v1 导师审阅清单

审阅对象：`docs/paper/GAPS_IoTJ_submission_candidate_v1.zh.html`
当前边界：只审阅中文叙事、结构和证据表达；不重开实验，不修改 frozen evidence。

## 1. 论文故事与贡献

- [ ] RQ1 是否清楚回答“真实设备联邦分类 + calibration-assisted server adaptation”？
- [ ] RQ2 是否清楚回答“sufficient-statistics Federated H1 + 105D target personalization”？
- [ ] RQ3 是否清楚回答“edge efficiency + selective-output trade-off + calibration boundary”？
- [ ] 三项贡献是否彼此不重复，并与三项 RQ 对应？
- [ ] selective aggregation、QC2、portable release 是否保持为机制/工程证据，而非核心算法创新？
- [ ] 是否接受把 portable release 只放在系统实现与证据边界中，不放入摘要贡献列表？

## 2. 方法身份

- [ ] B5 是否始终被描述为 frozen final classifier？
- [ ] server adaptation 是否始终写作 calibration-assisted，而非 zero-shot/UDA？
- [ ] Federated H1 是否始终指 sufficient-statistics source Ridge reference？
- [ ] target Ridge 是否始终是 104D rich + 1D H1 prediction 的 105D per-gas personalization？
- [ ] 是否明确 target Ridge 不属于 network FedAvg？
- [ ] 是否明确 v4 formal QC、v5 regression core 与 v5 QC2 candidate 是不同运行对象？

## 3. 实验协议与证据边界

- [ ] 是否统一使用 `calibrated-target held-out-window evaluation`？
- [ ] 是否明确 window construction precedes splitting？
- [ ] 是否明确 C5 calibration/test 为 320/1360 windows，按 gas class 与 concentration 分层？
- [ ] 是否明确相同具体 window/sample row 不跨 subset？
- [ ] 是否明确同一 original file 的不同 windows 可跨 calibration/test？
- [ ] 是否避免 original-file-independent、unseen-session、zero-shot 与 completely leakage-free 表述？
- [ ] 是否区分 filename overlap 与 test-label leakage？
- [ ] 是否明确 test labels 不用于 fit/select/refit、alpha/QC threshold/checkpoint selection？

## 4. 关键结论

- [ ] 是否接受 B5 five-seed conclusion 仅限 seeds 42–46、C1/C2→C5？
- [ ] 是否接受 H1 为 simplification-noninferiority choice，而非 absolute accuracy winner？
- [ ] 是否保留 all-prior 在 5/5 routes 上具有更低 S_CC 的不利结果？
- [ ] 是否明确回归 five-seed 只改变 B5 routes，而非所有 regression heads 重训五次？
- [ ] 是否接受 v5 QC2 未晋级、v4 保留 formal baseline 的表述？
- [ ] 是否接受 calibration sensitivity 为 post-freeze descriptive evidence？

## 5. 数字与表图

- [ ] Table I 是否足够概括 dataset/device/protocol？
- [ ] Table II 是否保留 Accuracy/Macro-F1/NLL/ECE 的 mean、sample SD 和 range？
- [ ] Table III 是否同时展示 RG0/RG1/RG2 与非劣决策？
- [ ] Table IV 是否清楚区分 runtime functional identity？
- [ ] 是否同意正文只保留五图四表？
- [ ] Fig. 3 与 Table III 是否形成 paired-route vs summary 的互补，而非重复？
- [ ] Fig. 4 与 Table A5 是否形成 visual trade-off vs full diagnostics 的互补？
- [ ] Fig. 5 与 Table A6 是否保持 group-aware sensitivity vs protocol harmonization 的区别？
- [ ] “28,737→844” 是否只称回归参数变化；classifier 22,765 是否单列？
- [ ] 是否保留 PC p95 未改善这一平台/percentile 例外？

## 6. Legacy 与附录

- [ ] Table A1 是否持续保留 `historical mechanism semantics`、`corrected single-seed screening`、`final five-seed evidence` 三种身份？
- [ ] 是否避免把 historical A7 与 final B5 比较？
- [ ] 是否避免把 legacy table 解释为 corrected final B5 的严格逐组件消融？
- [ ] Table A2 是否清楚区分 five-route per-gas mean 与 seed42 CO-high？
- [ ] Table A3 practical equivalence 是否避免写成数学恒等？
- [ ] Table A4 是否明确 transport bytes 未采集且无 secure aggregation/DP？
- [ ] Table A6 是否避免用 group-aware 10.8724 替换 historical 11.3416？

## 7. 参考文献

- [ ] 是否接受将 original Ref. 9 限定为 experience replay？
- [ ] 是否接受 Refs. 13/14 只说明 residual/attention 通用架构来源？
- [ ] 是否接受将 IEEE IoT-J author guideline 移出 scientific references？
- [ ] 英文稿阶段是否安排 DOI/venue/页码的最后一次人工核验？

## 8. 语言与投稿表达

- [ ] 是否避免“显著”用于未进行统计检验的简化结论？
- [ ] 是否使用“promotion guards 未满足，因此保留 formal baseline”替代“更强 baseline”？
- [ ] 是否避免 all-platform/all-percentile latency improvement？
- [ ] 中英文术语混排是否需要在英文稿前进一步中文化？
- [ ] 摘要是否在不展开 filename overlap 数字的情况下保留评价边界？
- [ ] 结论是否同时报告正面结果与 deployment limitations？

## 9. 导师决定后才执行的下一步

- [ ] 冻结中文候选稿修订意见。
- [ ] 从摘要开始逐节受控英译，不一次性自动翻译全文。
- [ ] 按 `figure_plan_submission_v1.md` 审核重绘需求；任何重绘只读取 frozen CSV。
- [ ] 压缩表格并映射 IEEEtran 双栏结构。
- [ ] 完成 BibTeX 与参考文献人工终审。
- [ ] 生成 LaTeX/PDF 后做版面、交叉引用、公式、单位和字体检查。

## 当前不需要的事项

- [x] 不需要新训练。
- [x] 不需要重新推理、评估或 benchmark。
- [x] 不需要重开 C5 test。
- [x] 不需要修改 runtime、QC 或 thresholds。
- [x] 不需要补 FedProx/FedAdam/SCAFFOLD、多目标或 original-file-level experiments。
