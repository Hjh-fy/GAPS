import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outDir = "results/iotj_advisor_metrics_20260721";
await fs.mkdir(outDir, { recursive: true });

const wb = Workbook.create();
console.log("stage:create-workbook");
const overview = wb.worksheets.add("导师汇报总览");
const system = wb.worksheets.add("系统指标_B2");
const algorithm = wb.worksheets.add("算法性能");
const gaps = wb.worksheets.add("模型与部署缺口");
const historical = wb.worksheets.add("历史部署参考");
const evidence = wb.worksheets.add("证据来源");
console.log("stage:add-sheets");

const navy = "#17365D";
const blue = "#2F75B5";
const lightBlue = "#D9EAF7";
const green = "#E2F0D9";
const amber = "#FFF2CC";
const red = "#FCE4D6";
const gray = "#E7E6E6";
const white = "#FFFFFF";
const border = { preset: "all", style: "thin", color: "#D9E2F3" };

function title(sheet, text, subtitle, lastCol) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastCol}1`).merge();
  sheet.getRange("A1").values = [[text]];
  sheet.getRange(`A1:${lastCol}1`).format = {
    fill: navy,
    font: { bold: true, color: white, size: 18 },
    rowHeight: 30,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastCol}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastCol}2`).format = {
    fill: lightBlue,
    font: { color: "#1F1F1F", italic: true, size: 10 },
    wrapText: true,
    rowHeight: 36,
    verticalAlignment: "center",
  };
}

function header(range) {
  range.format = {
    fill: blue,
    font: { bold: true, color: white },
    borders: border,
    wrapText: true,
    verticalAlignment: "center",
  };
}

function body(range) {
  range.format = { borders: border, wrapText: true, verticalAlignment: "top" };
}

title(
  overview,
  "GAPS IoT-J：导师汇报初步指标总览",
  "定位：已有算法证据 + B2 真实 ECS–Pi–ECS 两客户端 25 轮诊断证据。B2 attempt 完成训练但 Controller 回收阶段失败，因此系统数值不可称 canonical；B5 正在运行。",
  "H",
);
overview.getRange("A4:H4").values = [["类别", "指标", "B2", "B5", "单位", "当前判断", "证据等级", "论文边界"]];
header(overview.getRange("A4:H4"));
const overviewRows = [
  ["分类", "Accuracy", 0.9926470588, 0.9889705882, "%", "高", "historical seed-42 screening", "不可写成五种子 mean/std"],
  ["分类", "Macro-F1", 0.9926573079, 0.9889895838, "%", "高", "historical seed-42 screening", "同上"],
  ["分类", "NLL", 0.0690175779, 0.0703653693, "", "较低", "historical seed-42 screening", "缺正式多种子不确定性"],
  ["分类", "ECE", 0.0066814276, 0.0092726690, "%", "校准误差低", "historical seed-42 screening", "缺正式多种子不确定性"],
  ["回归", "R4/H8 FULL RMSE", 14.6564, 17.4473, "ppm", "B2 优于 B5", "formal C5 regression", "可支持正式回归结果"],
  ["回归/QC", "HC90 accepted RMSE", 11.5866, 15.3599, "ppm", "QC 后误差下降", "formal C5 regression/QC", "同时报告 yield"],
  ["回归/QC", "HC90 QC Yield", 0.8949, 0.8824, "%", "约 88–90% 自动覆盖", "formal C5 regression/QC", "被拒样本不计 accepted RMSE"],
  ["真实系统", "25轮应用层通信总量", 16.760642, null, "MiB", "量级可管理", "B2 recovered diagnostic", "无 transport bytes/对照，不能声称通信最优"],
  ["真实系统", "平均 round wall", 241.6543, null, "s", "仍偏慢", "B2 recovered diagnostic", "B5 待运行完成"],
  ["真实系统", "server DA 占 round wall", 0.679266, null, "%", "当前主瓶颈", "B2 recovered diagnostic", "真实拓扑的诊断结论"],
  ["边缘资源", "Pi peak RSS", 518.40625, null, "MiB", "可运行", "B2 recovered diagnostic", "非最终推理 RSS"],
  ["边缘资源", "Pi peak temperature", 62.25, null, "°C", "无热节流", "B2 recovered diagnostic", "约100分钟训练，不等同6h长稳"],
  ["可观测性", "Observer overhead ratio", 0.00100733, null, "%", "很小", "B2 recovered diagnostic", "测量开销约0.10%"],
  ["模型", "当前分类 checkpoint 文件", 184201, null, "bytes", "约0.176 MiB", "B2 current checkpoint diagnostic", "不是 final deployment bundle"],
];
overview.getRange(`A5:H${4 + overviewRows.length}`).values = overviewRows;
body(overview.getRange(`A5:H${4 + overviewRows.length}`));
overview.getRange("C5:D6").format.numberFormat = "0.0000%";
overview.getRange("C8:D8").format.numberFormat = "0.0000%";
overview.getRange("C11:D11").format.numberFormat = "0.00%";
overview.getRange("C14:D14").format.numberFormat = "0.0%";
overview.getRange("C17:D17").format.numberFormat = "0.000%";
overview.getRange(`F5:F${4 + overviewRows.length}`).conditionalFormats.add("containsText", { text: "高", format: { fill: green } });
overview.getRange(`F5:F${4 + overviewRows.length}`).conditionalFormats.add("containsText", { text: "偏慢", format: { fill: amber } });
overview.getRange(`G5:G${4 + overviewRows.length}`).conditionalFormats.add("containsText", { text: "diagnostic", format: { fill: amber } });
overview.getRange(`G5:G${4 + overviewRows.length}`).conditionalFormats.add("containsText", { text: "formal", format: { fill: green } });
overview.getRange("A21:H21").merge();
overview.getRange("A21").values = [["一句话结论：分类与回归精度已有较强初步/正式证据；真实系统已证明可运行，但 B2 系统数值仍是 recovered diagnostic，最终主文系统表应等待 B5 完成并明确两次代表性运行的证据边界。"]];
overview.getRange("A21:H21").format = { fill: amber, font: { bold: true }, wrapText: true, rowHeight: 44, borders: border };
overview.freezePanes.freezeRows(4);
overview.getRange("A1:H21").format.columnWidth = 16;
overview.getRange("F:H").format.columnWidth = 24;
console.log("stage:overview");

title(system, "B2 真实三机 25 轮系统诊断", "拓扑：Alibaba ECS server + Raspberry Pi C1 + Alibaba ECS C2；25 rounds，5 local epochs，batch 32。attempt a006 完成训练但回收阶段失败，原始证据已手工回收并通过结构验证。", "G");
system.getRange("A4:G4").values = [["分组", "指标", "数值", "单位", "参考/拆分", "表现判断", "证据边界"]];
header(system.getRange("A4:G4"));
const sysRows = [
  ["通信", "Application downlink", 8764216, "bytes", "25 rounds", "—", "serialized application layer"],
  ["通信", "Application uplink", 8810591, "bytes", "25 rounds", "—", "serialized application layer"],
  ["通信", "Application total", 17574807, "bytes", "16.760642 MiB", "可管理但无对照", "不是 transport wire bytes"],
  ["通信", "Mean per round", 702992.28, "bytes/round", "", "稳定量级", "B2 only"],
  ["时延", "Round wall mean", 241.6543, "s", "p50 228.0569; p95 269.6173", "偏慢", "B2 only"],
  ["时延", "Round wall total", 6041.3578, "s", "100.69 min", "完成25轮", "B2 only"],
  ["时延", "Server DA mean", 164.1475, "s/round", "67.93% of wall", "主瓶颈", "B2 only"],
  ["时延", "Server non-DA mean", 0.0681, "s/round", "", "很小", "B2 only"],
  ["时延", "Pi C1 local train mean", 42.0855, "s/round", "", "次要", "B2 only"],
  ["时延", "ECS C2 local train mean", 76.6098, "s/round", "", "次要", "B2 only"],
  ["Pi资源", "Active RSS mean", 514.2308, "MiB", "", "可运行", "training overlap"],
  ["Pi资源", "Peak RSS", 518.40625, "MiB", "", "可运行", "training overlap"],
  ["Pi资源", "Host CPU mean / peak", 84.7411, "%", "peak 91.2873%", "利用充分", "host-level sampler"],
  ["Pi资源", "Temperature mean / peak", 57.6917, "°C", "peak 62.25°C", "无热节流", "约100分钟"],
  ["ECS-C2资源", "Active RSS mean / peak", 522.2723, "MiB", "peak 523.9180 MiB", "可运行", "training overlap"],
  ["ECS-C2资源", "Host CPU mean / peak", 49.9478, "%", "peak 54.7388%", "仍有余量", "2 vCPU host-level"],
  ["Observer", "Total observer overhead", 6085.66168, "ms", "0.1007% of wall", "很小", "serialization + observer I/O"],
];
system.getRange(`A5:G${4 + sysRows.length}`).values = sysRows;
body(system.getRange(`A5:G${4 + sysRows.length}`));
system.getRange("C5:C21").format.numberFormat = "0.000";
system.freezePanes.freezeRows(4);
system.getRange("A1:G21").format.columnWidth = 19;
system.getRange("E:G").format.columnWidth = 24;
system.getRange("I4:J8").values = [
  ["阶段", "mean s/round"],
  ["Server DA", 164.1475],
  ["ECS-C2 local train", 76.6098],
  ["Pi C1 local train", 42.0855],
  ["Server non-DA", 0.0681],
];
header(system.getRange("I4:J4"));
body(system.getRange("I5:J8"));
const timingChart = system.charts.add("bar", system.getRange("I4:J8"));
timingChart.title = "B2 每轮主要阶段耗时（并行客户端不可直接相加）";
timingChart.hasLegend = false;
timingChart.setPosition("I10", "P25");
console.log("stage:system");

title(algorithm, "算法性能证据", "分类为 historical seed-42 screening；回归/QC 为已完成正式 C5 R0–R7 与 FULL/HC95/HC90 结果。", "J");
algorithm.getRange("A4:J4").values = [["任务", "方案", "集合/QC", "N", "Accuracy", "Macro-F1", "NLL", "ECE", "RMSE ppm", "NRMSE"]];
header(algorithm.getRange("A4:J4"));
const algRows = [
  ["分类", "B2", "C5 test", 1360, 0.9926470588, 0.9926573079, 0.0690175779, 0.0066814276, null, null],
  ["分类", "B5", "C5 test", 1360, 0.9889705882, 0.9889895838, 0.0703653693, 0.0092726690, null, null],
  ["回归", "B2 R4/H8", "FULL", 1360, null, null, null, null, 14.6564, 0.1059],
  ["回归", "B5 R4/H8", "FULL", 1360, null, null, null, null, 17.4473, 0.1352],
  ["回归/QC", "B2 R4/H8", "HC90 accepted", 1217, null, null, null, null, 11.5866, 0.0747],
  ["回归/QC", "B5 R4/H8", "HC90 accepted", 1200, null, null, null, null, 15.3599, 0.1151],
];
algorithm.getRange("A5:J10").values = algRows;
body(algorithm.getRange("A5:J10"));
algorithm.getRange("E5:F6").format.numberFormat = "0.0000%";
algorithm.getRange("H5:H6").format.numberFormat = "0.0000%";
algorithm.getRange("I5:J10").format.numberFormat = "0.0000";
algorithm.getRange("A12:F12").values = [["方案", "Ethanol recall", "CO recall", "Ethylene recall", "Methane recall", "证据等级"]];
header(algorithm.getRange("A12:F12"));
algorithm.getRange("A13:F14").values = [
  ["B2", 0.99117647, 1.0, 0.98235294, 0.99705882, "historical seed-42 screening"],
  ["B5", 0.98235294, 0.99411765, 0.98235294, 0.99705882, "historical seed-42 screening"],
];
body(algorithm.getRange("A13:F14"));
algorithm.getRange("B13:E14").format.numberFormat = "0.0000%";
algorithm.getRange("A16:J17").merge(true);
algorithm.getRange("A16").values = [["分类边界：B2/B5 当前均不能表述为正式五种子结论。"]];
algorithm.getRange("A17").values = [["回归边界：R4/H8 FULL 与 operational QC 可作为当前正式算法证据，但不等同于端到端全流程均为联邦训练。"]];
algorithm.getRange("A16:J17").format = { fill: amber, font: { bold: true }, wrapText: true, borders: border };
algorithm.freezePanes.freezeRows(4);
algorithm.getRange("A1:J17").format.columnWidth = 16;
console.log("stage:algorithm");

title(gaps, "模型、推理与投稿证据缺口", "空缺不是零值：尚未完成的指标明确写为“待测”，避免把 legacy runtime 或当前训练 checkpoint 误作 final C5 deployment。", "G");
gaps.getRange("A4:G4").values = [["项目", "当前状态", "已有数值", "为什么重要", "下一步", "优先级", "可否当前主文"]];
header(gaps.getRange("A4:G4"));
const gapRows = [
  ["Final C5 deployment bundle", "待构建", "—", "决定可部署资产总大小与完整性", "冻结最终策略后打包并生成 SHA-256", "P1", "否"],
  ["当前分类 checkpoint", "已有诊断", "184,201 bytes; 36,173 params", "仅说明当前分类器量级", "不得替代完整 bundle size", "辅助", "仅补充材料/诊断"],
  ["1360-row offline/runtime parity", "待测", "—", "证明部署运行时没有语义漂移", "逐字段 class/expert/QC/ppm parity", "P1", "否"],
  ["Pi/PC inference latency", "待测", "—", "回答端侧实时性", "batch=1 主；30 warm-up；>=100 repeats；p50/p95/p99", "P1", "否"],
  ["Runtime RSS/CPU", "待测", "—", "训练 RSS 不能代表推理 RSS", "随 inference benchmark 同步采集", "P1", "否"],
  ["B5 real-topology 25 rounds", "运行中", "—", "提供第二个代表性真实系统点", "完成、回收、验证并与B2并列", "P0", "完成后可"],
  ["B2/B5 5-seed confirmation", "暂缓", "—", "支持算法稳定性 mean±sample std", "如投稿需要，按双逻辑客户端快速拓扑执行", "P0/后续", "当前不可"],
  ["Availability / long-run", "待评估", "—", "系统鲁棒性证据", "视主文篇幅与审稿风险决定1h/6h", "P2", "非当前必要"],
];
gaps.getRange("A5:G12").values = gapRows;
body(gaps.getRange("A5:G12"));
gaps.getRange("B5:B12").conditionalFormats.add("containsText", { text: "待", format: { fill: gray } });
gaps.getRange("B5:B12").conditionalFormats.add("containsText", { text: "运行中", format: { fill: amber } });
gaps.getRange("B5:B12").conditionalFormats.add("containsText", { text: "已有", format: { fill: green } });
gaps.freezePanes.freezeRows(4);
gaps.getRange("A1:G12").format.columnWidth = 22;
console.log("stage:gaps");

title(historical, "历史 CPU 部署与复杂度参考", "仅作量级参照：这些结果来自旧 C3/C4/C5、H2.3/H8/H8+C4 或旧神经回归协议，设备仅记为 CPU，不能作为当前 C5 正式 Pi/PC benchmark。", "H");
historical.getRange("A4:H4").values = [["类型", "对象", "整包/Checkpoint", "模型/FP32参数", "平均/Batch=1延迟", "P90延迟", "证据状态", "使用方式"]];
header(historical.getRange("A4:H4"));
historical.getRange("A5:H10").values = [
  ["旧runtime", "H2.3", "24.0419 MB", "21.238 MB model file", "2.734 ms/window", "3.287 ms", "报告已核对", "仅作CPU量级参照"],
  ["旧runtime", "H8", "24.6380 MB", "21.238 MB model file", "2.632 ms/window", "2.850 ms", "报告已核对", "仅作CPU量级参照"],
  ["旧runtime", "H8+C4", "24.6414 MB", "21.238 MB model file", "2.586 ms/window", "2.722 ms", "报告已核对", "legacy C4，不可回主线"],
  ["旧复杂度", "分类器 A", "0.1041 MiB checkpoint", "22,765 params / 0.0868 MiB", "1.381 ms", "—", "历史组件审计", "与当前36,173参数checkpoint分开"],
  ["旧复杂度", "源域神经回归 B", "1.6827 MiB checkpoint", "425,974 params / 1.625 MiB", "3.214 ms", "—", "用户提供；当前分支原报告缺失", "不可作为正式C5 runtime"],
  ["旧复杂度", "目标域神经回归 B", "1.7991 MiB checkpoint", "425,974 params / 1.625 MiB", "3.110 ms", "—", "用户提供；当前分支原报告缺失", "不可作为正式C5 runtime"],
];
body(historical.getRange("A5:H10"));
historical.getRange("A12:H13").merge(true);
historical.getRange("A12").values = [["可引用口径：历史 CPU benchmark 说明端到端部署具有毫秒级推理的初步可行性，但硬件、线程和当前协议不一致，不能替代正式 Pi/PC 测量。"]];
historical.getRange("A13").values = [["轻量化线索：历史分类器 22,765 个序列化参数中 3,208 个不参与前向（约14.1%）；只能作为未来等价剪枝方向，当前冻结训练不得修改。"]];
historical.getRange("A12:H13").format = { fill: amber, font: { bold: true }, wrapText: true, borders: border, rowHeight: 40 };
historical.freezePanes.freezeRows(4);
historical.getRange("A1:H13").format.columnWidth = 21;
console.log("stage:historical");

title(evidence, "证据来源与审计边界", "所有数值均对应可追溯文件；大型逐轮原始证据保留本地/ECS，不在此工作簿中复制。", "E");
evidence.getRange("A4:E4").values = [["证据类别", "状态", "文件/目录", "支持内容", "边界"]];
header(evidence.getRange("A4:E4"));
const evidenceRows = [
  ["B2系统诊断JSON", "recovered diagnostic", "results/iotj_ecs_c2_representative_20260720/a006_manual_recovery_system_metrics_20260721.json", "25轮通信/时延/资源/observer", "attempt immutable failed；不得称 canonical"],
  ["B2系统诊断说明", "recovered diagnostic", "results/iotj_ecs_c2_representative_20260720/a006_manual_recovery_system_metrics_20260721.md", "哈希、验证与解释", "同上"],
  ["分类 v3", "historical screening", "results/iotj_classification_ablation_20260712_v3_summary/classification_per_run.csv", "B2/B5 seed-42 分类", "不进入未来 multi-seed mean/std"],
  ["正式回归/QC", "formal", "results/iotj_c5_formal_regression_20260713_v2_summary/formal_regression_report.md", "R4/H8 FULL/HC95/HC90", "仅支持回归/QC链路"],
  ["历史CPU runtime", "legacy reference", "results/runtime_profile_benchmark_20260626/runtime_profile_benchmark_report.md", "旧H2.3/H8/H8+C4包大小与CPU延迟", "非当前C5、非正式Pi"],
  ["B5真实系统", "running", "results/iotj_ecs_c2_representative_20260720/raw/c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001", "待完成通信/时延/资源", "完成并验证前不得填数"],
  ["实验笔记", "living record", "docs/experiments/iotj_system_experiment_notebook.md", "命令、失败、决策、下一步", "以最新追加记录为准"],
];
evidence.getRange("A5:E11").values = evidenceRows;
body(evidence.getRange("A5:E11"));
evidence.freezePanes.freezeRows(4);
evidence.getRange("A1:E11").format.columnWidth = 24;
evidence.getRange("C5:C11").format.columnWidth = 62;
console.log("stage:evidence");

overview.getRange("A4:H21").format.rowHeight = 34;
system.getRange("A4:G21").format.rowHeight = 30;
algorithm.getRange("A4:J17").format.rowHeight = 28;
gaps.getRange("A4:G12").format.rowHeight = 34;
historical.getRange("A4:H13").format.rowHeight = 32;
evidence.getRange("A4:E11").format.rowHeight = 34;
console.log("stage:format");

const outputPath = `${outDir}/iotj_advisor_system_algorithm_metrics_20260721_v2.xlsx`;
const xlsx = await SpreadsheetFile.exportXlsx(wb);
console.log("stage:exported");
await xlsx.save(outputPath);
console.log("stage:saved");

const check = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const summary = await check.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
const errors = await check.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 6000,
});
await fs.writeFile(`${outDir}/workbook_inspection.txt`, `${summary.ndjson}\nFORMULA_ERRORS\n${errors.ndjson}\n`, "utf8");
console.log(outputPath);
console.log(summary.ndjson);
console.log(errors.ndjson);
