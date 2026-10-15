import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outDir = "results/iotj_advisor_metrics_20260721";
const b2 = JSON.parse(await fs.readFile("results/iotj_ecs_c2_representative_20260720/a006_manual_recovery_system_metrics_20260721.json", "utf8"));
const b5 = JSON.parse(await fs.readFile("results/iotj_ecs_c2_b5_canonical_analysis_20260721/b5_canonical_system_metrics.json", "utf8"));
const wb = Workbook.create();
const overview = wb.worksheets.add("导师汇报总览");
const system = wb.worksheets.add("系统指标_B2_B5");
const algorithm = wb.worksheets.add("算法性能与边界");
const deployment = wb.worksheets.add("模型与部署证据");
const evidence = wb.worksheets.add("证据来源");

const navy = "#17365D", blue = "#2F75B5", pale = "#D9EAF7", green = "#E2F0D9", amber = "#FFF2CC", gray = "#E7E6E6", white = "#FFFFFF";
const border = { preset: "all", style: "thin", color: "#D9E2F3" };
const systemStatus = "B2 canonicalized recovery (authorized)；B5 canonical representative";

function title(sheet, main, sub, last) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${last}1`).merge(); sheet.getRange("A1").values = [[main]];
  sheet.getRange(`A1:${last}1`).format = { fill: navy, font: { bold: true, color: white, size: 18 }, rowHeight: 30, verticalAlignment: "center" };
  sheet.getRange(`A2:${last}2`).merge(); sheet.getRange("A2").values = [[sub]];
  sheet.getRange(`A2:${last}2`).format = { fill: pale, font: { italic: true, size: 10 }, wrapText: true, rowHeight: 40, verticalAlignment: "center" };
}
function header(range) { range.format = { fill: blue, font: { bold: true, color: white }, borders: border, wrapText: true, verticalAlignment: "center" }; }
function body(range) { range.format = { borders: border, wrapText: true, verticalAlignment: "top" }; }
function percent(range) { range.format.numberFormat = "0.00%"; }

title(overview, "GAPS IoT-J：导师汇报指标总览（B2/B5 代表性系统运行已完成）", "B5-s42/a001 已完成 25/25 rounds 并获 validator 接受；B2-s42/a006 的原始 controller 状态保留为 failed，但经完整恢复验证和用户授权，已受控提升为代表性系统证据。两者均不构成正式算法多种子统计。", "H");
overview.getRange("A4:H4").values = [["类别", "指标", "B2", "B5", "B5−B2", "单位", "证据等级", "结论边界"]]; header(overview.getRange("A4:H4"));
const topRows = [
  ["真实系统", "25轮 application communication", b2.communication.application_25round_total_mib, b5.communication.application_25round_total_mib, null, "MiB", systemStatus, "serialized application layer；transport 未采集"],
  ["真实系统", "Mean round wall", b2.timing_seconds.round_wall_mean, b5.timing_seconds.round_wall_mean, null, "s", systemStatus, "B5 canonical；B2 recovery-authorized"],
  ["真实系统", "Server DA share", b2.timing_seconds.server_da_share_of_round_wall, b5.timing_seconds.server_da_share_of_round_wall, null, "%", systemStatus, "两次运行均显示 DA 是主要时间构成"],
  ["边缘资源", "Pi peak RSS", b2.resources.pi_c1.rss_peak_mib, b5.resources.pi_c1.rss_peak_mib, null, "MiB", systemStatus, "训练重叠期；非最终推理 RSS"],
  ["边缘资源", "Pi peak temperature", b2.resources.pi_c1.temperature_peak_c, b5.resources.pi_c1.temperature_peak_c, null, "°C", systemStatus, "两次均无热节流"],
  ["可观测性", "Observer overhead ratio", b2.observer.overhead_to_round_wall_ratio, b5.observer.overhead_to_round_wall_ratio, null, "%", systemStatus, "两次均约 0.10%，未从原始时延扣除"],
  ["分类", "Accuracy（历史 seed-42）", 0.9926470588, 0.9889705882, null, "%", "historical screening", "不进入未来五种子 mean/std"],
  ["回归", "R4/H8 FULL RMSE", 14.6564, 17.4473, null, "ppm", "formal C5 regression", "正式回归链路，不是当前 B5 checkpoint 重评估"],
  ["回归/QC", "HC90 accepted RMSE", 11.5866, 15.3599, null, "ppm", "formal C5 regression/QC", "同时应报告 yield"],
  ["回归/QC", "HC90 QC Yield", 0.8949, 0.8824, null, "%", "formal C5 regression/QC", "被拒样本不计 accepted RMSE"],
];
overview.getRange("A5:H14").values = topRows; body(overview.getRange("A5:H14"));
for (let r = 5; r <= 14; r++) overview.getRange(`E${r}`).formulas = [[`=D${r}-C${r}`]];
overview.getRange("C7:E7").format.numberFormat = "0.00%"; overview.getRange("C10:E10").format.numberFormat = "0.000%"; overview.getRange("C14:E14").format.numberFormat = "0.00%";
overview.getRange("A16:H17").merge(true); overview.getRange("A16").values = [["系统初步结论：两个独立的 25 轮真实拓扑运行均稳定在约 16.76 MiB application communication、约 4 分钟/轮、DA 约占 68% round wall；因此当前优化优先点是 ECS server-side DA，而非 Pi 本地训练或 Observer。"]]; overview.getRange("A17").values = [["投稿边界：B5 是单次 canonical representative system run；B2 是 user-authorized canonicalized recovery（原始 controller 状态仍为 failed）。它们证明系统可运行与成本量级，不构成 B2/B5 的算法多种子稳定性或显著性比较。"]];
overview.getRange("A16:H17").format = { fill: amber, font: { bold: true }, borders: border, wrapText: true, rowHeight: 38 };
overview.freezePanes.freezeRows(4); overview.getRange("A1:H17").format.columnWidth = 18; overview.getRange("G:H").format.columnWidth = 28;

title(system, "B2/B5 真实 ECS + Pi + ECS-C2 系统成本", "同一冻结算法 archive、C1/C2 逻辑客户端、25 rounds、5 local epochs、batch 32；B5 为 canonical，B2 为用户授权的 canonicalized recovery（原始状态不改写）。", "I");
system.getRange("A4:I4").values = [["分组", "指标", "B2", "B5", "B5−B2", "单位", "B5 证据状态", "B2 证据状态", "说明"]]; header(system.getRange("A4:I4"));
const sysRows = [
  ["通信", "Application downlink", b2.communication.application_downlink_25round_total_bytes, b5.communication.application_downlink_25round_total_bytes, null, "bytes", "canonical", "canonicalized recovery", "serialized application message"],
  ["通信", "Application uplink", b2.communication.application_uplink_25round_total_bytes, b5.communication.application_uplink_25round_total_bytes, null, "bytes", "canonical", "canonicalized recovery", "serialized application message"],
  ["通信", "Application total", b2.communication.application_25round_total_bytes, b5.communication.application_25round_total_bytes, null, "bytes", "canonical", "canonicalized recovery", "transport/wire bytes not collected"],
  ["通信", "Mean per round", b2.communication.application_round_mean_bytes, b5.communication.application_round_mean_bytes, null, "bytes/round", "canonical", "canonicalized recovery", "two logical clients"],
  ["时延", "Round wall mean", b2.timing_seconds.round_wall_mean, b5.timing_seconds.round_wall_mean, null, "s", "canonical", "canonicalized recovery", "p50/p95 in analysis JSON"],
  ["时延", "Round wall total", b2.timing_seconds.round_wall_total, b5.timing_seconds.round_wall_total, null, "s", "canonical", "canonicalized recovery", "25 completed rounds"],
  ["时延", "Server DA mean", b2.timing_seconds.server_da_mean, b5.timing_seconds.server_da_mean, null, "s/round", "canonical", "canonicalized recovery", "main time component"],
  ["时延", "Server DA share", b2.timing_seconds.server_da_share_of_round_wall, b5.timing_seconds.server_da_share_of_round_wall, null, "%", "canonical", "canonicalized recovery", "aggregate-time / round wall"],
  ["时延", "Pi C1 local train mean", b2.timing_seconds.pi_c1_train_mean, b5.timing_seconds.pi_c1_train_mean, null, "s/round", "canonical", "canonicalized recovery", "parallel with C2"],
  ["时延", "ECS-C2 local train mean", b2.timing_seconds.ecs_c2_train_mean, b5.timing_seconds.ecs_c2_train_mean, null, "s/round", "canonical", "canonicalized recovery", "parallel with C1"],
  ["Pi资源", "Active RSS mean", b2.resources.pi_c1.rss_active_mean_mib, b5.resources.pi_c1.rss_active_mean_mib, null, "MiB", "canonical", "canonicalized recovery", "training overlap"],
  ["Pi资源", "Peak RSS", b2.resources.pi_c1.rss_peak_mib, b5.resources.pi_c1.rss_peak_mib, null, "MiB", "canonical", "canonicalized recovery", "training overlap"],
  ["Pi资源", "Host CPU mean", b2.resources.pi_c1.cpu_host_mean_percent, b5.resources.pi_c1.cpu_host_mean_percent, null, "%", "canonical", "canonicalized recovery", "host-scale sampler"],
  ["Pi资源", "Temperature peak", b2.resources.pi_c1.temperature_peak_c, b5.resources.pi_c1.temperature_peak_c, null, "°C", "canonical", "canonicalized recovery", "throttling false/false"],
  ["ECS-C2资源", "Peak RSS", b2.resources.ecs_c2.rss_peak_mib, b5.resources.ecs_c2.rss_peak_mib, null, "MiB", "canonical", "canonicalized recovery", "training overlap"],
  ["Observer", "Total overhead", b2.observer.total_overhead_ms, b5.observer.total_overhead_ms, null, "ms", "canonical", "canonicalized recovery", "all producers; not subtracted"],
  ["Observer", "Overhead / wall", b2.observer.overhead_to_round_wall_ratio, b5.observer.overhead_to_round_wall_ratio, null, "%", "canonical", "canonicalized recovery", "about 0.10%"],
];
system.getRange("A5:I21").values = sysRows; body(system.getRange("A5:I21"));
for (let r = 5; r <= 21; r++) system.getRange(`E${r}`).formulas = [[`=D${r}-C${r}`]];
system.getRange("C12:E12").format.numberFormat = "0.00%"; system.getRange("C21:E21").format.numberFormat = "0.000%";
system.getRange("K4:L8").values = [["阶段", "B5 mean s/round"], ["Server DA", b5.timing_seconds.server_da_mean], ["ECS-C2 train", b5.timing_seconds.ecs_c2_train_mean], ["Pi C1 train", b5.timing_seconds.pi_c1_train_mean], ["Server non-DA", b5.timing_seconds.server_non_da_mean]]; header(system.getRange("K4:L4")); body(system.getRange("K5:L8"));
const chart = system.charts.add("bar", system.getRange("K4:L8")); chart.title = "B5 每轮主要阶段耗时"; chart.hasLegend = false; chart.setPosition("K10", "R24");
system.freezePanes.freezeRows(4); system.getRange("A1:I21").format.columnWidth = 17; system.getRange("G:I").format.columnWidth = 24;

title(algorithm, "算法性能与论文表述边界", "分类为 historical seed-42 screening；回归/QC 为已有正式 C5 结果。B5 canonical system attempt 证明真实拓扑可运行，不自动生成新的五种子分类结论。", "J");
algorithm.getRange("A4:J4").values = [["任务", "方案", "范围", "N", "Accuracy", "Macro-F1", "NLL", "ECE", "RMSE ppm", "NRMSE"]]; header(algorithm.getRange("A4:J4"));
algorithm.getRange("A5:J10").values = [
  ["分类", "B2", "historical C5 test", 1360, 0.9926470588, 0.9926573079, 0.0690175779, 0.0066814276, null, null],
  ["分类", "B5", "historical C5 test", 1360, 0.9889705882, 0.9889895838, 0.0703653693, 0.0092726690, null, null],
  ["回归", "B2 R4/H8", "FULL", 1360, null, null, null, null, 14.6564, 0.1059],
  ["回归", "B5 R4/H8", "FULL", 1360, null, null, null, null, 17.4473, 0.1352],
  ["回归/QC", "B2 R4/H8", "HC90 accepted", 1217, null, null, null, null, 11.5866, 0.0747],
  ["回归/QC", "B5 R4/H8", "HC90 accepted", 1200, null, null, null, null, 15.3599, 0.1151],
]; body(algorithm.getRange("A5:J10")); algorithm.getRange("E5:F6").format.numberFormat = "0.0000%"; algorithm.getRange("H5:H6").format.numberFormat = "0.0000%"; algorithm.getRange("I5:J10").format.numberFormat = "0.0000";
algorithm.getRange("A12:J14").merge(true); algorithm.getRange("A12").values = [["安全表述：GAPS combines real-device federated classification with centrally pooled multi-source regression references and target-personalized calibration/QC. 不得写成端到端全流程均为联邦。"]]; algorithm.getRange("A13").values = [["当前 B5 系统 run：canonical B5-s42 单次代表性真实拓扑证据；当前 B2 系统 run：用户授权的 canonicalized recovery（原始 controller failed 状态保留）。"]]; algorithm.getRange("A14").values = [["尚缺：B2/B5 × seeds 42–46 的完整 canonical matrix，才可报告正式 mean±sample std、paired difference 与统计检验。"]]; algorithm.getRange("A12:J14").format = { fill: amber, font: { bold: true }, borders: border, wrapText: true, rowHeight: 32 };
algorithm.freezePanes.freezeRows(4); algorithm.getRange("A1:J14").format.columnWidth = 16;

title(deployment, "模型大小、推理性能与部署缺口", "当前只报告已测得的训练 checkpoint 与历史 CPU reference；最终 C5 deployment bundle、1360-row parity、Pi/PC inference p50/p95/p99 仍须在部署策略冻结后测量。", "H");
deployment.getRange("A4:H4").values = [["项目", "当前数值", "单位", "证据等级", "可否作为正式部署结果", "如何使用", "下一步", "备注"]]; header(deployment.getRange("A4:H4"));
deployment.getRange("A5:H12").values = [
  ["B5 canonical classifier checkpoint", b5.checkpoint.size_bytes, "bytes", "canonical training asset", "否", "说明当前分类训练 checkpoint 量级", "final C5 bundle", "184,201 bytes；不是完整 runtime bundle"],
  ["B5 canonical classifier checkpoint", b5.checkpoint.size_bytes / (1024 * 1024), "MiB", "canonical training asset", "否", "约 0.176 MiB", "final C5 bundle", "sha256 已记录"],
  ["Historical H8 deployment bundle", 24.6380, "MB", "legacy CPU reference", "否", "仅作历史 CPU 可行性量级参照", "current bundle", "旧 C3/C4/C5 protocol"],
  ["Historical H8 CPU mean latency", 2.63191, "ms/window", "legacy CPU reference", "否", "说明旧协议可达到毫秒级 CPU 推理", "Pi/PC formal inference", "非正式 Pi 结果"],
  ["Final C5 deployment bundle size", null, "MiB", "unknown", "待测", "不得由旧包替代", "P1: bundle + SHA-256", "需 classifier/R4/QC/schema/assets"],
  ["1360-row offline/runtime parity", null, "rows", "unknown", "待测", "不得以训练 checkpoint 替代", "P1: exact parity", "class/expert/QC/ppm"],
  ["Pi/PC batch=1 latency", null, "ms", "unknown", "待测", "正式推理可用性", "P1: p50/p95/p99", "30 warm-up; >=100 measures"],
  ["Pi/PC runtime RSS/CPU", null, "MiB / %", "unknown", "待测", "正式部署资源", "P1: with inference", "训练 RSS 不等于推理 RSS"],
]; body(deployment.getRange("A5:H12")); deployment.getRange("B5:B12").format.numberFormat = "0.0000"; deployment.freezePanes.freezeRows(4); deployment.getRange("A1:H12").format.columnWidth = 22; deployment.getRange("E:H").format.columnWidth = 25;

title(evidence, "B2/B5 证据来源与审计边界", "B5 canonical 结果由 validator audit、三端 events/resource sidecar 和独立重新计算的系统摘要绑定；B2 的 scoped promotion 不覆盖或修改其原始 failed 证据。", "F");
evidence.getRange("A4:F4").values = [["证据", "状态", "路径", "支持内容", "hash/审计", "边界"]]; header(evidence.getRange("A4:F4"));
evidence.getRange("A5:F10").values = [
  ["B5 attempt status", "canonical", "results/iotj_ecs_c2_representative_20260720/raw/c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/attempt_status.json", "最终状态", b5.audit_sha256, "单次 B5 代表性系统 run"],
  ["B5 validator audit", "valid", ".../attempt_audit.json", "25 rounds; 50 FitIns; 50 FitRes; resource coverage >=95%", b5.audit_sha256, "audit-bound raw evidence"],
  ["B5 recomputed metrics", "recomputed", "results/iotj_ecs_c2_b5_canonical_analysis_20260721/b5_canonical_system_metrics.json", "communication/timing/resource/observer/checkpoint", b5.checkpoint.sha256, "system cost only"],
  ["B2 authorization record", "canonicalized recovery", "results/iotj_ecs_c2_representative_20260720/b2_a006_canonicalization_authorization_20260721.json", "user-authorized representative system evidence", b2.manual_validation.audit_sha256, "original controller failed status preserved; not multi-seed algorithm evidence"],
  ["Formal regression/QC", "formal", "results/iotj_c5_formal_regression_20260713_v2_summary/formal_regression_report.md", "R4/H8 FULL/HC95/HC90", "reported", "separate regression evidence"],
  ["Historical CPU runtime", "legacy", "results/runtime_profile_benchmark_20260626/runtime_profile_benchmark_report.md", "legacy CPU bundle/latency", "reported", "not final C5 Pi/PC"],
]; body(evidence.getRange("A5:F10")); evidence.freezePanes.freezeRows(4); evidence.getRange("A1:F10").format.columnWidth = 22; evidence.getRange("C5:C10").format.columnWidth = 58; evidence.getRange("E:F").format.columnWidth = 28;

for (const [sheet, range] of [[overview,"A4:H17"],[system,"A4:I21"],[algorithm,"A4:J14"],[deployment,"A4:H12"],[evidence,"A4:F10"]]) sheet.getRange(range).format.rowHeight = 30;

const outputPath = `${outDir}/iotj_advisor_system_algorithm_metrics_20260721_v4.xlsx`;
const xlsx = await SpreadsheetFile.exportXlsx(wb); await xlsx.save(outputPath);
const check = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const inspection = await check.inspect({ kind: "workbook,sheet,table", maxChars: 6000, tableMaxRows: 8, tableMaxCols: 10 });
const errors = await check.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 200 }, maxChars: 4000 });
await fs.writeFile(`${outDir}/v4_workbook_inspection.txt`, `${inspection.ndjson}\nFORMULA_ERRORS\n${errors.ndjson}\n`, "utf8");
try {
  const preview = await check.render({ sheetName: "导师汇报总览", range: "A1:H17", autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outDir}/v4_overview_preview.png`, new Uint8Array(await preview.arrayBuffer()));
  await fs.writeFile(`${outDir}/v4_render_status.txt`, "artifact-tool overview render: passed\n", "utf8");
} catch (error) {
  await fs.writeFile(`${outDir}/v4_render_status.txt`, `artifact-tool overview render: failed\n${String(error)}\n`, "utf8");
}
console.log(outputPath); console.log(errors.ndjson);
