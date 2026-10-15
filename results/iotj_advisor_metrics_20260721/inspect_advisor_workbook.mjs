import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const dir = "results/iotj_advisor_metrics_20260721";
const path = `${dir}/iotj_advisor_system_algorithm_metrics_20260721_v2.xlsx`;
const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const checks = [];
for (const [sheetId, range] of [
  ["导师汇报总览", "A1:H21"],
  ["系统指标_B2", "A1:J21"],
  ["算法性能", "A1:J17"],
  ["模型与部署缺口", "A1:G12"],
  ["历史部署参考", "A1:H13"],
  ["证据来源", "A1:E11"],
]) {
  checks.push((await wb.inspect({ kind: "region", sheetId, range, maxChars: 12000 })).ndjson);
}
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 6000,
});
checks.push(`FORMULA_ERRORS\n${errors.ndjson}`);
await fs.writeFile(`${dir}/workbook_key_range_inspection.txt`, checks.join("\n---\n"), "utf8");
console.log(errors.ndjson);
