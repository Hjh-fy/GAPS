# CLASS逐轮图交付核验

采用既有Python/matplotlib流程。单panel quantitative validation：说明CLASS在三个目标的固定25轮轨迹，而不是展示相对TO/FT优势。来源`class_round_curves/round_accuracy.csv`，75原始点无筛选、平滑或插值；C3/C4/C5测试窗口2677/1360/1360，seed42，round25复用固定预测。轮次不作为独立重复；不显示SD/CI/p值。

已视检实际PNG：标签、图例完整，无重叠或裁切；0–101%纵轴诚实展示全过程，但99%附近差异较难读，精确端点与首次/持续99%轮次应配合CSV/表格，不从曲线像素估计精细差值。不同marker辅助颜色区分，灰度下仍有形状编码。

源码检查：13 PASS / 1 WARN / 0 FAIL。WARN仅为未输出TIFF；本任务已冻结交付SVG/PDF和600dpi PNG，接受该格式提示，不宣称已核对任何特定期刊最新TIFF要求。88.9mm×67.31mm，Arial/sans，8pt；SVG保留text，PDF配置type42字体。无本轮图片裁剪、色阶调整或AI生成。

交付：CLASS_accuracy_vs_round.svg/.pdf/.png；源脚本scripts/evaluate_fedongas_extension.py。图为CLASS，不是旧A4曲线。完整CAPTION.md原文保持，中文解释及样本量以本文件补充。
