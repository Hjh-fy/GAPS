from app.task_parser import parse_task


def test_parse_gaps_task():
    text = '''
@GAPS-Codex /task-code

模块：QC v3
标题：修复 rank/margin risk 占位问题

背景：
当前 class_response_rank_risk 和 class_response_margin_risk 仍然是 0。

涉及文件：
- evaluate_single_window_reliability.py
- route_aware_response_anchoring.py

要求：
1. 保持 CSV 字段兼容
2. 不改变训练主流程

验收：
生成新的 guardrail_summary.csv。
'''
    task = parse_task(text, default_labels=["codex-task", "needs-review"])
    assert task.command == "code"
    assert task.title == "修复 rank/margin risk 占位问题"
    assert task.module == "QC v3"
    assert "evaluate_single_window_reliability.py" in task.files
    assert "gaps-qc" in task.labels
    body = task.to_issue_body(sender="u1")
    assert "## 涉及文件" in body
    assert "guardrail_summary.csv" in body
