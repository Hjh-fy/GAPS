from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


COMMANDS = {
    "/task-code": "code",
    "/task-review": "review",
    "/task-exp": "experiment",
    "/task-doc": "document",
    "/task-issue": "issue",
}

MODULE_LABELS = {
    "qc": "gaps-qc",
    "qc v3": "gaps-qc",
    "回归": "gaps-regression",
    "regression": "gaps-regression",
    "flower": "gaps-flower",
    "部署": "gaps-deploy",
    "deploy": "gaps-deploy",
    "文档": "gaps-doc",
    "doc": "gaps-doc",
    "实验": "experiment-replay",
    "replay": "experiment-replay",
}


@dataclass
class ParsedTask:
    command: str = "issue"
    title: str = "飞书任务"
    module: str = ""
    background: str = ""
    files: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    acceptance: str = ""
    raw_text: str = ""
    labels: List[str] = field(default_factory=list)

    def to_issue_body(self, *, sender: str = "", codex_trigger_text: str = "", auto_mention_codex: bool = False) -> str:
        parts = []
        if auto_mention_codex and codex_trigger_text:
            parts.append(codex_trigger_text.strip())
            parts.append("")

        parts.append("## 飞书任务")
        parts.append("")
        parts.append(f"- 类型: `{self.command}`")
        if self.module:
            parts.append(f"- 模块: `{self.module}`")
        if sender:
            parts.append(f"- 来源: 飞书用户 `{sender}`")

        if self.background:
            parts.extend(["", "## 背景", "", self.background.strip()])

        if self.files:
            parts.extend(["", "## 涉及文件", ""])
            parts.extend([f"- `{x}`" for x in self.files])

        if self.requirements:
            parts.extend(["", "## 要求", ""])
            parts.extend([f"- {x}" for x in self.requirements])

        if self.acceptance:
            parts.extend(["", "## 验收标准", "", self.acceptance.strip()])

        parts.extend(["", "## 原始飞书消息", "", "```text", self.raw_text.strip(), "```"])
        return "\n".join(parts).strip() + "\n"


def strip_feishu_markup(text: str) -> str:
    # Remove Feishu at tags and normalize newlines.
    text = re.sub(r"<at[^>]*>.*?</at>", "", text, flags=re.I | re.S)
    text = text.replace("\\r\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def _extract_section(lines: List[str], header_names: List[str]) -> Dict[str, str]:
    headers = {}
    header_pattern = "|".join(re.escape(x) for x in header_names)
    current = None
    buf: Dict[str, List[str]] = {}
    for line in lines:
        clean = line.strip()
        m = re.match(rf"^({header_pattern})\s*[:：]\s*(.*)$", clean)
        if m:
            current = m.group(1)
            buf.setdefault(current, [])
            if m.group(2).strip():
                buf[current].append(m.group(2).strip())
            continue
        if current:
            buf[current].append(line.rstrip())
    for key, value in buf.items():
        headers[key] = "\n".join(value).strip()
    return headers


def _parse_list_block(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        item = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        item = re.sub(r"^\s*\d+[.)、]\s*", "", item).strip()
        if item:
            out.append(item)
    return out


def infer_labels(module: str, text: str, default_labels: List[str]) -> List[str]:
    labels = list(default_labels)
    haystack = f"{module}\n{text}".lower()
    for key, label in MODULE_LABELS.items():
        if key.lower() in haystack and label not in labels:
            labels.append(label)
    return labels


def parse_task(text: str, default_labels: List[str]) -> ParsedTask:
    raw = strip_feishu_markup(text)
    lines = [x.rstrip() for x in raw.splitlines() if x.strip()]

    command = "issue"
    for token, name in COMMANDS.items():
        if token in raw:
            command = name
            break

    headers = _extract_section(
        lines,
        ["模块", "标题", "背景", "涉及文件", "文件", "要求", "验收", "验收标准", "任务"],
    )

    title = headers.get("标题") or headers.get("任务") or ""
    module = headers.get("模块", "").splitlines()[0].strip() if headers.get("模块") else ""

    if not title:
        for line in lines:
            clean = line.strip()
            if clean.startswith("@") or clean.startswith("/"):
                continue
            if clean.startswith(tuple(COMMANDS.keys())):
                continue
            if "：" in clean and len(clean.split("：", 1)[0]) <= 6:
                continue
            title = clean[:80]
            break
    if not title:
        title = "飞书任务"

    files = _parse_list_block(headers.get("涉及文件") or headers.get("文件") or "")
    requirements = _parse_list_block(headers.get("要求", ""))
    acceptance = headers.get("验收标准") or headers.get("验收") or ""
    background = headers.get("背景", "")

    labels = infer_labels(module, raw, default_labels)

    return ParsedTask(
        command=command,
        title=title.strip(),
        module=module,
        background=background,
        files=files,
        requirements=requirements,
        acceptance=acceptance,
        raw_text=raw,
        labels=labels,
    )
