"""审计日志：每行一条 JSON，只追加。"""
import json
from datetime import datetime
from pathlib import Path

# 字段白名单：只留追踪事件所需的内容，姓名、提问原文和单据内容写不进日志
ALLOWED_FIELDS = {
    "action",
    "actor_id",
    "claim_id",
    "result",
    "reason",
    "elapsed_ms",
    "mode",
    "citations",
}


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, **event) -> None:
        unexpected = set(event) - ALLOWED_FIELDS
        if unexpected:
            raise ValueError(f"审计日志不允许记录这些字段：{sorted(unexpected)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {"time": datetime.now().isoformat(timespec="seconds"), **event}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self) -> dict:
        events = self.read_all()
        lookups = [e for e in events if e.get("action") == "claim_lookup"]
        questions = [e for e in events if e.get("action") == "policy_question"]
        return {
            "lookups": len(lookups),
            "denied": sum(1 for e in lookups if e["result"] == "deny"),
            "avg_lookup_ms": (
                round(sum(e["elapsed_ms"] for e in lookups) / len(lookups), 2) if lookups else 0
            ),
            "questions": len(questions),
            "handoffs": sum(1 for e in questions if e["result"] == "handoff"),
        }
