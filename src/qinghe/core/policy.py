"""加载制度文件，只返回状态为"有效"且可见范围为"公开"的唯一版本。"""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Clause:
    id: str
    title: str
    text: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    name: str
    version: str
    effective_date: str
    clauses: tuple[Clause, ...]


def load_active_policy(data_dir: Path) -> Policy:
    active = []
    for path in sorted(Path(data_dir).glob("policy*.yaml")):
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        # 可见范围在检索之前过滤：内部资料不进检索，也就到不了回答和模型上下文。
        # 没有标注可见范围的文件同样排除
        if doc["status"] == "有效" and doc.get("visibility") == "公开":
            active.append(doc)
    # 同时存在多个有效版本正是"口径不一致"，宁可启动失败也不要随便选一个
    if len(active) != 1:
        raise ValueError(f"有效公开制度版本应恰好 1 份，实际 {len(active)} 份")
    doc = active[0]
    clauses = tuple(
        Clause(c["id"], c["title"], c["text"], tuple(k.lower() for k in c["keywords"]))
        for c in doc["clauses"]
    )
    return Policy(doc["name"], doc["version"], str(doc["effective_date"]), clauses)
