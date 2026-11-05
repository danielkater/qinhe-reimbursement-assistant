"""加载制度文件，只返回状态为"有效"且可见范围为"公开"的唯一版本。"""
import re
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


# Markdown 制度的格式：开头是 YAML 元数据，每条条款是一个“## 第N条 标题”小节
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
CLAUSE_HEADING = re.compile(r"^## (第\d+条) (.+)$", re.M)
KEYWORDS_COMMENT = re.compile(r"<!--\s*keywords:\s*(.*?)\s*-->")


def load_policy_markdown(path: Path) -> Policy:
    """读取一份 Markdown 制度。

    【后续复用：核心模块】阶段 3 的制度后台上传的也是这种格式。

    和 load_active_policy 一样默认拒绝：不是“有效”且“公开”的文件一律不读取，
    所以内部资料（如 DOC-FIN）即使被配置到这里，也进不了检索和模型。
    """
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    match = FRONT_MATTER.match(text)
    if match is None:
        raise ValueError(f"{path} 缺少开头的元数据")
    meta = yaml.safe_load(match.group(1))
    if meta.get("status") != "有效" or meta.get("visibility") != "公开":
        raise ValueError(f"{path} 不是有效且公开的制度，不能进入检索")

    body = text[match.end():]
    headings = list(CLAUSE_HEADING.finditer(body))
    clauses = []
    for current, following in zip(headings, headings[1:] + [None]):
        end = following.start() if following else len(body)
        section = body[current.end():end]
        # keywords 写在注释里，不属于条款正文；它只给关键词降级检索使用
        keywords = KEYWORDS_COMMENT.search(section)
        words = keywords.group(1).split(", ") if keywords else []
        clause_text = KEYWORDS_COMMENT.sub("", section).strip()
        clauses.append(
            Clause(current.group(1), current.group(2).strip(), clause_text, tuple(k.lower() for k in words))
        )
    if not clauses:
        raise ValueError(f"{path} 中没有找到条款")
    return Policy(meta["name"], meta["version"], str(meta["effective_date"]), tuple(clauses))


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
