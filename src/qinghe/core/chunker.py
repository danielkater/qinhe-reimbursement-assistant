"""制度切片。

【后续复用：核心模块】阶段 3 的制度后台发布新版本时，也用这里的规则切片。

为什么和老师提示词的“按 500 字切片”不同（决策 X3）：
现有 15 条条款大多不到 100 字。如果整篇文档每 500 字切一段，一段会混进好几条条款，
也可能把一条条款切成两半。这样回答就很难准确引用到“第几条”，引用校验也无法判断依据。
所以这里以“条款”为单位切片，并保留老师要求的 500 字作为单片上限：
只有一条条款超过 500 字时，才在句号处继续切开，切开后仍然记录属于哪一条。
"""
import re
from dataclasses import dataclass

from .policy import Clause, Policy

DEFAULT_MAX_CHARS = 500
# 按中文句末标点切句，并把标点留在前一句的末尾
SENTENCE_END = re.compile(r"(?<=[。！？；])")


@dataclass(frozen=True)
class Chunk:
    """检索的最小单位。一条条款通常对应一个切片。"""

    id: str  # 切片编号，例如“第2条”；长条款切开后是“第2条-2”
    clause: Clause  # 所属条款，引用校验和展示原文都要用它
    text: str  # 这个切片的文字


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """把超长文字切成不超过 max_chars 的若干段，优先在句末切开。"""
    parts: list[str] = []
    current = ""
    for sentence in filter(None, SENTENCE_END.split(text)):
        # 单句本身就超长时，没有句号可用，只能按长度硬切
        while len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        # 放不下就先收起当前段，再开始新的一段
        if len(current) + len(sentence) > max_chars:
            parts.append(current)
            current = ""
        current += sentence
    if current:
        parts.append(current)
    return parts


def chunk_policy(policy: Policy, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """把制度切成检索用的切片。切片的文字带上标题，这样按标题提问也能命中。"""
    chunks: list[Chunk] = []
    for clause in policy.clauses:
        # 标题和正文一起计入长度，保证切片不超过上限
        heading = f"{clause.id} {clause.title}："
        parts = _split_long_text(clause.text, max_chars - len(heading))
        for index, part in enumerate(parts, start=1):
            chunk_id = clause.id if len(parts) == 1 else f"{clause.id}-{index}"
            chunks.append(Chunk(chunk_id, clause, heading + part))
    return chunks
