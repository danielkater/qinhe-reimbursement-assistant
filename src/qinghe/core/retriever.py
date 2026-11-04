"""本地关键词检索。

规则是"宁可转人工，不可乱答"：问题里只要出现制度没覆盖的词，就视为找不到依据。
例如"酒店的按摩费能报吗"虽然命中"酒店"，但"按摩费"无依据，必须转人工。
"""
import re

from .policy import Clause

# 不表达具体报销事项的词。除此之外的词都必须能在条款中找到。
STOP_PHRASES = sorted(
    """
    请问 我想问 想问 想知道 问一下 了解一下 帮我 告诉我 我们 公司 青禾 制度 规定 要求 报销 能报 可以报 能不能 可不可以
    费用 出差 差旅 多少钱 多少 怎么办 怎么算 怎么 如何 什么时候 什么 哪些 哪个 是否 是不是 需要 必须 应该 可以 能够
    上限 标准 最多 最高 不超过 一下 上周 昨天 今天 明天 下周 上个月 这个月 这次 那个 这个 时候 如果 的话 没有 还能
    还是 或者 以及 而且 但是 一共 总共 自己 员工 我的 花了 超了 部分 给我 需不需要 住 坐 去 到 从 在 给 和 与 及 或
    还 也 都 就 我 你 他 她 它 时 的 了 吗 呢 吧 啊 呀 是 有 要 能 可 会 对 为 被 把 几 钱 元 块 天 晚 次 张 个 人 每 该 算
    报 不 么 后 前 内 多 少 谁 哪 这 那 等 下 情况 时间 多久 多长
    """.split(),
    key=len,
    reverse=True,
)
NUMBER = re.compile(r"[0-9.]+|[一二三四五六七八九十百千万两半]+")
PUNCTUATION = re.compile(r"[\s,.!?;:，。！？；：、（）()\"'“”‘’《》【】-]+")


def normalize(text: str) -> str:
    return PUNCTUATION.sub(" ", text.lower()).strip()


def _extract_terms(question: str, keywords: set[str]) -> tuple[set[str], str]:
    """最长优先匹配关键词，返回命中的词和遮盖后的剩余文本。"""
    matched, rest = set(), question
    for keyword in sorted(keywords, key=len, reverse=True):
        if keyword in rest:
            matched.add(keyword)
            rest = rest.replace(keyword, " ")
    return matched, rest


def _residual_terms(rest: str) -> list[str]:
    for phrase in STOP_PHRASES:
        rest = rest.replace(phrase, " ")
    rest = NUMBER.sub(" ", rest)
    return rest.split()


def retrieve(
    question: str, clauses: tuple[Clause, ...], top_k: int = 2, keep_ratio: float = 0.7
) -> list[Clause]:
    """返回依据条款；没有足够依据时返回空列表。"""
    query = normalize(question)
    all_keywords = {k for c in clauses for k in c.keywords}
    matched, rest = _extract_terms(query, all_keywords)
    if not matched:
        return []

    # 长关键词更具体，权重按词长计算
    scored = sorted(
        ((sum(len(k) for k in c.keywords if k in matched), i, c) for i, c in enumerate(clauses)),
        key=lambda item: (-item[0], item[1]),
    )
    best_score = scored[0][0]
    selected = [c for score, _, c in scored[:top_k] if score >= max(1, best_score * keep_ratio)]

    evidence = "".join(c.text + c.title for c in selected).lower()
    if any(term not in evidence for term in _residual_terms(rest)):
        return []
    return selected
