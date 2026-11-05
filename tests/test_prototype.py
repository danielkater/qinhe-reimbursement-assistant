"""提示词 03 原型的测试。用假的嵌入和大模型，不联网、不下载模型。"""
import numpy as np
import pytest

from conftest import DATA_DIR, HANDOFF_MESSAGE
from qinghe.agent.router import is_claim_request
from qinghe.core.chunker import chunk_policy
from qinghe.core.embeddings import EmbeddingSettings, EmbeddingUnavailable, build_embedder
from qinghe.core.hybrid import HybridRetriever
from qinghe.core.policy import Clause, Policy, load_policy_markdown
from qinghe.core.qa import ANSWERED, HANDOFF, check_llm_answer
from qinghe.core.rag import BARE_SYSTEM_PROMPT, GuardedQA, answer_bare

POLICY_MD = DATA_DIR / "policies" / "policy_v3.md"
QUESTION = "上海住酒店一晚最多报多少？"


def fake_embedder(texts):
    """按字符统计的假向量：字面越接近，相似度越高。"""
    vectors = np.zeros((len(texts), 512), dtype="float32")
    for row, text in enumerate(texts):
        for char in text:
            vectors[row, ord(char) % 512] += 1
    return vectors


class RecordingLLM:
    """记录收到的提示词并返回固定回答；reply 是异常时就抛出它。"""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(system + user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def md_policy():
    return load_policy_markdown(POLICY_MD)


@pytest.fixture
def retriever(md_policy):
    return HybridRetriever(chunk_policy(md_policy), fake_embedder)


def guarded(policy, retriever, llm, min_similarity=0.0):
    return GuardedQA(policy, retriever, HANDOFF_MESSAGE, llm, min_similarity)


def test_markdown_policy_matches_yaml_and_rejects_non_public(policy, md_policy, tmp_path):
    assert md_policy == policy  # 条款编号、标题、正文、关键词和版本都一致

    internal = tmp_path / "internal.md"
    internal.write_text(
        "---\nname: 预算\nversion: V1\nstatus: 有效\nvisibility: 内部\neffective_date: 2026-01-01\n---\n"
        "\n## 第1条 预算\n\n内部内容\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_policy_markdown(internal)


def test_chunks_follow_clauses_and_respect_limit(md_policy):
    chunks = chunk_policy(md_policy, 500)
    assert [c.id for c in chunks] == [c.id for c in md_policy.clauses]
    assert all(len(c.text) <= 500 for c in chunks)

    # 超长条款在句末切开，切开后仍然属于原条款
    long_clause = Clause("第1条", "超长", "这是一句话。" * 200, ())
    long_chunks = chunk_policy(Policy("制度", "V1", "2026-01-01", (long_clause,)), 500)
    assert len(long_chunks) > 1
    assert all(len(c.text) <= 500 and c.clause is long_clause for c in long_chunks)
    assert all(c.text.endswith("。") for c in long_chunks)


def test_hybrid_returns_top3_and_works_without_vectors(md_policy, retriever):
    hits = retriever.search(QUESTION)
    assert len(hits) == 3
    assert all(hit.vector_score is not None for hit in hits)

    bm25_only = HybridRetriever(chunk_policy(md_policy))
    assert not bm25_only.uses_vectors
    assert all(hit.vector_score is None for hit in bm25_only.search("发票抬头"))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("BX-2026-0101 到哪一步了", True),
        ("R-0001 现在什么状态", True),
        ("帮我查一下小王的报销单", True),
        ("我的单据到哪了", True),
        ("报销进度怎么看", True),
        ("上海住酒店一晚最多报多少？", False),
        ("单据被财务退回怎么办", False),
        ("电子发票可以交截图吗", False),
        ("差旅报销要交什么材料", False),
    ],
)
def test_router_keeps_claim_requests_local(text, expected):
    assert is_claim_request(text) is expected


def test_guarded_mode_rules(policy, md_policy, retriever):
    # 第一关：低于门槛时不调用模型
    llm = RecordingLLM("不应被调用")
    result = guarded(md_policy, retriever, llm, min_similarity=2.0).ask(QUESTION)
    assert result.answer.status == HANDOFF and llm.prompts == []

    # 第二关：模型判定条款不足
    llm = RecordingLLM(HANDOFF_MESSAGE)
    assert guarded(md_policy, retriever, llm).ask(QUESTION).answer.status == HANDOFF

    # 第三关：合格的回答直接展示
    good = RecordingLLM("一线城市每晚不超过600元【第2条】。")
    result = guarded(md_policy, retriever, good).ask(QUESTION)
    assert result.answer.status == ANSWERED and result.answer.mode == "llm"
    assert [c.id for c in result.answer.citations] == ["第2条"]

    # 编造金额、编造条款、调用失败，都不展示模型文字，退回关键词摘录原文
    for reply in ("上海每晚可报800元【第2条】。", "每晚600元【第99条】。", TimeoutError("断网")):
        result = guarded(md_policy, retriever, RecordingLLM(reply)).ask(QUESTION)
        assert result.answer.mode == "local"
        assert "800" not in result.answer.text and "第99条" not in result.answer.text

    # 回答用列表时，序号不算数字；但列表里的编造金额仍要拦下
    clause = md_policy.clauses[12]
    assert check_llm_answer("1. 取得存根复印件；\n2. 超过1000元另行确认【第13条】", [clause])
    assert check_llm_answer("1. 超过800元另行确认【第13条】", [clause]) is None

    # 没有大模型时，无依据的问题仍然转人工
    no_llm = guarded(md_policy, retriever, None)
    assert no_llm.ask("健身卡可以报销吗？").answer.status == HANDOFF
    assert no_llm.ask(QUESTION).answer.mode == "local"

    # 严格模式：有制度外词语时不调用模型
    llm = RecordingLLM("不应被调用")
    strict = GuardedQA(md_policy, retriever, HANDOFF_MESSAGE, llm, 0.0, strict_unknown_terms=True)
    assert strict.ask("酒店的按摩费能报吗？").answer.status == HANDOFF and llm.prompts == []


def test_bare_mode_returns_model_text_unchecked(retriever):
    llm = RecordingLLM("编造的内容：每晚 9999 元")
    hits = retriever.search(QUESTION)
    assert answer_bare(QUESTION, hits, llm) == "编造的内容：每晚 9999 元"
    assert BARE_SYSTEM_PROMPT in llm.prompts[0]
    assert "【内部】" not in llm.prompts[0]  # 内部资料不在检索范围内


def test_embedder_config_errors_are_reported(tmp_path):
    settings = EmbeddingSettings("unknown", "m", tmp_path)
    with pytest.raises(EmbeddingUnavailable):
        build_embedder(settings)
    assert build_embedder(EmbeddingSettings("none", "m", tmp_path)) is None
    with pytest.raises(EmbeddingUnavailable):
        build_embedder(EmbeddingSettings("openai_compatible", "m", tmp_path))
