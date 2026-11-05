"""检索 + 大模型问答的两种模式（决策 X4）。

- 关闭防护（answer_bare）：【原型专用】只用于观察模型会不会编，不要接到正式页面。
- 开启防护（GuardedQA）：【后续复用：核心模块】后面阶段的“查制度”工具沿用这套规则。
"""
from dataclasses import dataclass

from .hybrid import DEFAULT_TOP_K, Hit, HybridRetriever
from .llm import LLMCall
from .policy import Policy
from .qa import ANSWERED, HANDOFF, SYSTEM_PROMPT, Answer, PolicyQA, check_llm_answer
from .retriever import retrieve

# 关闭防护时的提示词：有意不要求拒答，也不要求引用，这样才能看到模型本来的行为
BARE_SYSTEM_PROMPT = "你是公司报销助手。请参考下面的制度内容，回答员工的问题。"


def build_context(hits: list[Hit]) -> str:
    """把检索到的切片拼成提示词里的参考内容。"""
    return "\n".join(hit.chunk.text for hit in hits)


def answer_bare(question: str, hits: list[Hit], llm: LLMCall) -> str:
    """关闭防护：检索结果直接拼进提示词，模型的回答原样返回，不做任何校验。"""
    return llm(BARE_SYSTEM_PROMPT, f"制度内容：\n{build_context(hits)}\n\n员工问题：{question}")


@dataclass(frozen=True)
class GuardedResult:
    answer: Answer
    hits: list[Hit]
    reason: str  # 这次回答或拒答的原因，便于命令行展示和记录


class GuardedQA:
    """开启防护：依次经过依据门槛、模型判定和回答校验，任何一步不满足都不让模型自由作答。"""

    def __init__(
        self,
        policy: Policy,
        retriever: HybridRetriever,
        handoff_message: str,
        llm: LLMCall | None,
        min_similarity: float,
        strict_unknown_terms: bool = False,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.policy = policy
        self.retriever = retriever
        self.handoff_message = handoff_message
        self.llm = llm
        self.min_similarity = min_similarity
        self.strict_unknown_terms = strict_unknown_terms
        self.top_k = top_k
        # 现有的关键词问答。“出现制度外词语就拒答”的规则很严格，所以作为降级方案
        self._fallback = PolicyQA(policy, handoff_message)

    def _handoff(self, hits: list[Hit], reason: str, mode: str = "local") -> GuardedResult:
        return GuardedResult(Answer(HANDOFF, self.handoff_message, mode=mode), hits, reason)

    def _downgrade(self, question: str, hits: list[Hit], reason: str) -> GuardedResult:
        """退回关键词检索并摘录条款原文，不再让模型生成任何内容。"""
        answer = self._fallback.ask(question)
        if answer.status == ANSWERED:
            answer = Answer(ANSWERED, answer.text, answer.citations, "local", reason)
        return GuardedResult(answer, hits, f"{reason}，已退回关键词检索")

    def ask(self, question: str) -> GuardedResult:
        hits = self.retriever.search(question, self.top_k)

        # 降级：没有大模型或没有向量模型时，语义门槛不可用，改用严格的关键词规则
        if self.llm is None or not self.retriever.uses_vectors:
            return self._downgrade(question, hits, "大模型或向量模型不可用")

        # 第一关，依据门槛：最相关的切片也不够接近，就直接拒答，并且不调用模型
        best = max((hit.vector_score for hit in hits), default=0.0)
        if best < self.min_similarity:
            return self._handoff(hits, f"最高相似度 {best:.3f} 低于门槛 {self.min_similarity}")

        # 可选的严格模式：问题里只要有制度没覆盖的词就拒答，同样不调用模型
        if self.strict_unknown_terms and not retrieve(question, self.policy.clauses):
            return self._handoff(hits, "问题中有制度未覆盖的内容（严格模式）")

        # 第二关，模型判定：模型只能依据这些条款作答，条款不足时要输出固定的拒答语
        prompt = f"条款：\n{build_context(hits)}\n\n用户问题：{question}"
        try:
            text = self.llm(SYSTEM_PROMPT, prompt).strip()
        except Exception as exc:  # 断网、超时、Key 无效都走降级
            return self._downgrade(question, hits, f"大模型调用失败（{type(exc).__name__}）")
        if self.handoff_message in text:
            return self._handoff(hits, "模型判定条款不足", mode="llm")

        # 第三关，回答校验：引用必须来自本次检索，数字必须出现在被引用的条款中
        clauses = list({hit.chunk.clause.id: hit.chunk.clause for hit in hits}.values())
        cited = check_llm_answer(text, clauses)
        if cited is None:
            return self._downgrade(question, hits, "模型回答未通过引用校验")
        return GuardedResult(Answer(ANSWERED, text, cited, "llm"), hits, "通过全部防护")
