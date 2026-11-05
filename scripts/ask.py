"""提示词 03 的命令行问答原型，不涉及 Web。

【原型专用】本脚本只负责命令行交互；可复用的逻辑都在 src/qinghe 中（决策 X5）。

用法：
    python scripts/ask.py                               开启防护，交互提问
    python scripts/ask.py --mode guard-off              关闭防护，观察模型会不会编
    python scripts/ask.py --mode guard-on -q "发票丢了怎么办？"

流程：读取 Markdown 制度 → 按条款切片 → 向量化存入 FAISS
      → 问题走 BM25 + 向量混合检索取 top3 → 拼入提示词 → 调用大模型。
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qinghe.agent.router import is_claim_request
from qinghe.core.chunker import chunk_policy
from qinghe.core.embeddings import EmbeddingUnavailable, build_embedder
from qinghe.core.hybrid import Hit, HybridRetriever
from qinghe.core.llm import LLMCall, build_llm
from qinghe.core.policy import Policy, load_policy_markdown
from qinghe.core.qa import HANDOFF
from qinghe.core.rag import GuardedQA, answer_bare
from qinghe.utils.config import load_config, load_embedding_settings, load_llm_settings

GUARD_ON, GUARD_OFF = "guard-on", "guard-off"
CLAIM_NOTICE = "这是查单请求，不会发给外部模型。请到“授权查单”页查询，系统会先校验你的身份。"


@dataclass
class Runtime:
    """一次运行需要的全部对象，启动时准备一次，之后每个问题共用。"""

    policy: Policy
    retriever: HybridRetriever
    llm: LLMCall | None
    guarded: GuardedQA
    model_name: str
    embedding_note: str


def load_runtime() -> Runtime:
    config = load_config()
    hybrid = config["retrieval"]["hybrid"]

    # 第 1 步：读取 Markdown 制度。只接受有效且公开的文件，内部资料不会进入后面的流程
    policy = load_policy_markdown(ROOT / config["paths"]["policy_markdown"])

    # 第 2 步：按条款切片，单片不超过 500 字（为什么这样切，见 core/chunker.py）
    chunks = chunk_policy(policy, hybrid["chunk_max_chars"])

    # 第 3 步：加载嵌入模型。失败时不中断，只用 BM25，并告诉用户原因
    settings = load_embedding_settings(config)
    try:
        embedder = build_embedder(settings)
        embedding_note = settings.model if embedder else "未启用（配置为 none）"
    except EmbeddingUnavailable as exc:
        embedder = None
        embedding_note = f"不可用，已改用 BM25：{exc}"

    # 第 4 步：建立 BM25 和 FAISS 索引
    retriever = HybridRetriever(chunks, embedder, settings.query_prefix, hybrid["rrf_k"])

    # 第 5 步：准备大模型。供应商、地址和模型名都来自 .env，没有配置时 llm 为 None
    llm_settings = load_llm_settings()
    llm = build_llm(llm_settings, config["llm"]["timeout_seconds"], config["llm"]["temperature"])

    guarded = GuardedQA(
        policy,
        retriever,
        config["handoff"]["message"],
        llm,
        hybrid["min_similarity"],
        hybrid["strict_unknown_terms"],
        hybrid["top_k"],
    )
    return Runtime(
        policy, retriever, llm, guarded, llm_settings.model or "未配置", embedding_note
    )


def format_hits(hits: list[Hit]) -> str:
    """显示检索到的条款和得分，方便判断问题出在检索还是模型。"""
    lines = []
    for rank, hit in enumerate(hits, start=1):
        similarity = "无" if hit.vector_score is None else f"{hit.vector_score:.3f}"
        lines.append(
            f"  {rank}. {hit.chunk.id} {hit.chunk.clause.title}"
            f"（相似度 {similarity}，BM25 {hit.bm25_score:.2f}）"
        )
    return "\n".join(lines) or "  （没有检索到内容）"


def answer_question(runtime: Runtime, mode: str, question: str) -> dict:
    """回答一个问题，返回结构化结果。查单请求在这里被拦下，两种模式都一样。"""
    # “关闭防护”只关问答防护，不放宽数据边界：查单请求任何时候都不外发
    if is_claim_request(question):
        return {"status": "claim", "hits": [], "text": CLAIM_NOTICE, "reason": "本地分流", "citations": []}

    if mode == GUARD_OFF:
        if runtime.llm is None:
            raise SystemExit("关闭防护模式需要大模型，请先在 .env 中配置 LLM_API_KEY 和 LLM_MODEL。")
        hits = runtime.retriever.search(question)
        text = answer_bare(question, hits, runtime.llm)
        return {"status": "bare", "hits": hits, "text": text, "reason": "未做任何防护", "citations": []}

    result = runtime.guarded.ask(question)
    return {
        "status": result.answer.status,
        "hits": result.hits,
        "text": result.answer.text,
        "reason": result.reason,
        "citations": list(result.answer.citations),
    }


def print_result(runtime: Runtime, result: dict) -> None:
    print("\n检索到的 top3：")
    print(format_hits(result["hits"]))
    print(f"\n回答：\n{result['text']}")
    print(f"\n判定：{result['reason']}")
    if result["status"] == HANDOFF:
        print(load_config()["handoff"]["contact"])
    # 开启防护时附上版本和条款原文，让用户能自己核对
    if result["citations"]:
        print(f"\n依据（{runtime.policy.name} {runtime.policy.version}）：")
        for clause in result["citations"]:
            print(f"  {clause.id} {clause.title}：{clause.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="青禾报销助手命令行问答原型")
    parser.add_argument(
        "--mode",
        choices=[GUARD_ON, GUARD_OFF],
        default=GUARD_ON,
        help="guard-on：无依据就拒答并转人工（默认）；guard-off：关闭防护，观察模型会不会编",
    )
    parser.add_argument("-q", "--question", help="只问一个问题；不提供时进入交互模式")
    args = parser.parse_args()

    runtime = load_runtime()
    print(f"模式：{args.mode}｜大模型：{runtime.model_name}｜嵌入模型：{runtime.embedding_note}")
    if args.mode == GUARD_OFF:
        print("提醒：关闭防护只用于观察，回答可能是编造的，不能作为报销依据。")

    if args.question:
        print_result(runtime, answer_question(runtime, args.mode, args.question))
        return

    print("输入问题后按回车，输入 q 退出。")
    while True:
        question = input("\n问题：").strip()
        if question.lower() in {"q", "quit", "exit", "退出"}:
            break
        if question:
            print_result(runtime, answer_question(runtime, args.mode, question))


if __name__ == "__main__":
    main()
