"""BM25 + 向量（FAISS）混合检索。

【后续复用：核心模块】后面阶段的“查制度”工具直接调用这里。

两种检索互补：
- BM25 看字面是否相同，适合“第2条”“OFD”这类明确的词；
- 向量看意思是否接近，适合“住酒店”和“住宿”这类换了说法的问题。

本模块只负责找出最相关的条款，不判断“够不够作为依据”。
混合检索总能返回结果，是否有依据由 core/rag.py 的依据门槛判断。
"""
from dataclasses import dataclass

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .embeddings import Embedder
from .retriever import PUNCTUATION

jieba.setLogLevel("WARNING")  # 关闭 jieba 加载词典时的提示
DEFAULT_TOP_K = 3
# RRF 融合的常数，60 是这个算法常用的默认值
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    rrf_score: float  # 融合后的得分，只用于排序
    bm25_score: float  # 字面匹配得分
    vector_score: float | None  # 余弦相似度；没有向量模型时为 None


def tokenize(text: str) -> list[str]:
    """中文分词，并去掉空白和标点。"""
    return jieba.lcut(PUNCTUATION.sub(" ", text.lower()).replace(" ", ""))


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder | None = None,
        query_prefix: str = "",
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.query_prefix = query_prefix
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._index = self._build_index() if embedder else None

    @property
    def uses_vectors(self) -> bool:
        return self._index is not None

    def _build_index(self) -> faiss.IndexFlatIP:
        """向量化全部切片并存入 FAISS。

        归一化后，内积就等于余弦相似度，取值越接近 1 表示意思越接近。
        原型每次启动时重新构建；阶段 3 发布制度时再改为保存到磁盘。
        """
        vectors = self.embedder([c.text for c in self.chunks])
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def _vector_scores(self, query: str) -> np.ndarray:
        """计算问题和每个切片的余弦相似度。"""
        vector = self.embedder([self.query_prefix + query])
        faiss.normalize_L2(vector)
        scores, ids = self._index.search(vector, len(self.chunks))
        result = np.zeros(len(self.chunks), dtype="float32")
        result[ids[0]] = scores[0]
        return result

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Hit]:
        """返回最相关的 top_k 个切片。

        用 RRF（倒数排名融合）合并两个排名：名次越靠前得分越高。
        它只看名次，所以不用处理 BM25 分数和相似度量纲不同的问题。
        """
        bm25_scores = self._bm25.get_scores(tokenize(query))
        rrf = np.zeros(len(self.chunks))
        # BM25 得分为 0 表示字面完全没有重合，不参与排名
        bm25_order = [i for i in np.argsort(-bm25_scores) if bm25_scores[i] > 0]
        for rank, i in enumerate(bm25_order, start=1):
            rrf[i] += 1 / (self.rrf_k + rank)

        vector_scores = None
        if self.uses_vectors:
            vector_scores = self._vector_scores(query)
            for rank, i in enumerate(np.argsort(-vector_scores), start=1):
                rrf[i] += 1 / (self.rrf_k + rank)

        order = [i for i in np.argsort(-rrf) if rrf[i] > 0][:top_k]
        return [
            Hit(
                chunk=self.chunks[i],
                rrf_score=float(rrf[i]),
                bm25_score=float(bm25_scores[i]),
                vector_score=None if vector_scores is None else float(vector_scores[i]),
            )
            for i in order
        ]
