"""嵌入模型：把文字转成向量。供应商和模型都由配置决定。

【后续复用：核心模块】后面阶段的混合检索和制度后台都通过 build_embedder 取得嵌入模型。

默认使用本地的中文小模型 BAAI/bge-small-zh-v1.5：
- DeepSeek 没有 embedding 接口（/embeddings 返回 404），不能和对话模型共用一家服务；
- 本地模型不需要额外的 API Key，问题也不用再发给另一家服务；
- 首次运行要联网下载约 90 MB，之后可以断网使用。
"""
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 输入一组文字，返回“文字数 × 向量维度”的矩阵
Embedder = Callable[[list[str]], np.ndarray]


class EmbeddingUnavailable(RuntimeError):
    """嵌入模型用不了，例如模型下载失败或接口不通。调用方应走降级路径。"""


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str  # fastembed（本地）、openai_compatible（在线接口）、none（不用向量）
    model: str
    cache_dir: Path  # 本地模型的存放目录
    query_prefix: str = ""  # 部分模型建议给“查询”加的前缀
    api_key: str = ""  # 只有在线接口才需要
    base_url: str = ""


def _build_fastembed(settings: EmbeddingSettings) -> Embedder:
    # Windows 默认不支持符号链接，这只影响缓存占用，不影响结果，所以关掉这条提示
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # 延迟导入：Streamlit 页面不用向量，不必加载这些较重的依赖
    from fastembed import TextEmbedding

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(settings.model, cache_dir=str(settings.cache_dir))

    def embed(texts: list[str]) -> np.ndarray:
        return np.array(list(model.embed(texts)), dtype="float32")

    return embed


def _build_openai_compatible(settings: EmbeddingSettings) -> Embedder:
    # 需要在线嵌入服务时使用。注意：这样问题会多发给一家外部服务
    if not settings.api_key:
        raise EmbeddingUnavailable("在线嵌入接口需要在 .env 中配置 EMBEDDING_API_KEY")
    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url or None)

    def embed(texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(model=settings.model, input=texts)
        return np.array([item.embedding for item in response.data], dtype="float32")

    return embed


def build_embedder(settings: EmbeddingSettings) -> Embedder | None:
    """按配置创建嵌入模型。provider 为 none 时返回 None，表示明确不用向量。"""
    builders = {"fastembed": _build_fastembed, "openai_compatible": _build_openai_compatible}
    if settings.provider == "none":
        return None
    if settings.provider not in builders:
        raise EmbeddingUnavailable(f"不支持的嵌入供应商：{settings.provider}")
    try:
        return builders[settings.provider](settings)
    except EmbeddingUnavailable:
        raise
    except Exception as exc:  # 断网、下载失败、依赖缺失都归到这里，由调用方决定如何降级
        raise EmbeddingUnavailable(f"{type(exc).__name__}: {exc}") from exc
