"""读取 config/default.yaml 和 .env。"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config" / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_embedding_settings(config: dict, env_file: Path | None = None) -> "EmbeddingSettings":
    """读取嵌入模型配置。供应商和模型在 YAML 里，在线接口的密钥只放 .env。"""
    from qinghe.core.embeddings import EmbeddingSettings

    load_dotenv(env_file or ROOT / ".env")
    embedding = config["embedding"]
    return EmbeddingSettings(
        provider=embedding["provider"],
        model=embedding["model"],
        cache_dir=ROOT / embedding["cache_dir"],
        query_prefix=embedding.get("query_prefix", ""),
        api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
        base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
    )


def load_llm_settings(env_file: Path | None = None) -> LLMSettings:
    load_dotenv(env_file or ROOT / ".env")
    return LLMSettings(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "").strip(),
        model=os.getenv("LLM_MODEL", "").strip(),
    )
