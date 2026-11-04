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


def load_llm_settings(env_file: Path | None = None) -> LLMSettings:
    load_dotenv(env_file or ROOT / ".env")
    return LLMSettings(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "").strip(),
        model=os.getenv("LLM_MODEL", "").strip(),
    )
