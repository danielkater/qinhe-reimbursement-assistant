"""大模型调用（OpenAI 兼容接口）。查单数据永远不经过这里。"""
from collections.abc import Callable

from qinghe.utils.config import LLMSettings

LLMCall = Callable[[str, str], str]


def build_llm(settings: LLMSettings, timeout: float, temperature: float = 0) -> LLMCall | None:
    """没配置 Key 时返回 None，由上层走本地模式。"""
    if not settings.enabled:
        return None
    from openai import OpenAI  # 延迟导入，本地模式不依赖它

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url or None,
        timeout=timeout,
        max_retries=0,
    )

    def call(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=settings.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    return call
