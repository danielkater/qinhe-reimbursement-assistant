"""大模型调用（OpenAI 兼容接口）。查单数据永远不经过这里。

【后续复用：核心模块】后面阶段的智能体编排也通过这里调用模型。
供应商由 .env 决定：换一家服务只要改 LLM_BASE_URL 和 LLM_MODEL，不用改代码。
"""
from collections.abc import Callable

from qinghe.utils.config import LLMSettings

# 统一的调用形式：输入系统提示词和用户内容，返回模型回答的文字
LLMCall = Callable[[str, str], str]


def call_llm(
    settings: LLMSettings, system: str, user: str, timeout: float = 10, temperature: float = 0
) -> str:
    """调用一次大模型，所有模型调用都走这个函数。"""
    from openai import OpenAI  # 延迟导入，本地模式不依赖它

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url or None,
        timeout=timeout,
        max_retries=0,  # 不重试：失败就走降级，避免现场一直等待
    )
    response = client.chat.completions.create(
        model=settings.model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def build_llm(settings: LLMSettings, timeout: float, temperature: float = 0) -> LLMCall | None:
    """绑定好配置，返回可以直接调用的函数。没配置 Key 时返回 None，由上层走本地模式。"""
    if not settings.enabled:
        return None

    def call(system: str, user: str) -> str:
        return call_llm(settings, system, user, timeout, temperature)

    return call
