from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext

from home_agent.core.settings import AppSettings

CHAT_SERVICE_ID = "chat"

_tools_logger = logging.getLogger("home_agent.tools")


async def _tool_invocation_filter(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Coroutine[Any, Any, None]],
) -> None:
    plugin = context.function.plugin_name or "?"
    fn = context.function.name

    args = context.arguments
    if args:
        pairs = []
        for k, v in args.items():
            val = str(v)
            if len(val) > 120:
                val = val[:117] + "..."
            pairs.append(f'{k}="{val}"')
        args_str = "  [ " + "  ".join(pairs) + " ]" if pairs else ""
    else:
        args_str = ""

    _tools_logger.info("→ INVOKE  %s → %s%s", plugin, fn, args_str)

    try:
        await next(context)
    except Exception as exc:
        _tools_logger.info("✗ ERROR   %s → %s: %s", plugin, fn, exc)
        raise

    result = context.result
    if result is not None:
        result_str = str(result.value) if hasattr(result, "value") else str(result)
        if len(result_str) > 200:
            result_str = result_str[:197] + "..."
        _tools_logger.info("✓ RESULT  %s", result_str)


def build_kernel(settings: AppSettings) -> Kernel:
    kernel = Kernel()

    if settings.ai_provider == "openai":
        chat = OpenAIChatCompletion(
            api_key=settings.openai_api_key,
            ai_model_id=settings.openai_chat_model_id,
            service_id=CHAT_SERVICE_ID,
        )
    else:
        ollama_client = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
        )
        chat = OpenAIChatCompletion(
            async_client=ollama_client,
            ai_model_id=settings.ollama_chat_model_id,
            service_id=CHAT_SERVICE_ID,
        )
    kernel.add_service(chat)
    kernel.add_filter("function_invocation", _tool_invocation_filter)

    return kernel
