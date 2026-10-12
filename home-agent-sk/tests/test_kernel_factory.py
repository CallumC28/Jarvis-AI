from __future__ import annotations

from semantic_kernel.functions import kernel_function

from home_agent.core.kernel_factory import build_kernel
from home_agent.core.settings import AppSettings


class _Echo:
    @kernel_function(name="echo")
    def echo(self, text: str) -> str:
        return text


async def test_plugins_can_be_invoked_through_the_logging_filter():
    kernel = build_kernel(AppSettings(ai_provider="ollama"))
    kernel.add_plugin(_Echo(), plugin_name="t")

    result = await kernel.invoke(plugin_name="t", function_name="echo", text="hi")

    assert str(result) == "hi"
