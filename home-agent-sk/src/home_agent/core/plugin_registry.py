from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from semantic_kernel import Kernel

from home_agent.core.settings import AppSettings
from home_agent.memory.json_store import JsonStore
from home_agent.memory.recall import Recall
from home_agent.plugins.assistant import AssistantPlugin
from home_agent.plugins.browser import BrowserPlugin
from home_agent.plugins.code_reader import CodeReaderPlugin
from home_agent.plugins.desktop_folder import DesktopFolderPlugin
from home_agent.plugins.git_status import GitStatusPlugin
from home_agent.plugins.image_generation import ImageGenerationPlugin
from home_agent.plugins.memory import MemoryPlugin
from home_agent.plugins.pc import PCPlugin
from home_agent.plugins.personal_organiser import PersonalOrganizerPlugin
from home_agent.plugins.screenshot import ScreenshotPlugin
from home_agent.plugins.tavily_search import TavilySearchPlugin


class RegisteredPlugins(NamedTuple):
    tavily: TavilySearchPlugin | None
    image: ImageGenerationPlugin | None
    screenshot: ScreenshotPlugin


def register_default_plugins(
    kernel: Kernel,
    settings: AppSettings,
    store: JsonStore,
    recall: Recall,
    on_progress: Callable[[str], None] | None = None,
    on_image: Callable[[object], None] | None = None,
) -> RegisteredPlugins:
    kernel.add_plugin(PersonalOrganizerPlugin(store), plugin_name="organizer")
    kernel.add_plugin(DesktopFolderPlugin(), plugin_name="desktop")
    screenshot_plugin = ScreenshotPlugin(on_image=on_image)
    kernel.add_plugin(screenshot_plugin, plugin_name="screenshot")
    kernel.add_plugin(BrowserPlugin(), plugin_name="browser")
    kernel.add_plugin(PCPlugin(), plugin_name="pc")
    kernel.add_plugin(MemoryPlugin(store, recall), plugin_name="memory")
    kernel.add_plugin(AssistantPlugin(), plugin_name="assistant")

    image_plugin: ImageGenerationPlugin | None = None
    if settings.ai_provider == "openai":
        image_plugin = ImageGenerationPlugin(
            api_key=settings.openai_api_key,
            model_id=settings.openai_image_model_id,
            on_image=on_image,
        )
        kernel.add_plugin(image_plugin, plugin_name="images")

    tavily_plugin: TavilySearchPlugin | None = None
    if settings.tavily_api_key:
        tavily_plugin = TavilySearchPlugin(
            api_key=settings.tavily_api_key,
            progress=on_progress,
        )
        kernel.add_plugin(tavily_plugin, plugin_name="web")

    if settings.code_folders:
        kernel.add_plugin(CodeReaderPlugin(settings.code_folders), plugin_name="code")

    if settings.gitstat_dir:
        kernel.add_plugin(GitStatusPlugin(settings.gitstat_dir), plugin_name="git")

    return RegisteredPlugins(tavily=tavily_plugin, image=image_plugin, screenshot=screenshot_plugin)
