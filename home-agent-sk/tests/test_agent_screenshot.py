from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
from semantic_kernel.contents import AuthorRole, ChatHistory, ImageContent

from home_agent.core.agent import HomeAgent


def _agent(tmp_path: Path, replies: list[str], capture_on_first_call: bool) -> HomeAgent:
    shot = tmp_path / "screen.png"
    Image.new("RGB", (8, 8), "red").save(shot)
    screenshot_plugin = MagicMock(captures=[])

    async def fake_stream(**_kwargs):
        if capture_on_first_call and not screenshot_plugin.captures:
            screenshot_plugin.captures.append(shot)   # the model called capture_screen
        yield MagicMock(content=replies.pop(0), items=[], role=AuthorRole.ASSISTANT)

    service = MagicMock(get_streaming_chat_message_content=MagicMock(side_effect=fake_stream))
    kernel = MagicMock(get_service=MagicMock(return_value=service))
    return HomeAgent(
        kernel=kernel,
        history=ChatHistory(),
        settings=MagicMock(),
        screenshot_plugin=screenshot_plugin,
    )


def _images(agent: HomeAgent) -> list[ImageContent]:
    return [i for m in agent.history.messages for i in m.items if isinstance(i, ImageContent)]


async def test_screenshot_is_shown_to_model_and_answer_comes_from_follow_up(tmp_path: Path):
    agent = _agent(tmp_path, ["(blind guess)", "I can see a red square."], True)

    reply = await agent.chat("what's on my screen?")

    assert reply == "I can see a red square."
    assert len(_images(agent)) == 1


async def test_no_follow_up_without_screenshot(tmp_path: Path):
    agent = _agent(tmp_path, ["Hello, sir."], False)

    assert await agent.chat("hi") == "Hello, sir."
    assert _images(agent) == []


async def test_only_latest_screenshot_keeps_its_image(tmp_path: Path):
    agent = _agent(tmp_path, ["a", "first look", "b", "second look"], True)
    await agent.chat("look")
    agent.screenshot_plugin.captures.clear()   # next call captures again
    await agent.chat("look again")

    assert len(_images(agent)) == 1


async def test_blind_guess_before_the_screenshot_is_never_spoken(tmp_path: Path):
    agent = _agent(tmp_path, ["I'd guess a spreadsheet.", "It's a red square, sir."], True)
    spoken: list[str] = []

    await agent.chat("what's on my screen?", on_sentence=spoken.append)

    assert spoken == ["It's a red square, sir."]
