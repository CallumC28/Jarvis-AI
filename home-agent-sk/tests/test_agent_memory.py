from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)

from home_agent.core.agent import HomeAgent, pop_sentences, strip_think
from home_agent.core.composition import _load_history
from home_agent.memory.json_store import JsonStore
from home_agent.memory.models import AgentState
from home_agent.memory.recall import Recall, append_turn


def _chunk(text: str) -> MagicMock:
    return MagicMock(content=text, items=[], role=AuthorRole.ASSISTANT)


def _agent(tmp_path: Path, pieces: list[str], summary_reply: str = "") -> HomeAgent:
    async def stream(**_kwargs):
        for p in pieces:
            yield _chunk(p)

    service = MagicMock(
        get_streaming_chat_message_content=MagicMock(side_effect=stream),
        get_chat_message_content=AsyncMock(return_value=MagicMock(content=summary_reply)),
    )
    return HomeAgent(
        kernel=MagicMock(get_service=MagicMock(return_value=service)),
        history=ChatHistory(),
        settings=MagicMock(extra_body=None),
        history_path=tmp_path / "history.json",
        system_prompt=lambda: "BASE",
        turn_context=lambda: "[Monday 09:00]",
        log_path=tmp_path / "log.jsonl",
    )


def test_pop_sentences_splits_finished_sentences_and_keeps_the_tail():
    sentences, tail = pop_sentences("Ah. That one is easy enough. The answer is")
    assert sentences == ["Ah. That one is easy enough."]
    assert tail == "The answer is"


def test_pop_sentences_never_leaks_thinking():
    assert pop_sentences("<think>Hmm, the user wants. Something.</think>Four, sir. ") == (
        [], "Four, sir. "
    )
    sentences, tail = pop_sentences("<think>still going. and going. ")
    assert sentences == [] and tail.startswith("<think>")
    assert strip_think("Hi <think>unfinished") == "Hi "


async def test_chat_streams_sentences_and_prefixes_context(tmp_path: Path):
    agent = _agent(tmp_path, ["The weather is ", "dry today. ", "Take a ", "jacket anyway."])
    spoken: list[str] = []

    reply = await agent.chat("weather?", on_sentence=spoken.append)

    assert spoken == ["The weather is dry today.", "Take a jacket anyway."]
    assert reply == "The weather is dry today. Take a jacket anyway."
    assert agent.history.messages[0].content == "BASE"
    assert agent.history.messages[1].content == "[Monday 09:00]\nweather?"
    logged = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8"))
    assert logged["user"] == "weather?"   # log keeps what the user said, not the context line


async def test_old_messages_fold_into_a_persisted_summary(tmp_path: Path):
    agent = _agent(tmp_path, ["Noted."], summary_reply="They like jazz and live in Leeds.")
    for i in range(25):
        agent.history.add_user_message(f"question {i}")
        agent.history.add_assistant_message(f"answer {i}")

    await agent.chat("one more")

    body = [m for m in agent.history.messages if m.role != AuthorRole.SYSTEM]
    assert len(body) <= 21 and body[0].role == AuthorRole.USER
    assert agent.summary == "They like jazz and live in Leeds."
    assert "They like jazz" in agent.history.messages[0].content

    restored = ChatHistory()
    assert _load_history(restored, tmp_path / "history.json") == agent.summary
    assert len(restored.messages) == len(body)


def test_recall_finds_the_turn_by_meaning(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    append_turn(log, "my sister Anna lives in Perth", "Noted, sir.")
    append_turn(log, "what's 2+2", "Four.")
    words = ["sister", "perth", "anna", "maths"]

    def embed(texts: list[str]) -> list[list[float]]:   # bag-of-words stand-in for Ollama
        return [[float(w in t.lower()) + 0.01 for w in words] for t in texts]

    recall = Recall(log, tmp_path / "index.json", embed)
    assert "Anna" in recall.search("where does my sister live")[0]
    assert recall.search("sister", extra=["Note: buy milk"], k=1)[0].count("Anna") == 1


async def test_summary_step_saves_facts_the_model_forgot_to(tmp_path: Path):
    store = JsonStore(tmp_path / "state.json")
    store.save(AgentState(facts=["Sister Anna lives in Perth."]))
    reply = (
        "They chatted about coffee.\nFACTS:\n- Takes coffee black.\n"
        "- sister anna lives in perth."   # already known, different case: not duplicated
    )
    agent = _agent(tmp_path, ["Noted."], summary_reply=reply)
    agent.fact_store = store
    for i in range(25):
        agent.history.add_user_message(f"question {i}")
        agent.history.add_assistant_message(f"answer {i}")

    await agent.chat("one more")

    assert store.load().facts == ["Sister Anna lives in Perth.", "Takes coffee black."]
    assert agent.summary == "They chatted about coffee."


def test_saved_history_keeps_tool_calls_so_a_restart_sees_what_was_really_done(tmp_path: Path):
    agent = _agent(tmp_path, [])
    agent.history.add_user_message("open the steam store")
    agent.history.add_message(ChatMessageContent(role=AuthorRole.ASSISTANT, items=[
        FunctionCallContent(id="c1", plugin_name="pc", function_name="open_uri",
                            arguments='{"uri": "steam://open/games"}'),
    ]))
    agent.history.add_message(ChatMessageContent(role=AuthorRole.TOOL, items=[
        FunctionResultContent(id="c1", plugin_name="pc", function_name="open_uri",
                              result="Opened steam://open/games."),
    ]))
    agent.history.add_assistant_message("Steam's up.")
    agent._persist_history()

    restored = ChatHistory()
    _load_history(restored, tmp_path / "history.json")

    kinds = [type(m.items[0]).__name__ for m in restored.messages]
    assert kinds == ["TextContent", "FunctionCallContent", "FunctionResultContent", "TextContent"]


def test_old_text_only_history_still_loads(tmp_path: Path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"summary": "s", "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "Evening."},
    ]}), encoding="utf-8")
    restored = ChatHistory()
    assert _load_history(restored, path) == "s"
    assert [m.content for m in restored.messages] == ["hi", "Evening."]
