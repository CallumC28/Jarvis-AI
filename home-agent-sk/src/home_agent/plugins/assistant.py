from __future__ import annotations

from semantic_kernel.functions import kernel_function


class AssistantPlugin:
    """Conversation controls, so the user can say these however they like. The functions do
    nothing themselves: VoicePipeline reads agent.last_tools after the turn and acts."""

    @kernel_function(
        name="end_conversation",
        description=(
            "Call when the user is finished talking to you for now, however they put it "
            "('right, I'm off', 'that'll do', 'leave it there', 'cheers, mate'). "
            "You stop listening once your reply is spoken."
        ),
    )
    def end_conversation(self) -> str:
        return "Ending. Sign off in two or three words."

    @kernel_function(
        name="start_dictation",
        description=(
            "Call when the user wants to dictate: type out what they say next into the window "
            "they're in ('type this for me', 'take this down', 'I'll dictate an email'). "
            "Saying 'stop dictation' ends it."
        ),
    )
    def start_dictation(self) -> str:
        return "Dictation starts after your reply. Say only 'Go ahead.'"

    @kernel_function(
        name="use_openai",
        description="Switch yourself to the OpenAI (cloud) model when the user asks for it.",
    )
    def use_openai(self) -> str:
        return "Switching after your reply. Say only 'Switching.'"

    @kernel_function(
        name="use_local",
        description="Switch yourself to the local (Ollama) model when the user asks for it.",
    )
    def use_local(self) -> str:
        return "Switching after your reply. Say only 'Switching.'"
