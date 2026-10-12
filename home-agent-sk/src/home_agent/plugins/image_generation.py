from __future__ import annotations

import base64
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from openai import AsyncOpenAI
from semantic_kernel.functions import kernel_function


class ImageGenerationPlugin:
    """OpenAI image generation tools."""

    _VALID_SIZES = {"1024x1024", "1536x1024", "1024x1536", "auto"}
    _VALID_QUALITIES = {"low", "medium", "high", "auto"}
    _EDIT_SAFETY_PREFIX = (
        "Preserve the full original image composition. "
        "Do not crop, zoom, trim, or cut off any part of the source image. "
        "Keep all added text fully inside the frame with comfortable margins. "
        "If space is tight, reduce text size or reflow text so nothing is clipped."
    )

    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-image-1",
        on_image: Callable[[Path], None] | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model_id = model_id
        self._on_image = on_image
        self._output_dir = Path(tempfile.gettempdir()) / "home-agent-generated-images"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._latest_user_images: list[Path] = []

    def set_latest_user_images(self, image_paths: list[Path]) -> None:
        self._latest_user_images = list(image_paths)

    @classmethod
    def _normalise_size(cls, size: str) -> str:
        value = (size or "").strip().lower()
        return value if value in cls._VALID_SIZES else "auto"

    @classmethod
    def _normalise_quality(cls, quality: str) -> str:
        value = (quality or "").strip().lower()
        return value if value in cls._VALID_QUALITIES else "auto"

    def _save_b64_image(self, b64: str, prefix: str, label: str) -> str:
        """Decode a base64 image, save it, fire the on_image callback, and return markdown."""
        try:
            raw = base64.b64decode(b64)
            out = self._output_dir / f"{prefix}-{uuid.uuid4().hex}.png"
            out.write_bytes(raw)
            if self._on_image is not None:
                self._on_image(out)
            return f"![{label}]({out.resolve().as_posix()})"
        except Exception as exc:
            return f"Image {prefix} failed: could not save image data ({exc})."

    @kernel_function(
        name="generate_image",
        description=(
            "Generate an image from a text prompt. Returns markdown image output "
            "that can be rendered directly in chat."
        ),
    )
    async def generate_image(
        self,
        prompt: Annotated[str, "Text description of the image to generate"],
        size: Annotated[str, "Requested size (for example: 1024x1024)"] = "1024x1024",
        quality: Annotated[str, "Generation quality (for example: low, medium, high)"] = "medium",
    ) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "Image prompt cannot be empty."

        try:
            response = await self._client.images.generate(
                model=self._model_id,
                prompt=prompt,
                size=self._normalise_size(size),
                quality=self._normalise_quality(quality),
            )
        except Exception as exc:
            return f"Image generation failed: {exc}"

        if not response.data:
            return "Image generation failed: empty response."

        first = response.data[0]
        if b64 := getattr(first, "b64_json", None):
            return self._save_b64_image(b64, "generated", "generated image")
        if url := getattr(first, "url", None):
            return f"![generated image]({url})"
        return "Image generation succeeded, but no image payload was returned."

    @kernel_function(
        name="edit_last_uploaded_image",
        description=(
            "Edit the user's most recently uploaded image with a prompt. "
            "Great for meme-style edits like adding a funny caption. "
            "Returns markdown image output that can be rendered directly in chat."
        ),
    )
    async def edit_last_uploaded_image(
        self,
        prompt: Annotated[str, "How to edit the uploaded image (for example meme caption text and style)"],
        size: Annotated[str, "Requested output size (for example: 1024x1024)"] = "1024x1024",
        quality: Annotated[str, "Generation quality (for example: low, medium, high)"] = "medium",
    ) -> str:
        prompt = prompt.strip()
        if not prompt:
            return "Edit prompt cannot be empty."
        if not self._latest_user_images:
            return "No uploaded image found. Upload or paste an image first, then ask me to edit it."

        image_path = self._latest_user_images[-1]
        if not image_path.exists():
            return f"Uploaded image could not be found on disk: {image_path}"

        try:
            with image_path.open("rb") as f:
                safe_prompt = f"{self._EDIT_SAFETY_PREFIX}\n\nUser request: {prompt}"
                response = await self._client.images.edit(
                    model=self._model_id,
                    image=f,
                    prompt=safe_prompt,
                    size=self._normalise_size(size),
                    quality=self._normalise_quality(quality),
                )
        except Exception as exc:
            return f"Image edit failed: {exc}"

        if not response.data:
            return "Image edit failed: empty response."

        first = response.data[0]
        if b64 := getattr(first, "b64_json", None):
            return self._save_b64_image(b64, "edited", "edited image")
        if url := getattr(first, "url", None):
            return f"![edited image]({url})"
        return "Image edit succeeded, but no image payload was returned."
