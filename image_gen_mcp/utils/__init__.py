"""Utility functions and helpers."""

from .cache import CacheManager
from .openai_client import OpenAIClientManager

__all__ = ["CacheManager", "OpenAIClientManager"]


def detect_image_mime(data_url: str | None, image_bytes: bytes) -> tuple[str, str]:
    """Return (filename, mime_type) from a data-URL prefix or byte magic.

    Falls back to PNG when the format cannot be determined.
    """
    if data_url and data_url.startswith("data:"):
        # e.g. "data:image/webp;base64,..."
        mime = data_url.split(";", 1)[0].removeprefix("data:")
        ext = mime.split("/", 1)[1] if "/" in mime else "png"
        return f"image.{ext}", mime

    # Magic-byte sniffing
    if image_bytes[:4] == b"\x89PNG":
        return "image.png", "image/png"
    if image_bytes[:2] == b"\xff\xd8":
        return "image.jpeg", "image/jpeg"
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image.webp", "image/webp"

    return "image.png", "image/png"
