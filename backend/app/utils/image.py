from __future__ import annotations

import io

from PIL import Image, ImageOps


def load_image(data: bytes, max_side: int = 1280) -> tuple[Image.Image, bytes]:
    """Decode an upload, honour EXIF rotation, cap resolution.
    Returns the PIL image and a JPEG re-encode for Gemini."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return img, buf.getvalue()
