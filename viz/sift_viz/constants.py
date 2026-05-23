from __future__ import annotations

import shutil

VIDEO_EXTS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".m4v",
        ".flv",
        ".wmv",
        ".mpg",
        ".mpeg",
        ".ts",
    }
)

IMAGE_EXTS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tga",
        ".gif",
        ".ppm",
        ".pgm",
        ".hdr",
        ".pic",
    }
)

POINT_HIT_RADIUS_PX: int = 12

BG_COLOR: str = "#1a1a1a"
PANEL_BG: str = "#242424"
TEXT_COLOR: str = "#cccccc"
DIM_COLOR: str = "#666666"
ACCENT_COLOR: str = "#4a9eff"
UNGROUPED_COLOR: str = "#606060"

VIEWER_W: int = 300
THUMB_W: int = 268
THUMB_H: int = 240

# Resolved once at import time so every call is O(1).
HAS_FFMPEG: bool = shutil.which("ffmpeg") is not None
HAS_FFPROBE: bool = shutil.which("ffprobe") is not None
HAS_FFPLAY: bool = shutil.which("ffplay") is not None
