from __future__ import annotations

import io
import json
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image

from sift_viz.constants import HAS_FFMPEG, HAS_FFPROBE, VIDEO_EXTS
from sift_viz.models import VideoInfo

log = logging.getLogger("sift-viz")


def find_sift_binary() -> str | None:
    """Locate the sift binary by checking the build directory then PATH."""
    import shutil

    script_dir = Path(__file__).resolve().parent.parent
    for candidate in (
        script_dir.parent / "build" / "sift",
        script_dir / "build" / "sift",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("sift")


def system_open(path: str, reveal: bool = False) -> None:
    """Open *path* in the OS default application."""
    p = Path(path)
    system = platform.system()
    try:
        if system == "Darwin":
            args = ["open", "-R", path] if reveal else ["open", path]
            subprocess.Popen(args)
        elif system == "Windows":
            if reveal:
                subprocess.Popen(["explorer", "/select,", path])
            else:
                os.startfile(path)  # type: ignore[attr-defined]
        else:
            target = str(p.parent) if reveal else path
            subprocess.Popen(["xdg-open", target], stderr=subprocess.DEVNULL)
    except Exception as exc:
        log.warning("system_open failed for %s: %s", path, exc)


def cluster_palette(n: int) -> list[tuple[float, float, float, float]]:
    """Return *n* RGBA colours from the tab20 colourmap."""
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def load_media(path: str) -> Image.Image | None:
    """Load a media file as a PIL RGB image (first frame for videos)."""
    if Path(path).suffix.lower() in VIDEO_EXTS:
        return _extract_video_frame(path)
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        log.debug("load_media failed for %s: %s", path, exc)
        return None


def _extract_video_frame(path: str) -> Image.Image | None:
    """Extract the first frame of a video via ffmpeg."""
    if not HAS_FFMPEG:
        return None
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                "-i",
                path,
                "-ss",
                "0",
                "-vframes",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            capture_output=True,
            timeout=10,
        )
        if r.returncode != 0 or not r.stdout:
            return None
        return Image.open(io.BytesIO(r.stdout)).convert("RGB")
    except Exception as exc:
        log.debug("ffmpeg frame extract failed: %s", exc)
        return None


def probe_video_info(path: str) -> VideoInfo:
    """Return video metadata via ffprobe, or a zeroed VideoInfo on failure."""
    if not HAS_FFPROBE:
        return VideoInfo(width=0, height=0, fps=0.0, has_audio=False, duration=0.0)
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode != 0:
            return VideoInfo(width=0, height=0, fps=0.0, has_audio=False, duration=0.0)
        data = json.loads(r.stdout)
        streams = data.get("streams", [])
        duration = float(data.get("format", {}).get("duration") or 0)
        w = h = 0
        fps = 0.0
        has_audio = False
        for stream in streams:
            codec = stream.get("codec_type")
            if codec == "video" and not w:
                w, h = stream.get("width", 0), stream.get("height", 0)
                fps_str = stream.get("r_frame_rate", "25/1")
                num, den = (int(x) for x in fps_str.split("/"))
                fps = max(1.0, min(num / den if den else 25.0, 120.0))
            elif codec == "audio":
                has_audio = True
        return VideoInfo(width=w, height=h, fps=fps, has_audio=has_audio, duration=duration)
    except Exception as exc:
        log.debug("probe_video_info failed: %s", exc)
        return VideoInfo(width=0, height=0, fps=0.0, has_audio=False, duration=0.0)


def get_file_info(path: str) -> tuple[str, str]:
    """Return (type_line, meta_line) for the viewer panel."""
    p = Path(path)
    ext = p.suffix.upper().lstrip(".")
    size = _fmt_size(p.stat().st_size) if p.exists() else ""

    if p.suffix.lower() in VIDEO_EXTS:
        dims, dur = _video_meta(path)
        return (
            f"VIDEO  ·  {ext}",
            "  ·  ".join(x for x in [dur, dims, size] if x),
        )
    dims = _image_dims(path)
    return (
        f"IMAGE  ·  {ext}",
        "  ·  ".join(x for x in [dims, size] if x),
    )


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Resize *img* to fit within max_w × max_h, preserving aspect ratio."""
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def fmt_duration(seconds: float) -> str:
    """Format seconds as M:SS.mmm."""
    s = max(0.0, seconds)
    m = int(s) // 60
    sec = s - m * 60
    return f"{m}:{sec:06.3f}"


def fmt_hash_stats(n: int, total: int, elapsed: float) -> str:
    """Return a tqdm-style stats string for progress labels."""
    pct = int(n / total * 100) if total else 0
    speed = n / elapsed if elapsed > 0.2 else 0.0
    speed_str = f"{speed:.1f}/s" if speed > 0 else ""
    if n >= total:
        parts = ["100%", f"{n}/{total}", speed_str, f"done in {fmt_duration(elapsed)}"]
    else:
        eta_str = f"ETA {fmt_duration((total - n) / speed)}" if speed > 0 else "ETA --:--"
        parts = [f"{pct}%", f"{n}/{total}", speed_str, eta_str]
    return "  ·  ".join(p for p in parts if p)


def fmt_ago(seconds: float) -> str:
    """Format elapsed seconds as a human-readable 'ago' string."""
    m = int(seconds) // 60
    if m < 1:
        return "just now"
    if m < 60:
        return f"{m} min"
    h = m // 60
    return f"{h}h {m % 60}m" if h < 24 else f"{h // 24}d {h % 24}h"


def _video_meta(path: str) -> tuple[str, str]:
    """Return (dims, duration) strings for a video file via probe_video_info."""
    info = probe_video_info(path)
    dims = f"{info.width} × {info.height}" if info.is_valid else ""
    dur = f"{int(info.duration) // 60}:{int(info.duration) % 60:02d}" if info.duration > 0 else ""
    return dims, dur


def _image_dims(path: str) -> str:
    """Return 'W × H' for an image file, or '' on failure."""
    try:
        with Image.open(path) as img:
            return f"{img.width} × {img.height}"
    except Exception:
        return ""


def _fmt_size(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.0f} KB"
    return f"{n} B"


def log_stderr(stderr: str, prefix: str) -> None:
    """Emit non-empty, non-noise stderr lines as INFO log entries."""
    _IGNORED_PREFIXES = ("libpng warning:", "libpng error:")
    for line in stderr.splitlines():
        line = line.strip()
        if line and not any(line.startswith(p) for p in _IGNORED_PREFIXES):
            log.info("[sift %s] %s", prefix, line)


# kept for Any usage in caller context — matplotlib cmap returns Any
__all__: list[Any] = []
