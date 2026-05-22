#!/usr/bin/env python3
"""sift viz — interactive visualization frontend for the sift CLI tool.

Flow:
  1. Select folder + hash algo + projection method + dimensions → Run
     → calls `sift hash` then `sift project` (2D or 3D canvas)
  2. Adjust clustering algo + sliders → live re-cluster
     → calls `sift cluster`, recolors points by group
  3. Click any point → if it belongs to a group, open that group with ◀ ▶ nav;
     otherwise show that single file. The current point is always highlighted.
     (In 3D mode: drag to rotate, scroll to zoom)
"""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any


import matplotlib

matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import proj3d  # noqa: F401 (side-effects + used in _hit_test_3d)
from PIL import Image, ImageTk

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[sift-viz] %(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sift-viz")

# ── Constants ─────────────────────────────────────────────────────────────────

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

# Resolved once at import time so every call to has_ffmpeg/has_ffprobe is O(1).
HAS_FFMPEG: bool = shutil.which("ffmpeg") is not None
HAS_FFPROBE: bool = shutil.which("ffprobe") is not None
HAS_FFPLAY: bool = shutil.which("ffplay") is not None


# ── Tool / binary detection ───────────────────────────────────────────────────


def find_sift_binary() -> str | None:
    """Locate the sift binary by checking the build directory then PATH.

    Looks for ``<repo_root>/build/sift`` first (the default CMake output
    location), then falls back to whatever ``shutil.which`` finds on PATH.

    Returns:
        Absolute path string if a usable binary is found, otherwise ``None``.
    """
    script_dir = Path(__file__).resolve().parent
    for candidate in (
        script_dir.parent / "build" / "sift",
        script_dir / "build" / "sift",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("sift")


# ── System integration ────────────────────────────────────────────────────────


def system_open(path: str, reveal: bool = False) -> None:
    """Open *path* in the OS default application.

    Args:
        path: Absolute path to the file or directory to open.
        reveal: When ``True``, open the file's *parent folder* and select the
            file rather than launching it (macOS / Windows only; on Linux the
            parent directory is opened instead).
    """
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
                os.startfile(path)
        else:
            target = str(p.parent) if reveal else path
            subprocess.Popen(["xdg-open", target], stderr=subprocess.DEVNULL)
    except Exception as exc:
        log.warning("system_open failed for %s: %s", path, exc)


# ── Colors ────────────────────────────────────────────────────────────────────


def cluster_palette(n: int) -> list[Any]:
    """Return a list of *n* RGBA colours from the ``tab20`` colourmap.

    Args:
        n: Number of colours to generate.

    Returns:
        List of RGBA tuples, cycling through the 20-colour map if *n* > 20.
    """
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


# ── Media loading ─────────────────────────────────────────────────────────────


def load_media(path: str) -> Image.Image | None:
    """Load a media file as a PIL RGB image.

    For video files the first frame is extracted via ffmpeg.  Returns ``None``
    if the file cannot be decoded or ffmpeg is unavailable.

    Args:
        path: Path to an image or video file.

    Returns:
        RGB ``PIL.Image`` on success, or ``None`` on failure.
    """
    if Path(path).suffix.lower() in VIDEO_EXTS:
        return _extract_video_frame(path)
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        log.debug("load_media failed for %s: %s", path, exc)
        return None


def _extract_video_frame(path: str) -> Image.Image | None:
    """Extract the first frame of a video file using ffmpeg.

    Args:
        path: Path to a video file.

    Returns:
        RGB ``PIL.Image`` of the first frame, or ``None`` if extraction fails.
    """
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


def get_file_info(path: str) -> tuple[str, str]:
    """Return human-readable type and metadata strings for the viewer panel.

    Args:
        path: Path to an image or video file.

    Returns:
        A ``(type_line, meta_line)`` tuple, e.g.
        ``("IMAGE · PNG", "1920 × 1080  ·  2.4 MB")``.
    """
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


def _fmt_size(n: int) -> str:
    """Format a byte count as a human-readable string (e.g. ``"2.4 MB"``).

    Args:
        n: File size in bytes.

    Returns:
        Formatted string using MB / KB / B depending on magnitude.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.0f} KB"
    return f"{n} B"


def _image_dims(path: str) -> str:
    """Return ``"W × H"`` for an image file, or an empty string on failure.

    Args:
        path: Path to an image file.

    Returns:
        Dimension string like ``"1920 × 1080"``, or ``""`` if unreadable.
    """
    try:
        with Image.open(path) as img:
            return f"{img.width} × {img.height}"
    except Exception:
        return ""


def _video_meta(path: str) -> tuple[str, str]:
    """Query video dimensions and duration via ffprobe.

    Args:
        path: Path to a video file.

    Returns:
        ``(dims, duration)`` where *dims* is ``"W × H"`` and *duration* is
        ``"M:SS"``.  Either field may be an empty string if unavailable.
    """
    if not HAS_FFPROBE:
        return "", ""
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
            return "", ""
        data = json.loads(r.stdout)
        fmt_dur = float(data.get("format", {}).get("duration", 0))
        dur = f"{int(fmt_dur) // 60}:{int(fmt_dur) % 60:02d}" if fmt_dur > 0 else ""
        dims = ""
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w, h = stream.get("width", 0), stream.get("height", 0)
                if w and h:
                    dims = f"{w} × {h}"
                break
        return dims, dur
    except Exception as exc:
        log.debug("ffprobe failed: %s", exc)
        return "", ""


def _probe_video_info(path: str) -> tuple[int, int, float, bool, float]:
    """Return ``(width, height, fps, has_audio, duration)`` for a video file via ffprobe.

    Args:
        path: Path to a video file.

    Returns:
        ``(width, height, fps, has_audio, duration)`` on success, or
        ``(0, 0, 0.0, False, 0.0)`` on failure.
    """
    if not HAS_FFPROBE:
        return 0, 0, 0.0, False, 0.0
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-show_format", path,
            ],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode != 0:
            return 0, 0, 0.0, False, 0.0
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
        return w, h, fps, has_audio, duration
    except Exception as exc:
        log.debug("_probe_video_info failed: %s", exc)
        return 0, 0, 0.0, False, 0.0


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Resize *img* in-place to fit within *max_w* × *max_h*, preserving aspect ratio.

    Args:
        img: PIL image to resize (modified in-place via ``thumbnail``).
        max_w: Maximum output width in pixels.
        max_h: Maximum output height in pixels.

    Returns:
        The same image object after thumbnailing.
    """
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``"M:SS"``.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string, e.g. ``"1:07"`` or ``"0:00"``.
    """
    s = int(max(0.0, seconds))
    return f"{s // 60}:{s % 60:02d}"


# ── State dataclasses ─────────────────────────────────────────────────────────


@dataclass
class ViewerState:
    """Tracks which file(s) are currently loaded in the right-hand viewer panel.

    Attributes:
        files: Ordered list of absolute file paths available for viewing.
        index: Currently displayed index into *files*.
        group_id: Cluster group index when viewing a multi-file group, or
            ``None`` for a single-point selection.
    """

    files: list[str] = field(default_factory=list)
    index: int = 0
    group_id: int | None = None

    @property
    def current_path(self) -> str | None:
        """Return the path of the currently displayed file, or ``None``."""
        return self.files[self.index] if self.files else None

    @property
    def is_multi(self) -> bool:
        """Return ``True`` when more than one file is loaded (group selection)."""
        return len(self.files) > 1


@dataclass
class SelectionState:
    """Tracks which scatter point or cluster hull is currently selected.

    Attributes:
        point_idx: Index into the projection point array for a single selection.
        group_idx: Cluster group index for a hull selection.
        member_indices: Point-array indices belonging to the selected group.
    """

    point_idx: int | None = None
    group_idx: int | None = None
    member_indices: list[int] = field(default_factory=list)

    def clear(self) -> None:
        """Reset all selection fields to their empty defaults."""
        self.point_idx = None
        self.group_idx = None
        self.member_indices = []


# ── Logging helpers ───────────────────────────────────────────────────────────


def _log_stderr(stderr: str, prefix: str) -> None:
    """Emit each non-empty, non-noise line of *stderr* as an INFO log entry.

    Args:
        stderr: Raw stderr text from a sift subprocess.
        prefix: Short label prepended to each line (e.g. ``"hash"``).
    """
    _IGNORED_PREFIXES = ("libpng warning:", "libpng error:")
    for line in stderr.splitlines():
        line = line.strip()
        if line and not any(line.startswith(p) for p in _IGNORED_PREFIXES):
            log.info("[sift %s] %s", prefix, line)


# ── Inline video player ───────────────────────────────────────────────────────


class VideoPlayer:
    """Renders ffmpeg-decoded video frames onto a tkinter Canvas with audio.

    The video ffmpeg process pipes raw RGB frames into a queue consumed by
    the main-thread tick loop.  Audio is played via a separate ``ffplay``
    process which handles device output independently — this avoids the
    ALSA/PortAudio crashes that come from driving a sounddevice stream from
    within a Python thread.

    Frame timing uses elapsed wall-clock time so the video self-corrects for
    drift (drops frames when behind, waits when ahead).  A/V sync is wall-
    clock based: both ffmpeg and ffplay are started from the same seek offset
    at nearly the same instant, keeping them within ~50 ms in practice.

    Supports pause/resume (freezes at the current frame position, restarts
    from there) and seek (restarts both processes from an arbitrary timestamp).
    """

    def __init__(self, canvas: tk.Canvas, on_end: Any) -> None:
        """Args:
            canvas: The tkinter Canvas to draw frames onto.
            on_end: Zero-argument callable invoked when playback finishes
                naturally (end of stream).  Not called when ``stop()`` is
                used directly.
        """
        self._canvas = canvas
        self._on_end = on_end

        # Persistent across seeks (set once on play())
        self._path = ""
        self._duration = 0.0
        self._has_audio = False
        self._frame_w = 0
        self._frame_h = 0
        self._fps = 25.0
        self._position_cb: Any = None  # callable(float) or None

        # Per-segment state (reset on each _start_at call)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._frame_q: queue.Queue[bytes | None] = queue.Queue(maxsize=16)
        self._frames_shown = 0
        self._seek_pos = 0.0      # start offset of current ffmpeg segment
        self._play_start = 0.0   # monotonic time when this segment began

        # Audio (optional ffplay subprocess)
        self._audio_proc: subprocess.Popen | None = None

        # Playback state
        self._active = False      # ffmpeg running and ticking
        self._paused = False      # stopped at a known position, ready to resume
        self._pause_pos = 0.0

        self._after_id: str | None = None
        self._photo: Any = None   # ImageTk.PhotoImage — held to prevent GC

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        """``True`` while a video is actively playing (not paused)."""
        return self._active

    @property
    def is_paused(self) -> bool:
        """``True`` while playback is paused at a known position."""
        return self._paused

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        if self._paused:
            return self._pause_pos
        if not self._active:
            return 0.0
        return self._seek_pos + (time.monotonic() - self._play_start)

    @property
    def duration(self) -> float:
        """Total video duration in seconds (0.0 if unknown)."""
        return self._duration

    def play(self, path: str, seek: float = 0.0, position_cb: Any = None) -> bool:
        """Start playback of *path* on the canvas.

        Stops any in-progress playback first.  Returns ``False`` if the video
        cannot be probed (ffprobe unavailable or unreadable file).

        Args:
            path: Absolute path to a video file.
            seek: Start position in seconds (default 0).
            position_cb: Optional callable ``(float)`` invoked each frame tick
                with the current playback position in seconds.

        Returns:
            ``True`` if playback started, ``False`` on failure.
        """
        self.stop()

        vid_w, vid_h, fps, has_audio, duration = _probe_video_info(path)
        if not vid_w or not vid_h:
            return False

        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)
        scale = min((cw - 8) / vid_w, (ch - 8) / vid_h)
        self._frame_w = max(1, int(vid_w * scale))
        self._frame_h = max(1, int(vid_h * scale))
        self._fps = fps
        self._path = path
        self._duration = duration
        self._has_audio = has_audio
        self._position_cb = position_cb

        self._start_at(seek)
        return True

    def pause(self) -> None:
        """Pause playback at the current position."""
        if not self._active:
            return
        self._pause_pos = self.position
        self._active = False
        self._paused = True
        self._cancel_tick()
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._drain_queue()
        self._teardown_audio()

    def resume(self) -> None:
        """Resume playback from the paused position."""
        if not self._paused:
            return
        self._paused = False
        self._start_at(self._pause_pos)

    def seek(self, t: float) -> None:
        """Seek to *t* seconds and resume playback.

        Works whether currently playing or paused.
        """
        if not self._path:
            return
        self._active = False
        self._paused = False
        self._cancel_tick()
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._drain_queue()
        self._teardown_audio()
        self._start_at(t)

    def stop(self) -> None:
        """Immediately halt playback and release all resources."""
        self._active = False
        self._paused = False
        self._cancel_tick()
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._drain_queue()
        self._teardown_audio()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _start_at(self, t: float) -> None:
        """Start (or restart) ffmpeg and ffplay from position *t* seconds."""
        if self._duration > 0:
            t = max(0.0, min(t, self._duration))
        else:
            t = max(0.0, t)
        self._seek_pos = t
        self._frames_shown = 0
        self._active = True

        seek_args = ["-ss", f"{t:.3f}"] if t > 0 else []

        # ── Video process ─────────────────────────────────────────────────────
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "quiet",
                *seek_args,
                "-i", self._path,
                "-vf", f"scale={self._frame_w}:{self._frame_h}:flags=fast_bilinear",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._read_frames, daemon=True)
        self._thread.start()

        # ── Audio process via ffplay (no sounddevice / PortAudio involved) ────
        if self._has_audio and HAS_FFPLAY:
            try:
                self._audio_proc = subprocess.Popen(
                    [
                        "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                        *seek_args,
                        self._path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                log.debug("ffplay audio failed, continuing without audio: %s", exc)
                self._audio_proc = None

        self._play_start = time.monotonic()
        self._tick()

    def _cancel_tick(self) -> None:
        """Cancel any pending after() tick callback."""
        if self._after_id is not None:
            try:
                self._canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _drain_queue(self) -> None:
        """Discard all pending frames so the reader thread can exit cleanly."""
        while True:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

    def _teardown_audio(self) -> None:
        """Terminate the ffplay audio process if running."""
        if self._audio_proc is not None:
            self._audio_proc.terminate()
            self._audio_proc = None

    def _read_frames(self) -> None:
        """Reader thread: pulls raw RGB frames from ffmpeg stdout into the queue."""
        frame_bytes = self._frame_w * self._frame_h * 3
        proc = self._proc
        while self._active and proc and proc.stdout:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                try:
                    self._frame_q.put(None, timeout=2.0)
                except queue.Full:
                    pass
                break
            try:
                self._frame_q.put(raw, timeout=2.0)
            except queue.Full:
                pass  # drop frame — display is lagging

    def _tick(self) -> None:
        """Main-thread timer: dequeue the next due frame and reschedule."""
        if not self._active:
            return

        elapsed = time.monotonic() - self._play_start
        target = int(elapsed * self._fps)

        # Drop frames that we're behind on (video fell behind clock)
        while self._frames_shown < target:
            try:
                stale = self._frame_q.get_nowait()
            except queue.Empty:
                break
            if stale is None:
                self._active = False
                self._on_end()
                return
            self._frames_shown += 1

        # Fetch the next frame to display
        try:
            raw = self._frame_q.get_nowait()
        except queue.Empty:
            self._after_id = self._canvas.after(5, self._tick)
            return

        if raw is None:
            self._active = False
            self._on_end()
            return

        img = Image.frombytes("RGB", (self._frame_w, self._frame_h), raw)
        photo = ImageTk.PhotoImage(img)
        self._photo = photo

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        self._canvas.delete("all")
        self._canvas.create_image(
            max(0, (cw - self._frame_w) // 2),
            max(0, (ch - self._frame_h) // 2),
            anchor="nw", image=photo,
        )
        self._frames_shown += 1

        # Report position to the UI callback (main thread — safe to update widgets)
        if self._position_cb is not None:
            pos = self._seek_pos + self._frames_shown / self._fps
            self._position_cb(pos)

        # Schedule next tick exactly when the next frame is due
        next_due = self._frames_shown / self._fps
        delay = max(1, int((next_due - (time.monotonic() - self._play_start)) * 1000))
        self._after_id = self._canvas.after(delay, self._tick)


# ── Main application ──────────────────────────────────────────────────────────


class SiftViz(tk.Tk):
    """Main application window for the sift interactive visualizer.

    Arranges three resizable panels (controls | scatter plot | media viewer)
    inside a ``tk.PanedWindow``.  Background threads communicate back to the
    UI via a ``queue.Queue`` polled every 50 ms with ``after()``.

    The scatter plot supports both 2D and 3D projection modes.  In 3D mode
    matplotlib's built-in ``Axes3D`` provides free mouse-drag rotation and
    scroll-to-zoom; click-to-select uses screen-space projection to find the
    nearest point.
    """

    def __init__(self) -> None:
        """Initialize the application, locate the sift binary, and build the UI."""
        super().__init__()
        self.title("sift viz")
        self.minsize(1100, 620)
        self.configure(bg=BG_COLOR)

        self.binary = find_sift_binary()
        if self.binary:
            log.info("sift binary: %s", self.binary)
        else:
            log.warning("sift binary not found — build the project first")

        self.tmpdir = tempfile.mkdtemp(prefix="sift_viz_")
        self._hashes_json = os.path.join(self.tmpdir, "hashes.json")
        self._proj_2d_json = os.path.join(self.tmpdir, "projection_2d.json")
        self._proj_3d_json = os.path.join(self.tmpdir, "projection_3d.json")
        self._clusters_json = os.path.join(self.tmpdir, "clusters.json")

        self._q: queue.Queue[dict] = queue.Queue()
        self._projection_2d: dict | None = None
        self._projection_3d: dict | None = None
        self._cluster_data: dict | None = None
        self._running = False
        self._cancel_requested = False
        self._current_proc: subprocess.Popen | None = None
        self._cluster_timer: str | None = None
        self._poll_id: str | None = None

        # Hit-test state rebuilt on every _redraw
        self._point_positions: np.ndarray | None = None  # (N,2) or (N,3)
        self._point_files: list[str] = []
        self._point_group: dict[int, int] = {}  # point_idx → group_idx
        self._overlay_artists: list[Artist] = []

        self._viewer = ViewerState()
        self._selection = SelectionState()
        self._current_img: Image.Image | None = None
        self._video_player: VideoPlayer | None = None  # created in _build_viewer_panel

        # tk vars
        self.folder_var = tk.StringVar()
        self.algo_var = tk.StringVar(value="dhash")
        self.hash_size_var = tk.IntVar(value=8)
        self.media_var = tk.StringVar(value="images")
        self.frames_var = tk.IntVar(value=8)
        self.proj_var = tk.StringVar(value="pca")
        self.dims_var = tk.StringVar(value="2")
        self.perplexity_var = tk.IntVar(value=30)
        self.iterations_var = tk.IntVar(value=1000)
        self.cluster_method_var = tk.StringVar(value="threshold")
        self.threshold_var = tk.IntVar(value=10)
        self.cut_height_var = tk.IntVar(value=10)
        self.linkage_var = tk.StringVar(value="complete")
        self.min_group_var = tk.IntVar(value=3)
        self.min_filter_var = tk.IntVar(value=2)

        self._build_ui()
        self._update_tsne_visibility()
        self._update_frames_visibility()
        self._rebuild_cluster_params()
        self._poll()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        if not self.binary:
            self._set_status(
                "sift binary not found — build with: cmake --build build/",
                error=True,
            )
            self.run_btn.config(state="disabled")

    def _on_close(self) -> None:
        """Cancel the poll loop, close matplotlib, clean up temp files, and exit."""
        log.info("shutting down")
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        player = self._video_player
        if player and (player.is_playing or player.is_paused):
            player.stop()
        plt.close("all")
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.destroy()
        sys.exit(0)

    def _is_3d(self) -> bool:
        """Return ``True`` when the 3D projection mode is selected."""
        return self.dims_var.get() == "3"

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Create the root PanedWindow and delegate each panel to its builder."""
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        pw = tk.PanedWindow(
            self, orient="horizontal", sashwidth=5, sashrelief="flat", bg="#333", bd=0
        )
        pw.grid(row=0, column=0, sticky="nsew")

        left = tk.Frame(pw, bg=PANEL_BG)
        center = tk.Frame(pw, bg=BG_COLOR)
        right = tk.Frame(pw, bg=PANEL_BG)

        pw.add(left, minsize=200, width=270, stretch="never")
        pw.add(center, minsize=400, stretch="always")
        pw.add(right, minsize=220, width=VIEWER_W, stretch="never")

        self._build_left_panel(left)
        self._build_scatter_panel(center)
        self._build_viewer_panel(right)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self, ctrl: tk.Frame) -> None:
        """Populate the left control panel with all input, hash, and cluster widgets.

        Args:
            ctrl: The parent ``tk.Frame`` for the left panel.
        """
        ctrl.columnconfigure(0, weight=1)
        r = 0

        # Folder picker
        self._section_label(ctrl, r, "INPUT")
        r += 1
        ff = tk.Frame(ctrl, bg=PANEL_BG)
        ff.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 8))
        r += 1
        ff.columnconfigure(0, weight=1)
        tk.Entry(
            ff,
            textvariable=self.folder_var,
            bg="#2e2e2e",
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief="flat",
            bd=4,
        ).grid(row=0, column=0, sticky="ew")
        self._flat_btn(ff, "Browse", self._browse).grid(row=0, column=1, padx=(4, 0))

        # Hash algo
        self._section_label(ctrl, r, "HASH ALGORITHM")
        r += 1
        for algo in ("dhash", "phash", "whash"):
            self._radio(ctrl, r, algo, self.algo_var, algo)
            r += 1

        # Hash size
        self._section_label(ctrl, r, "HASH SIZE")
        r += 1
        self.size_lbl = self._slider(
            ctrl,
            r,
            self.hash_size_var,
            4,
            32,
            label_fn=lambda v: f"{v}×{v} · {v * v} bits",
        )
        r += 1

        # Media type
        self._section_label(ctrl, r, "MEDIA TYPE")
        r += 1
        for value, label in (("images", "Images only"), ("videos", "Videos only"), ("all", "Both")):
            tk.Radiobutton(
                ctrl,
                text=label,
                variable=self.media_var,
                value=value,
                bg=PANEL_BG,
                fg=TEXT_COLOR,
                selectcolor=PANEL_BG,
                activebackground=PANEL_BG,
                activeforeground=TEXT_COLOR,
                font=("", 9),
                anchor="w",
                command=self._on_media_type_change,
            ).grid(row=r, column=0, sticky="w", padx=12)
            r += 1

        self.frames_frame = tk.Frame(ctrl, bg=PANEL_BG)
        self.frames_frame.grid(row=r, column=0, sticky="ew")
        r += 1
        self.frames_frame.columnconfigure(0, weight=1)
        tk.Label(
            self.frames_frame, text="Frames per video", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(4, 0))
        self._slider(
            self.frames_frame,
            1,
            self.frames_var,
            1,
            32,
            label_fn=lambda v: f"{v} frame{'s' if v != 1 else ''}",
        )
        self._update_frames_visibility()

        # Projection method
        self._section_label(ctrl, r, "PROJECTION")
        r += 1
        for proj, label in (("pca", "PCA"), ("tsne", "t-SNE")):
            self._radio(
                ctrl,
                r,
                label,
                self.proj_var,
                proj,
                command=self._update_tsne_visibility,
            )
            r += 1

        self.tsne_frame = tk.Frame(ctrl, bg=PANEL_BG)
        self.tsne_frame.grid(row=r, column=0, sticky="ew")
        r += 1
        self.tsne_frame.columnconfigure(0, weight=1)
        self._labeled_slider(self.tsne_frame, 0, "Perplexity", self.perplexity_var, 5, 100)
        self._labeled_slider(self.tsne_frame, 2, "Iterations", self.iterations_var, 250, 3000)

        self._divider(ctrl, r)
        r += 1

        # Run button + progress + status
        self.run_btn = tk.Button(
            ctrl,
            text="Run",
            command=self._on_run,
            bg=ACCENT_COLOR,
            fg="white",
            relief="flat",
            font=("", 10, "bold"),
            pady=6,
            cursor="hand2",
            activebackground="#3a8ee6",
            activeforeground="white",
        )
        self.run_btn.grid(row=r, column=0, sticky="ew", padx=12, pady=(6, 4))
        r += 1

        self.progress = ttk.Progressbar(ctrl, mode="determinate", maximum=100, value=0, length=100)
        self.progress.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 4))
        r += 1

        self.status_lbl = tk.Label(
            ctrl,
            text="Ready.",
            bg=PANEL_BG,
            fg=DIM_COLOR,
            font=("", 8),
            wraplength=240,
            justify="left",
        )
        self.status_lbl.grid(row=r, column=0, sticky="w", padx=12, pady=(0, 6))
        r += 1

        # Clustering
        self._divider(ctrl, r)
        r += 1
        self._section_label(ctrl, r, "CLUSTERING")
        r += 1

        self.cluster_progress = ttk.Progressbar(
            ctrl, mode="determinate", maximum=100, value=0, length=100
        )
        self.cluster_progress.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 6))
        r += 1

        for method in ("threshold", "hierarchical", "hdbscan"):
            self._radio(
                ctrl,
                r,
                method,
                self.cluster_method_var,
                method,
                command=self._on_method_change,
            )
            r += 1

        self.cluster_params_frame = tk.Frame(ctrl, bg=PANEL_BG)
        self.cluster_params_frame.grid(row=r, column=0, sticky="ew")
        r += 1
        self.cluster_params_frame.columnconfigure(0, weight=1)

        self._section_label(ctrl, r, "MIN GROUP SIZE")
        r += 1
        self._slider(
            ctrl,
            r,
            self.min_filter_var,
            2,
            20,
            on_change=self._on_slider,
            label_fn=lambda v: str(v),
        )
        r += 1

        # Dimensions toggle — visualization only, no re-run needed
        self._divider(ctrl, r)
        r += 1
        self._section_label(ctrl, r, "VIEW DIMENSIONS")
        r += 1
        dims_row = tk.Frame(ctrl, bg=PANEL_BG)
        dims_row.grid(row=r, column=0, sticky="w", padx=12, pady=(0, 8))
        r += 1
        for label, value in (("2D", "2"), ("3D", "3")):
            tk.Radiobutton(
                dims_row,
                text=label,
                variable=self.dims_var,
                value=value,
                bg=PANEL_BG,
                fg=TEXT_COLOR,
                selectcolor=PANEL_BG,
                activebackground=PANEL_BG,
                activeforeground=TEXT_COLOR,
                font=("", 9),
                command=self._on_dims_change,
            ).pack(side="left", padx=(0, 12))

    # ── Scatter panel ─────────────────────────────────────────────────────────

    def _build_scatter_panel(self, pf: tk.Frame) -> None:
        """Create the matplotlib scatter canvas in the centre panel.

        Args:
            pf: The parent ``tk.Frame`` for the scatter panel.
        """
        pf.columnconfigure(0, weight=1)
        pf.rowconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(7, 6))
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax = self.fig.add_subplot(111)  # replaced by _rebuild_axes on first run
        self._style_axes_2d()

        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_mpl_click)
        self.canvas.draw()

    # ── Viewer panel ──────────────────────────────────────────────────────────

    def _build_viewer_panel(self, v: tk.Frame) -> None:
        """Build the right-hand media viewer with image canvas and action buttons.

        Layout (top → bottom):
          - Media canvas (fills available height)
          - File name / type / meta / cluster labels
          - Seek bar with time labels (hidden when not playing/paused)
          - Playback controls: ⏮ skip-back | ▶/⏸ play-pause | ⏭ skip-forward
          - Navigation + actions: ◀ group | N/M | ↗ Open | ⌂ Folder | ▶ group

        Args:
            v: The parent ``tk.Frame`` for the viewer panel.
        """
        v.columnconfigure(0, weight=1)
        v.rowconfigure(1, weight=1)

        self._section_label(v, 0, "MEDIA VIEWER")

        self.img_canvas = tk.Canvas(v, bg="#111111", highlightthickness=0)
        self.img_canvas.grid(row=1, column=0, padx=12, pady=(4, 0), sticky="nsew")
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)
        self._draw_placeholder()
        self._video_player = VideoPlayer(self.img_canvas, self._on_video_end)

        self.viewer_name_lbl = self._viewer_label(v, 2, font=("", 9, "bold"), fg=TEXT_COLOR)
        self.viewer_type_lbl = self._viewer_label(v, 3)
        self.viewer_meta_lbl = self._viewer_label(v, 4)
        self.viewer_cluster_lbl = self._viewer_label(
            v, 5, text="Click a point or cluster\nto preview media", fg="#444"
        )

        # ── Seek / scrub bar (shown only during video playback / pause) ───────
        self.seek_frame = tk.Frame(v, bg=PANEL_BG)
        self.seek_frame.columnconfigure(1, weight=1)
        self.seek_frame.grid(row=6, column=0, sticky="ew", padx=12, pady=(6, 0))
        self.seek_frame.grid_remove()  # hidden until playback starts

        self.seek_pos_lbl = tk.Label(
            self.seek_frame, text="0:00", bg=PANEL_BG, fg=DIM_COLOR, font=("", 7), width=5, anchor="w"
        )
        self.seek_pos_lbl.grid(row=0, column=0, sticky="w")

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = tk.Scale(
            self.seek_frame,
            from_=0.0,
            to=100.0,
            orient="horizontal",
            variable=self.seek_var,
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            troughcolor="#333",
            highlightthickness=0,
            showvalue=False,
            command=self._on_seek_drag,
        )
        self.seek_scale.grid(row=0, column=1, sticky="ew")
        self.seek_dur_lbl = tk.Label(
            self.seek_frame, text="0:00", bg=PANEL_BG, fg=DIM_COLOR, font=("", 7), width=5, anchor="e"
        )
        self.seek_dur_lbl.grid(row=0, column=2, sticky="e")

        self._user_scrubbing = False
        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)

        # ── Playback controls ─────────────────────────────────────────────────
        pb = tk.Frame(v, bg=PANEL_BG)
        pb.grid(row=7, column=0, pady=(6, 0))
        self.skip_back_btn = self._icon_btn(pb, "⏮", self._skip_back, col=0)
        self.play_pause_btn = self._action_btn(pb, "▶  Play", self._toggle_play_pause, col=1)
        self.skip_fwd_btn = self._icon_btn(pb, "⏭", self._skip_forward, col=2)
        for b in (self.skip_back_btn, self.play_pause_btn, self.skip_fwd_btn):
            b.config(state="disabled")

        # ── Group navigation + file actions (one row) ─────────────────────────
        act = tk.Frame(v, bg=PANEL_BG)
        act.grid(row=8, column=0, pady=(8, 0))
        self.prev_btn = self._icon_btn(act, "◀", self._viewer_prev, col=0)
        self.nav_lbl = tk.Label(act, text="", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8), width=5)
        self.nav_lbl.grid(row=0, column=1, padx=2)
        self.open_btn = self._action_btn(act, "↗  Open", self._open_media, col=2)
        self.folder_btn = self._action_btn(act, "⌂  Folder", self._open_in_folder, col=3)
        self.next_btn = self._icon_btn(act, "▶", self._viewer_next, col=4)
        for b in (self.prev_btn, self.next_btn, self.open_btn, self.folder_btn):
            b.config(state="disabled")

        self.viewer_path_lbl = self._viewer_label(v, 9, fg="#383838", font=("", 7), pady=(6, 12))

    # ── Widget factory helpers ────────────────────────────────────────────────

    def _section_label(self, parent: tk.Widget, row: int, text: str) -> None:
        """Add a small uppercase section heading label to *parent* at *row*.

        Args:
            parent: Container widget.
            row: Grid row index.
            text: Heading text (displayed uppercase, dim colour).
        """
        tk.Label(parent, text=text, bg=PANEL_BG, fg=DIM_COLOR, font=("", 8, "bold")).grid(
            row=row, column=0, sticky="w", padx=12, pady=(14, 2)
        )

    def _radio(
        self,
        parent: tk.Widget,
        row: int,
        text: str,
        var: tk.Variable,
        value: str,
        command: Any = None,
    ) -> None:
        """Add a dark-themed radiobutton to *parent*.

        Args:
            parent: Container widget.
            row: Grid row index.
            text: Button label.
            var: Shared ``tk.StringVar`` for the radio group.
            value: Value assigned to *var* when this button is selected.
            command: Optional callback invoked on selection.
        """
        tk.Radiobutton(
            parent,
            text=text,
            variable=var,
            value=value,
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            selectcolor=PANEL_BG,
            activebackground=PANEL_BG,
            activeforeground=TEXT_COLOR,
            font=("", 9),
            anchor="w",
            command=command,
        ).grid(row=row, column=0, sticky="w", padx=12)

    def _divider(self, parent: tk.Widget, row: int) -> None:
        """Insert a 1-pixel horizontal divider line.

        Args:
            parent: Container widget.
            row: Grid row index.
        """
        tk.Frame(parent, bg="#333", height=1).grid(
            row=row, column=0, sticky="ew", padx=12, pady=(10, 0)
        )

    def _flat_btn(self, parent: tk.Widget, text: str, command: Any) -> tk.Button:
        """Create and return a flat dark-themed button (not yet gridded).

        Args:
            parent: Container widget.
            text: Button label.
            command: Click callback.

        Returns:
            Configured ``tk.Button`` instance.
        """
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#3a3a3a",
            fg=TEXT_COLOR,
            relief="flat",
            padx=6,
            activebackground="#484848",
            activeforeground=TEXT_COLOR,
            cursor="hand2",
        )

    def _icon_btn(self, parent: tk.Widget, text: str, command: Any, col: int) -> tk.Button:
        """Create, grid, and return a small icon/navigation button.

        Args:
            parent: Container widget.
            text: Button symbol (e.g. ``"◀"``).
            command: Click callback.
            col: Grid column index within *parent*.

        Returns:
            Configured and gridded ``tk.Button`` instance.
        """
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            width=3,
            bg="#2e2e2e",
            fg=TEXT_COLOR,
            relief="flat",
            font=("", 11),
            activebackground="#3a3a3a",
            activeforeground=TEXT_COLOR,
            cursor="hand2",
        )
        btn.grid(row=0, column=col, padx=4)
        return btn

    def _action_btn(self, parent: tk.Widget, text: str, command: Any, col: int) -> tk.Button:
        """Create, grid, and return a viewer action button (Play / Open / Folder).

        Args:
            parent: Container widget.
            text: Button label.
            command: Click callback.
            col: Grid column index within *parent*.

        Returns:
            Configured and gridded ``tk.Button`` instance.
        """
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg="#2e2e2e",
            fg=TEXT_COLOR,
            relief="flat",
            font=("", 8),
            padx=6,
            pady=3,
            activebackground="#3a3a3a",
            activeforeground=TEXT_COLOR,
            cursor="hand2",
        )
        btn.grid(row=0, column=col, padx=3)
        return btn

    def _viewer_label(
        self,
        parent: tk.Widget,
        row: int,
        *,
        text: str = "",
        fg: str = DIM_COLOR,
        font: tuple = ("", 8),
        pady: tuple | int = (1, 0),
    ) -> tk.Label:
        """Create, grid, and return a centred label for the viewer info area.

        Args:
            parent: Container widget.
            row: Grid row index.
            text: Initial label text.
            fg: Foreground colour string.
            font: Tkinter font tuple.
            pady: Vertical padding.

        Returns:
            Configured and gridded ``tk.Label`` instance.
        """
        lbl = tk.Label(
            parent,
            text=text,
            bg=PANEL_BG,
            fg=fg,
            font=font,
            wraplength=VIEWER_W - 24,
            justify="center",
        )
        lbl.grid(row=row, column=0, padx=12, pady=pady, sticky="ew")
        return lbl

    def _slider(
        self,
        parent: tk.Widget,
        row: int,
        var: tk.IntVar,
        lo: int,
        hi: int,
        on_change: Any = None,
        label_fn: Any = None,
    ) -> tk.Label:
        """Add a horizontal slider with an adjacent value label.

        Args:
            parent: Container widget.
            row: Grid row index for the slider frame.
            var: ``tk.IntVar`` bound to the slider.
            lo: Minimum slider value.
            hi: Maximum slider value.
            on_change: Optional callback invoked whenever the value changes.
            label_fn: Optional callable ``(int) -> str`` to format the value
                label; defaults to ``str``.

        Returns:
            The value ``tk.Label`` widget.
        """
        f = tk.Frame(parent, bg=PANEL_BG)
        f.grid(row=row, column=0, sticky="ew", padx=12, pady=(2, 8))
        f.columnconfigure(0, weight=1)

        def _cb(_: Any = None) -> None:
            v = int(var.get())
            lbl.config(text=label_fn(v) if label_fn else str(v))
            if on_change:
                on_change()

        tk.Scale(
            f,
            from_=lo,
            to=hi,
            orient="horizontal",
            variable=var,
            command=_cb,
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            troughcolor="#333",
            highlightthickness=0,
            showvalue=False,
        ).grid(row=0, column=0, sticky="ew")
        initial = int(var.get())
        lbl = tk.Label(
            f,
            text=label_fn(initial) if label_fn else str(initial),
            bg=PANEL_BG,
            fg=DIM_COLOR,
            font=("", 8),
            width=19,
            anchor="w",
        )
        lbl.grid(row=0, column=1, padx=(10, 0))
        return lbl

    def _labeled_slider(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        var: tk.IntVar,
        lo: int,
        hi: int,
        on_change: Any = None,
    ) -> None:
        """Add a text label followed by a slider on the next row.

        Args:
            parent: Container widget.
            row: Grid row for the label; slider goes on ``row + 1``.
            label: Text to display above the slider.
            var: ``tk.IntVar`` bound to the slider.
            lo: Minimum slider value.
            hi: Maximum slider value.
            on_change: Optional callback passed through to ``_slider``.
        """
        tk.Label(parent, text=label, bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)).grid(
            row=row, column=0, sticky="w", padx=12, pady=(4, 0)
        )
        self._slider(parent, row + 1, var, lo, hi, on_change=on_change)

    def _cluster_slider(
        self, parent: tk.Widget, row: int, label: str, var: tk.IntVar, lo: int, hi: int
    ) -> None:
        """Add a labelled slider that triggers ``_on_slider`` on every change.

        Args:
            parent: Container widget.
            row: Grid row for the label; slider goes on ``row + 1``.
            label: Text to display above the slider.
            var: ``tk.IntVar`` bound to the slider.
            lo: Minimum slider value.
            hi: Maximum slider value.
        """
        tk.Label(parent, text=label, bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)).grid(
            row=row, column=0, sticky="w", padx=12, pady=(6, 0)
        )
        self._slider(parent, row + 1, var, lo, hi, on_change=self._on_slider)

    # ── Control callbacks ─────────────────────────────────────────────────────

    def _browse(self) -> None:
        """Open a directory picker and set ``folder_var`` to the chosen path."""
        if d := filedialog.askdirectory(title="Select image folder"):
            self.folder_var.set(d)

    def _update_tsne_visibility(self) -> None:
        """Show or hide the t-SNE parameter frame based on the projection selection."""
        if self.proj_var.get() == "tsne":
            self.tsne_frame.grid()
        else:
            self.tsne_frame.grid_remove()

    def _update_frames_visibility(self) -> None:
        """Show or hide the frames-per-video slider based on the media type selection."""
        if self.media_var.get() in ("videos", "all"):
            self.frames_frame.grid()
        else:
            self.frames_frame.grid_remove()

    def _on_media_type_change(self) -> None:
        """Update frames slider visibility when the media type selection changes."""
        self._update_frames_visibility()

    def _on_dims_change(self) -> None:
        """Switch the scatter view between 2D and 3D without re-running the pipeline."""
        if self._projection_2d is not None or self._projection_3d is not None:
            self._redraw()

    def _on_method_change(self) -> None:
        """Rebuild the cluster parameter widgets and schedule a re-cluster."""
        self._rebuild_cluster_params()
        self._schedule_recluster()

    def _rebuild_cluster_params(self) -> None:
        """Destroy and recreate the cluster parameter widgets for the current method."""
        for w in self.cluster_params_frame.winfo_children():
            w.destroy()
        f = self.cluster_params_frame
        method = self.cluster_method_var.get()

        if method == "threshold":
            self._cluster_slider(f, 0, "Threshold", self.threshold_var, 1, 100)

        elif method == "hierarchical":
            tk.Label(f, text="Linkage", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)).grid(
                row=0, column=0, sticky="w", padx=12, pady=(6, 0)
            )
            for i, lnk in enumerate(("single", "complete", "average")):
                tk.Radiobutton(
                    f,
                    text=lnk,
                    variable=self.linkage_var,
                    value=lnk,
                    bg=PANEL_BG,
                    fg=TEXT_COLOR,
                    selectcolor=PANEL_BG,
                    activebackground=PANEL_BG,
                    activeforeground=TEXT_COLOR,
                    font=("", 9),
                    anchor="w",
                    command=self._schedule_recluster,
                ).grid(row=i + 1, column=0, sticky="w", padx=12)
            self._cluster_slider(f, 4, "Cut Height", self.cut_height_var, 1, 100)

        elif method == "hdbscan":
            self._cluster_slider(f, 0, "Min Cluster Size", self.min_group_var, 2, 30)

    def _on_slider(self) -> None:
        """Slider change callback — debounce and schedule a re-cluster."""
        self._schedule_recluster()

    def _set_status(self, msg: str, error: bool = False) -> None:
        """Update the status label text and colour.

        Args:
            msg: Message to display.
            error: When ``True`` the label is shown in red.
        """
        self.status_lbl.config(text=msg, fg="#e05252" if error else DIM_COLOR)

    # ── Message queue poll ────────────────────────────────────────────────────

    def _poll(self) -> None:
        """Drain the inter-thread queue and dispatch UI updates.

        Rescheduled every 50 ms via ``after()``.
        """
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg["kind"]
                if kind == "status":
                    self._set_status(msg["text"], msg.get("error", False))
                elif kind == "progress":
                    n, total = msg["value"], msg["total"]
                    if total > 0:
                        pct = (n / total) * 100
                        self.progress.config(mode="determinate", value=pct)
                elif kind == "progress_indeterminate":
                    self.progress.config(mode="indeterminate")
                    self.progress.start(12)
                elif kind == "pipeline_done":
                    self._on_pipeline_done(msg["projection_2d"], msg["projection_3d"])
                elif kind == "cluster_done":
                    self._on_cluster_done(msg["clusters"])
                elif kind == "cancelled":
                    self._running = False
                    self._reset_run_button()
                    self._set_status("Cancelled.")
                elif kind == "cluster_error":
                    self.cluster_progress.stop()
                    self.cluster_progress.config(mode="determinate", value=0)
                    self._set_status(msg["text"], error=True)
                elif kind == "error":
                    self._on_error(msg["text"])
        except queue.Empty:
            pass
        self._poll_id = self.after(50, self._poll)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _reset_run_button(self) -> None:
        """Restore the Run button to its normal (non-cancel) state."""
        self.run_btn.config(
            text="Run",
            bg=ACCENT_COLOR,
            activebackground="#3a8ee6",
            command=self._on_run,
            state="normal" if self.binary else "disabled",
        )
        self.progress.stop()
        self.progress.config(mode="determinate", value=0)

    def _on_cancel(self) -> None:
        """Cancel the running pipeline by terminating the current subprocess."""
        if not self._running:
            return
        log.info("cancelling pipeline")
        self._cancel_requested = True
        proc = self._current_proc
        if proc is not None:
            proc.terminate()

    def _on_run(self) -> None:
        """Validate inputs and start the hash → project pipeline in a daemon thread."""
        if self._running:
            return
        folder = self.folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            self._set_status("Select a valid input folder first.", error=True)
            return
        if not self.binary:
            self._set_status("sift binary not found.", error=True)
            return

        log.info(
            "starting pipeline — folder=%s  algo=%s  proj=%s  (2D+3D)",
            folder,
            self.algo_var.get(),
            self.proj_var.get(),
        )
        self._running = True
        self._cancel_requested = False
        self._projection_2d = None
        self._projection_3d = None
        self._cluster_data = None
        self.run_btn.config(
            text="Cancel",
            bg="#c0392b",
            activebackground="#a93226",
            command=self._on_cancel,
        )
        self.progress.config(mode="determinate", value=0)
        self._clear_viewer()
        self._reset_axes()
        self.canvas.draw()
        threading.Thread(target=self._pipeline_thread, daemon=True).start()

    def _run_subprocess(
        self,
        cmd: list[str],
        stage: str,
        *,
        stream_progress: bool = False,
    ) -> tuple[int, str]:
        """Run *cmd* in a subprocess, streaming stderr line by line.

        Lines matching ``sift: progress N/TOTAL`` are turned into
        ``{"kind": "progress"}`` queue messages rather than being logged.
        All other non-empty lines are forwarded to the logger.

        Args:
            cmd: Full argument list for the subprocess.
            stage: Short label used in log messages (e.g. ``"hash"``).
            stream_progress: When ``True``, parse progress lines and emit
                queue updates; otherwise log all stderr lines normally.

        Returns:
            ``(returncode, joined_stderr)`` tuple.
        """
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self._current_proc = proc
        stderr_lines: list[str] = []

        for raw in proc.stderr:
            if self._cancel_requested:
                proc.terminate()
                break
            line = raw.strip()
            if not line:
                continue
            if stream_progress and line.startswith("sift: progress "):
                try:
                    frac = line.split()[-1]  # "N/TOTAL"
                    n, total = map(int, frac.split("/"))
                    self._q.put({"kind": "progress", "value": n, "total": total})
                except Exception:
                    pass
            else:
                _log_stderr(line, stage)
            stderr_lines.append(line)

        proc.wait()
        if self._current_proc is proc:
            self._current_proc = None
        return proc.returncode, "\n".join(stderr_lines)

    def _pipeline_thread(self) -> None:
        """Background thread: run ``sift hash`` then ``sift project``.

        Uses ``_run_subprocess`` so stderr is streamed line by line; progress
        lines from hashing are turned into bar updates.  Puts ``pipeline_done``,
        ``cancelled``, or ``error`` onto ``_q`` when finished.
        """
        try:
            folder = self.folder_var.get().strip()
            algo = self.algo_var.get()
            size = int(self.hash_size_var.get())
            media = self.media_var.get()
            frames = int(self.frames_var.get())
            proj_m = self.proj_var.get()

            # ── Hash ──────────────────────────────────────────────────────────
            media_label = {"images": "images", "videos": "videos", "all": "images+videos"}[media]
            self._q.put(
                {"kind": "status", "text": f"Hashing {media_label} with {algo} {size}×{size}…"}
            )
            hash_cmd = [
                self.binary,
                "hash",
                folder,
                f"--algo={algo}",
                f"--size={size}",
                f"--media={media}",
                f"--output={self._hashes_json}",
            ]
            if media in ("videos", "all"):
                hash_cmd.append(f"--frames={frames}")

            rc, stderr = self._run_subprocess(hash_cmd, "hash", stream_progress=True)

            if self._cancel_requested:
                self._q.put({"kind": "cancelled"})
                return
            if rc != 0:
                raise RuntimeError(stderr.splitlines()[-1] if stderr else "sift hash failed")

            # ── Project (both 2D and 3D so toggling needs no re-run) ──────────
            self._q.put({"kind": "progress_indeterminate"})

            def run_projection(dims: int, out_path: str) -> dict:
                self._q.put(
                    {
                        "kind": "status",
                        "text": f"Running {proj_m.upper()} projection ({dims}D)…",
                    }
                )
                cmd = [
                    self.binary,
                    "project",
                    self._hashes_json,
                    f"--method={proj_m}",
                    f"--dims={dims}",
                    f"--output={out_path}",
                ]
                if proj_m == "tsne":
                    cmd += [
                        f"--perplexity={int(self.perplexity_var.get())}",
                        f"--iterations={int(self.iterations_var.get())}",
                    ]
                rc, stderr = self._run_subprocess(cmd, "project")
                if self._cancel_requested:
                    return {}
                if rc != 0:
                    raise RuntimeError(stderr.splitlines()[-1] if stderr else "sift project failed")
                with open(out_path) as fh:
                    return json.load(fh)

            proj_2d = run_projection(2, self._proj_2d_json)
            if self._cancel_requested:
                self._q.put({"kind": "cancelled"})
                return

            proj_3d = run_projection(3, self._proj_3d_json)
            if self._cancel_requested:
                self._q.put({"kind": "cancelled"})
                return

            self._q.put(
                {"kind": "pipeline_done", "projection_2d": proj_2d, "projection_3d": proj_3d}
            )

        except Exception as exc:
            log.error("pipeline error: %s", exc)
            self._q.put({"kind": "error", "text": str(exc)})

    def _on_pipeline_done(self, proj_2d: dict, proj_3d: dict) -> None:
        """Handle successful pipeline completion — store both projections and cluster.

        Args:
            proj_2d: Parsed JSON dict from ``sift project --dims=2``.
            proj_3d: Parsed JSON dict from ``sift project --dims=3``.
        """
        self._running = False
        self._projection_2d = proj_2d
        self._projection_3d = proj_3d
        n = len(proj_2d["files"])
        self._reset_run_button()
        method = proj_2d.get("method", "?").upper()
        msg = f"{method} done — {n} files, 2D+3D ready. Adjust clustering below."
        log.info(msg)
        self._set_status(msg)
        self._run_cluster()

    def _on_error(self, msg: str) -> None:
        """Handle a pipeline or cluster error reported via the queue.

        Args:
            msg: Human-readable error description.
        """
        self._running = False
        self._reset_run_button()
        self._set_status(f"Error: {msg}", error=True)

    # ── Clustering ────────────────────────────────────────────────────────────

    def _schedule_recluster(self) -> None:
        """Debounce cluster requests: wait 350 ms after the last slider move."""
        if self._projection_2d is None:
            return
        if self._cluster_timer:
            self.after_cancel(self._cluster_timer)
        self._cluster_timer = self.after(350, self._run_cluster)

    def _run_cluster(self) -> None:
        """Build the ``sift cluster`` command from current UI state and run it."""
        if self._projection_2d is None:
            return
        self.cluster_progress.config(mode="indeterminate")
        self.cluster_progress.start(12)
        method = self.cluster_method_var.get()
        min_filter = int(self.min_filter_var.get())
        cmd = [
            self.binary,
            "cluster",
            self._hashes_json,
            f"--method={method}",
            f"--min-filter={min_filter}",
            f"--output={self._clusters_json}",
        ]
        if method == "threshold":
            cmd.append(f"--threshold={int(self.threshold_var.get())}")
        elif method == "hierarchical":
            cmd += [
                f"--linkage={self.linkage_var.get()}",
                f"--cut-height={int(self.cut_height_var.get())}",
            ]
        elif method == "hdbscan":
            cmd.append(f"--min-group={int(self.min_group_var.get())}")
        log.debug("cluster cmd: %s", " ".join(cmd))
        threading.Thread(target=self._cluster_thread, args=(cmd,), daemon=True).start()

    def _cluster_thread(self, cmd: list[str]) -> None:
        """Background thread: run the cluster command and push results to the queue.

        Args:
            cmd: Full argument list for ``sift cluster``.
        """
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "sift cluster failed")
            _log_stderr(r.stderr, "cluster")
            with open(self._clusters_json) as fh:
                clusters = json.load(fh)
            self._q.put({"kind": "cluster_done", "clusters": clusters})
        except Exception as exc:
            log.warning("cluster error: %s", exc)
            self._q.put({"kind": "cluster_error", "text": str(exc)})

    def _on_cluster_done(self, clusters: dict) -> None:
        """Store cluster results, update the status bar, and redraw the plot.

        Args:
            clusters: Parsed JSON dict from ``sift cluster``.
        """
        self.cluster_progress.stop()
        self.cluster_progress.config(mode="determinate", value=0)
        self._cluster_data = clusters
        n_groups = len(clusters.get("groups", []))
        n_ungrouped = len(clusters.get("ungrouped", []))
        msg = f"{n_groups} groups · {n_ungrouped} ungrouped"
        log.info("cluster done — %s", msg)
        self._set_status(msg)
        self._redraw()

    # ── Axes management ───────────────────────────────────────────────────────

    def _rebuild_axes(self) -> None:
        """Clear the figure and recreate ``self.ax`` for the current dimension mode.

        In 3D mode the axes gains built-in mouse-drag rotation and scroll-zoom
        provided by ``Axes3D`` at no extra cost.
        """
        self.fig.clear()
        if self._is_3d():
            self.ax = self.fig.add_subplot(111, projection="3d")
            self._style_axes_3d()
        else:
            self.ax = self.fig.add_subplot(111)
            self._style_axes_2d()

    def _style_axes_2d(self) -> None:
        """Apply the dark idle style to a 2D axes."""
        self.ax.set_facecolor(BG_COLOR)
        self.ax.set_title("Run the pipeline to begin", color=DIM_COLOR, fontsize=11)
        self.ax.tick_params(colors=DIM_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color("#333")
        self.fig.tight_layout()

    def _style_axes_3d(self) -> None:
        """Apply the dark idle style to a 3D axes."""
        self.ax.set_facecolor(BG_COLOR)
        for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#2e2e2e")
        self.ax.tick_params(colors=DIM_COLOR, labelsize=7)
        self.ax.set_title("Run the pipeline to begin", color=DIM_COLOR, fontsize=11)
        self.fig.tight_layout()

    def _reset_axes(self) -> None:
        """Rebuild axes and render the idle placeholder state."""
        self._rebuild_axes()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        """Replot all scatter points coloured by cluster membership.

        No boundary shapes are drawn — cluster membership is conveyed purely
        through point colour.  Calls ``_apply_selection_overlay`` at the end to
        restore any active selection ring.
        """
        proj = self._projection_3d if self._is_3d() else self._projection_2d
        if proj is None:
            proj = self._projection_2d or self._projection_3d
        if proj is None:
            return
        files = proj["files"]
        pts = np.array(proj["points"])
        n = len(files)
        is_3d = self._is_3d() and pts.ndim == 2 and pts.shape[1] == 3

        # Assign per-point colours and build point→group reverse mapping
        colors: list[Any] = [UNGROUPED_COLOR] * n
        self._point_group = {}
        n_groups = 0

        if self._cluster_data:
            groups = self._cluster_data.get("groups", [])
            n_groups = len(groups)
            palette = cluster_palette(n_groups)
            for gi, group in enumerate(groups):
                for idx in group["members"]:
                    if 0 <= idx < n:
                        colors[idx] = palette[gi]
                        self._point_group[idx] = gi

        self._rebuild_axes()

        # Normalise to a uniform RGBA array (Axes3D fails on mixed str/tuple lists)
        c_rgba = mcolors.to_rgba_array(colors)

        if is_3d:
            self.ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=c_rgba,
                s=36,
                linewidths=0.4,
                edgecolors="white",
                alpha=0.92,
                depthshade=True,
            )
        else:
            self.ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=c_rgba,
                s=36,
                linewidths=0.4,
                edgecolors="white",
                alpha=0.92,
                zorder=3,
            )

        self._point_positions = pts
        self._point_files = files

        method = proj.get("method", "pca")
        if method == "pca":
            ve = proj.get("variance_explained", [])
            xl = f"PC1  ({ve[0] * 100:.1f}%)" if ve else "PC1"
            yl = f"PC2  ({ve[1] * 100:.1f}%)" if len(ve) > 1 else "PC2"
            zl = f"PC3  ({ve[2] * 100:.1f}%)" if len(ve) > 2 else "PC3"
        else:
            xl, yl, zl = "t-SNE 1", "t-SNE 2", "t-SNE 3"

        self.ax.set_xlabel(xl, color=DIM_COLOR, fontsize=9)
        self.ax.set_ylabel(yl, color=DIM_COLOR, fontsize=9)
        if is_3d:
            self.ax.set_zlabel(zl, color=DIM_COLOR, fontsize=9)
            for pane in (self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("#2e2e2e")
            self.ax.tick_params(colors="#555", labelsize=7)
        else:
            self.ax.tick_params(colors="#555", labelsize=8)
            for spine in self.ax.spines.values():
                spine.set_color("#2e2e2e")

        dim_label = "3D" if is_3d else "2D"
        self.ax.set_title(
            f"{n} files — {n_groups} cluster{'s' if n_groups != 1 else ''}  [{dim_label}]"
            if self._cluster_data
            else f"{n} files  [{dim_label}]",
            color=TEXT_COLOR,
            fontsize=11,
            pad=10,
        )
        self.fig.tight_layout()
        self._overlay_artists = []
        self._apply_selection_overlay()

    # ── Selection overlay ─────────────────────────────────────────────────────

    def _apply_selection_overlay(self) -> None:
        """Remove stale overlay artists and draw a ring on the currently viewed point.

        In 3D mode the axis limits are saved and restored around the draw so
        that adding the ring scatter does not trigger autoscale and move the
        camera.
        """
        for a in self._overlay_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay_artists = []

        pts = self._point_positions
        if pts is None:
            self.canvas.draw()
            return

        sel = self._selection

        pt_idx: int | None = None
        if sel.group_idx is not None:
            if sel.member_indices and 0 <= self._viewer.index < len(sel.member_indices):
                pt_idx = sel.member_indices[self._viewer.index]
        elif sel.point_idx is not None:
            pt_idx = sel.point_idx

        # Snapshot 3D view state before adding any artists — new scatter calls
        # trigger autoscale_view() which would reset zoom/pan.
        is_3d = pts.ndim == 2 and pts.shape[1] == 3
        if is_3d:
            xlim = self.ax.get_xlim3d()
            ylim = self.ax.get_ylim3d()
            zlim = self.ax.get_zlim3d()

        if pt_idx is not None and 0 <= pt_idx < len(pts):
            self._draw_selection_ring(pts[pt_idx])

        if is_3d:
            self.ax.set_xlim3d(xlim)
            self.ax.set_ylim3d(ylim)
            self.ax.set_zlim3d(zlim)

        self.canvas.draw()

    def _draw_selection_ring(self, xy: np.ndarray) -> None:
        """Draw a white ring + accent dot at the given scatter coordinate.

        Works in both 2D (x, y) and 3D (x, y, z) based on the shape of *xy*.

        Args:
            xy: 2- or 3-element array of data-space coordinates.
        """
        if len(xy) == 3:
            x, y, z = float(xy[0]), float(xy[1]), float(xy[2])
            outer = self.ax.scatter(
                [x], [y], [z], s=220, c="none", edgecolors="white", linewidths=2.5
            )
            inner = self.ax.scatter(
                [x], [y], [z], s=70, c=[ACCENT_COLOR], edgecolors="none", alpha=0.9
            )
        else:
            x, y = float(xy[0]), float(xy[1])
            outer = self.ax.scatter(
                [x], [y], s=220, c="none", edgecolors="white", linewidths=2.5, zorder=7
            )
            inner = self.ax.scatter(
                [x], [y], s=70, c=[ACCENT_COLOR], edgecolors="none", alpha=0.9, zorder=8
            )
        self._overlay_artists.extend([outer, inner])

    # ── Click detection ───────────────────────────────────────────────────────

    def _on_mpl_click(self, event: Any) -> None:
        """Handle a matplotlib mouse click in both 2D and 3D modes.

        In 2D: hull bubbles take priority over points (contains_point test).
        In 3D: nearest-point only via screen-space projection (hull contains_point
        is not reliable in 3D display space).

        Args:
            event: Matplotlib ``MouseEvent`` from the canvas.
        """
        if event.inaxes is not self.ax or self._point_positions is None:
            return

        pts = self._point_positions
        is_3d = pts.ndim == 2 and pts.shape[1] == 3

        if is_3d:
            self._hit_test_3d(event, pts)
        else:
            self._hit_test_2d(event, pts)

    def _select_nearest(self, nearest: int) -> None:
        """Select the nearest point, opening its group starting at that point.

        Args:
            nearest: Index of the closest scatter point to the click.
        """
        gi = self._point_group.get(nearest)
        if gi is not None:
            self._show_group(gi, start_point=nearest)
        else:
            self._show_single(nearest)

    def _hit_test_2d(self, event: Any, pts: np.ndarray) -> None:
        """2D click: find nearest scatter point in pixel space and select it.

        Args:
            event: Matplotlib mouse event.
            pts: ``(N, 2)`` point positions in data space.
        """
        click_pt = np.array([event.xdata, event.ydata])
        click_disp = self.ax.transData.transform(click_pt.reshape(1, 2))[0]
        pts_disp = self.ax.transData.transform(pts)
        dists = np.linalg.norm(pts_disp - click_disp, axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] <= POINT_HIT_RADIUS_PX:
            self._select_nearest(nearest)

    def _hit_test_3d(self, event: Any, pts: np.ndarray) -> None:
        """3D click: project all points to screen space, select nearest.

        Uses ``mpl_toolkits.mplot3d.proj3d.proj_transform`` to obtain 2D
        display coordinates for each point, then computes pixel distance from
        the mouse click.

        Args:
            event: Matplotlib mouse event (``event.x`` / ``event.y`` are raw
                canvas pixels in 3D mode).
            pts: ``(N, 3)`` point positions in data space.
        """
        try:
            proj_mat = self.ax.get_proj()
            x2d, y2d, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], proj_mat)
            pts_ax = np.column_stack([x2d, y2d])
            pts_disp = self.ax.transData.transform(pts_ax)
            click_disp = np.array([event.x, event.y])
            dists = np.linalg.norm(pts_disp - click_disp, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] <= POINT_HIT_RADIUS_PX:
                self._select_nearest(nearest)
        except Exception as exc:
            log.debug("3D hit test failed: %s", exc)

    # ── Viewer: selection ─────────────────────────────────────────────────────

    def _show_single(self, point_idx: int) -> None:
        """Select a single scatter point and update the viewer for its file.

        Args:
            point_idx: Index into ``_point_files`` / ``_point_positions``.
        """
        log.debug("selected point %d: %s", point_idx, self._point_files[point_idx])
        self._viewer = ViewerState(files=[self._point_files[point_idx]])
        self._selection = SelectionState(point_idx=point_idx)
        self._update_viewer()
        self._apply_selection_overlay()

    def _show_group(self, group_idx: int, start_point: int | None = None) -> None:
        """Select a cluster group, starting the viewer at the clicked point.

        Args:
            group_idx: Index into ``_cluster_data["groups"]``.
            start_point: Point-array index of the clicked point; the viewer
                opens at that position within the group so the user sees the
                file they actually clicked, not always the first member.
        """
        if not self._cluster_data:
            return
        groups = self._cluster_data.get("groups", [])
        if group_idx >= len(groups):
            return
        members = groups[group_idx]["members"]
        valid_members = [i for i in members if 0 <= i < len(self._point_files)]
        files = [self._point_files[i] for i in valid_members]
        if not files:
            return
        initial_index = 0
        if start_point is not None and start_point in valid_members:
            initial_index = valid_members.index(start_point)
        log.debug(
            "selected group %d (%d files), starting at member %d",
            group_idx,
            len(files),
            initial_index,
        )
        self._viewer = ViewerState(files=files, index=initial_index, group_id=group_idx)
        self._selection = SelectionState(group_idx=group_idx, member_indices=valid_members)
        self._update_viewer()
        self._apply_selection_overlay()

    # ── Viewer: display ───────────────────────────────────────────────────────

    def _update_viewer(self) -> None:
        """Refresh all viewer widgets (nav buttons, labels, image) for the current file."""
        player = self._video_player
        if player and (player.is_playing or player.is_paused):
            player.stop()
            self.play_pause_btn.config(text="▶  Play")
            self.seek_frame.grid_remove()
            self.skip_back_btn.config(state="disabled")
            self.skip_fwd_btn.config(state="disabled")

        v = self._viewer
        if not v.files:
            return

        path = v.current_path
        is_video = Path(path).suffix.lower() in VIDEO_EXTS

        self.prev_btn.config(state="normal" if v.is_multi else "disabled")
        self.next_btn.config(state="normal" if v.is_multi else "disabled")
        self.nav_lbl.config(text=f"{v.index + 1} / {len(v.files)}" if v.is_multi else "")

        self.play_pause_btn.config(state="normal" if is_video else "disabled")
        self.skip_back_btn.config(state="disabled")
        self.skip_fwd_btn.config(state="disabled")
        self.open_btn.config(state="normal")
        self.folder_btn.config(state="normal")

        self.viewer_name_lbl.config(text=Path(path).name)
        self.viewer_type_lbl.config(text="")
        self.viewer_meta_lbl.config(text="")
        self.viewer_cluster_lbl.config(
            text=f"Cluster {v.group_id + 1}  ·  {len(v.files)} files"
            if v.group_id is not None
            else ""
        )
        self.viewer_path_lbl.config(text=str(Path(path).parent))

        threading.Thread(target=self._load_and_display, args=(path,), daemon=True).start()

    def _load_and_display(self, path: str) -> None:
        """Load media and metadata in a background thread, then schedule a UI update.

        Args:
            path: Absolute path to the file to display.
        """
        img = load_media(path)
        type_line, meta = get_file_info(path)
        self.after(0, self._display_image, img, path, type_line, meta)

    def _display_image(self, img: Image.Image | None, path: str, type_line: str, meta: str) -> None:
        """Apply a loaded image to the canvas (called on the main thread via ``after``).

        Args:
            img: Decoded ``PIL.Image``, or ``None`` if loading failed.
            path: The file path that was loaded (staleness check).
            type_line: First info line (e.g. ``"IMAGE · PNG"``).
            meta: Second info line (dimensions, size, duration).
        """
        if not self._viewer.files or self._viewer.current_path != path:
            return

        self.viewer_type_lbl.config(text=type_line)
        self.viewer_meta_lbl.config(text=meta)
        self._current_img = img
        self._render_image()

    def _render_image(self) -> None:
        """Render ``_current_img`` into the canvas scaled to its current dimensions."""
        self.img_canvas.delete("all")
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()

        if cw <= 1 or ch <= 1:
            return

        if self._current_img is None:
            path = self._viewer.current_path
            ext = Path(path).suffix.lower() if path else ""
            if ext in VIDEO_EXTS:
                msg = (
                    "VIDEO\n(install ffmpeg\nto preview frames)"
                    if not HAS_FFMPEG
                    else "VIDEO\n(preview failed)"
                )
            else:
                msg = "Click a point\nor cluster" if not path else "?"
            self.img_canvas.create_text(
                cw // 2, ch // 2, text=msg, fill=DIM_COLOR, font=("", 12), justify="center"
            )
            return

        pad = 8
        fitted = fit_image(self._current_img.copy(), cw - pad, ch - pad)
        iw, ih = fitted.size
        photo = ImageTk.PhotoImage(fitted)
        self._thumb_ref = photo
        self.img_canvas.create_image((cw - iw) // 2, (ch - ih) // 2, anchor="nw", image=photo)

    def _on_canvas_resize(self, _: Any = None) -> None:
        """Re-render the current image whenever the canvas is resized."""
        self._render_image()

    def _draw_placeholder(self) -> None:
        """Draw the initial 'Click a point or cluster' placeholder text."""
        cw = self.img_canvas.winfo_width() or THUMB_W
        ch = self.img_canvas.winfo_height() or THUMB_H
        self.img_canvas.create_text(
            cw // 2,
            ch // 2,
            text="Click a point\nor cluster",
            fill=DIM_COLOR,
            font=("", 11),
            justify="center",
        )

    def _clear_viewer(self) -> None:
        """Reset the viewer panel to its empty/placeholder state."""
        player = self._video_player
        if player and (player.is_playing or player.is_paused):
            player.stop()
        self.seek_frame.grid_remove()
        self._viewer = ViewerState()
        self._selection = SelectionState()
        self._thumb_ref = None
        self._current_img = None
        self.img_canvas.delete("all")
        self._draw_placeholder()
        for lbl in (
            self.viewer_name_lbl,
            self.viewer_type_lbl,
            self.viewer_meta_lbl,
            self.viewer_path_lbl,
        ):
            lbl.config(text="")
        self.viewer_cluster_lbl.config(text="Click a point or cluster\nto preview media")
        self.nav_lbl.config(text="")
        self.play_pause_btn.config(text="▶  Play")
        for btn in (
            self.prev_btn, self.next_btn, self.play_pause_btn,
            self.skip_back_btn, self.skip_fwd_btn,
            self.open_btn, self.folder_btn,
        ):
            btn.config(state="disabled")

    # ── Viewer: navigation ────────────────────────────────────────────────────

    def _viewer_prev(self) -> None:
        """Navigate to the previous file in the current group (wraps around)."""
        if not self._viewer.files:
            return
        self._viewer.index = (self._viewer.index - 1) % len(self._viewer.files)
        self._update_viewer()
        self._apply_selection_overlay()

    def _viewer_next(self) -> None:
        """Navigate to the next file in the current group (wraps around)."""
        if not self._viewer.files:
            return
        self._viewer.index = (self._viewer.index + 1) % len(self._viewer.files)
        self._update_viewer()
        self._apply_selection_overlay()

    # ── Viewer: playback controls ─────────────────────────────────────────────

    def _toggle_play_pause(self) -> None:
        """Toggle between playing and paused states for the current video."""
        player = self._video_player
        if not player:
            return

        if player.is_playing:
            # Currently playing → pause
            player.pause()
            self.play_pause_btn.config(text="▶  Play")
            # Keep seek frame visible so user can scrub while paused

        elif player.is_paused:
            # Paused → resume
            player.resume()
            self.play_pause_btn.config(text="⏸  Pause")

        else:
            # Stopped → start from beginning
            path = self._viewer.current_path
            if not path:
                return
            if not HAS_FFMPEG:
                log.warning("ffmpeg not found — falling back to system open")
                system_open(path)
                return
            log.info("starting inline playback: %s", path)
            ok = player.play(path, position_cb=self._on_position_update)
            if not ok:
                log.warning("could not start playback for %s", path)
                return
            # Configure seek bar
            dur = player.duration
            self.seek_scale.config(to=max(1.0, dur))
            self.seek_var.set(0.0)
            self.seek_pos_lbl.config(text="0:00")
            self.seek_dur_lbl.config(text=_fmt_duration(dur))
            self.seek_frame.grid()
            self.play_pause_btn.config(text="⏸  Pause")
            self.skip_back_btn.config(state="normal")
            self.skip_fwd_btn.config(state="normal")
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")

    def _skip_back(self) -> None:
        """Skip backward 5 seconds."""
        player = self._video_player
        if player and (player.is_playing or player.is_paused):
            player.seek(max(0.0, player.position - 5.0))
            self.play_pause_btn.config(text="⏸  Pause")

    def _skip_forward(self) -> None:
        """Skip forward 5 seconds."""
        player = self._video_player
        if player and (player.is_playing or player.is_paused):
            dur = player.duration
            t = player.position + 5.0
            if dur > 0:
                t = min(t, dur)
            player.seek(t)
            self.play_pause_btn.config(text="⏸  Pause")

    def _on_seek_press(self, _: Any) -> None:
        """Mark the start of a user scrub drag."""
        self._user_scrubbing = True

    def _on_seek_release(self, _: Any) -> None:
        """Seek to the scrub position when the user releases the slider."""
        self._user_scrubbing = False
        player = self._video_player
        if player and (player.is_playing or player.is_paused):
            player.seek(self.seek_var.get())
            self.play_pause_btn.config(text="⏸  Pause")

    def _on_seek_drag(self, val: Any) -> None:
        """Update the current-position time label while the user drags."""
        self.seek_pos_lbl.config(text=_fmt_duration(float(val)))

    def _on_position_update(self, pos: float) -> None:
        """Called by VideoPlayer each frame tick with the current position."""
        if not self._user_scrubbing:
            self.seek_var.set(pos)
            self.seek_pos_lbl.config(text=_fmt_duration(pos))

    def _on_video_end(self) -> None:
        """Called when playback ends naturally — restore viewer to thumbnail state."""
        self.play_pause_btn.config(text="▶  Play")
        self.seek_frame.grid_remove()
        self.skip_back_btn.config(state="disabled")
        self.skip_fwd_btn.config(state="disabled")
        v = self._viewer
        can_nav = v.is_multi
        self.prev_btn.config(state="normal" if can_nav else "disabled")
        self.next_btn.config(state="normal" if can_nav else "disabled")
        self._render_image()

    # ── Viewer: file actions ──────────────────────────────────────────────────

    def _open_media(self) -> None:
        """Open the current file with the system default application."""
        if path := self._viewer.current_path:
            log.info("opening: %s", path)
            system_open(path)

    def _open_in_folder(self) -> None:
        """Reveal the current file in the system file manager."""
        if path := self._viewer.current_path:
            log.info("revealing in folder: %s", path)
            system_open(path, reveal=True)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Create and run the SiftViz application."""
    app = SiftViz()
    app.mainloop()


if __name__ == "__main__":
    main()
