#!/usr/bin/env python3
"""sift viz — interactive visualization frontend for the sift CLI tool.

Flow:
  1. Select folder + hash algo + projection method → Run
     → calls `sift hash` then `sift project` (fixed 2D canvas)
  2. Adjust clustering algo + sliders → live re-cluster
     → calls `sift cluster`, recolors points + redraws hull bubbles
  3. Click a hull bubble → viewer shows all files in that cluster with ◀ ▶ nav
     Click an ungrouped point → viewer shows that single file
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
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
from PIL import Image, ImageTk

try:
    from scipy.spatial import ConvexHull

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
HULL_EXPAND: float = 1.06

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


# ── Convex hull ───────────────────────────────────────────────────────────────


def build_hull_verts(pts: np.ndarray) -> np.ndarray | None:
    """Compute expanded convex hull vertices for a cluster's scatter points.

    For fewer than 3 points (or when scipy is unavailable) an ellipse
    approximation is returned instead.

    Args:
        pts: ``(N, 2)`` array of 2-D scatter coordinates.

    Returns:
        ``(M, 2)`` closed polygon vertices (first == last), or ``None`` if
        fewer than 2 points are provided.
    """
    if len(pts) < 2:
        return None
    if len(pts) < 3 or not HAS_SCIPY:
        cx, cy = pts.mean(axis=0)
        rx = max(float(np.std(pts[:, 0])) * 2.5, 0.02)
        ry = max(float(np.std(pts[:, 1])) * 2.5, 0.02)
        angles = np.linspace(0, 2 * np.pi, 32)
        return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        centroid = verts.mean(axis=0)
        expanded = centroid + (verts - centroid) * HULL_EXPAND
        return np.vstack([expanded, expanded[0]])
    except Exception:
        return None


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
    group_id: int | None = None  # None = single-point selection

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

    Exactly one of *point_idx* or *group_idx* is set at a time; the other
    remains ``None``.

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
    """Emit each non-empty line of *stderr* as an INFO log entry.

    Args:
        stderr: Raw stderr text from a sift subprocess.
        prefix: Short label prepended to each line (e.g. ``"hash"``).
    """
    _IGNORED_PREFIXES = ("libpng warning:", "libpng error:")
    for line in stderr.splitlines():
        line = line.strip()
        if line and not any(line.startswith(p) for p in _IGNORED_PREFIXES):
            log.info("[sift %s] %s", prefix, line)


# ── Main application ──────────────────────────────────────────────────────────


class SiftViz(tk.Tk):
    """Main application window for the sift interactive visualizer.

    Arranges three resizable panels (controls | scatter plot | media viewer)
    inside a ``tk.PanedWindow``.  Background threads communicate back to the
    UI via a ``queue.Queue`` polled every 50 ms with ``after()``.
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
        self._proj_json = os.path.join(self.tmpdir, "projection.json")
        self._clusters_json = os.path.join(self.tmpdir, "clusters.json")

        self._q: queue.Queue[dict] = queue.Queue()
        self._projection_data: dict | None = None
        self._cluster_data: dict | None = None
        self._running = False
        self._cluster_timer: str | None = None
        self._poll_id: str | None = None

        # Hit-test state rebuilt on every _redraw
        self._point_positions: np.ndarray | None = None
        self._point_files: list[str] = []
        self._hull_paths: list[tuple[mpath.Path, int]] = []
        self._hull_verts: list[tuple[np.ndarray, int, Any]] = []
        self._overlay_artists: list[Artist] = []

        self._viewer = ViewerState()
        self._selection = SelectionState()
        self._current_img: Image.Image | None = None  # kept for canvas resize re-render

        # tk vars
        self.folder_var = tk.StringVar()
        self.algo_var = tk.StringVar(value="dhash")
        self.hash_size_var = tk.IntVar(value=8)
        self.proj_var = tk.StringVar(value="pca")
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
        plt.close("all")
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.destroy()
        sys.exit(0)

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

        # Projection
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

        self.progress = ttk.Progressbar(ctrl, mode="indeterminate", length=100)
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

    # ── Scatter panel ─────────────────────────────────────────────────────────

    def _build_scatter_panel(self, pf: tk.Frame) -> None:
        """Create the matplotlib scatter canvas in the centre panel.

        Args:
            pf: The parent ``tk.Frame`` for the scatter panel.
        """
        pf.columnconfigure(0, weight=1)
        pf.rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.fig.patch.set_facecolor(BG_COLOR)
        self._reset_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_mpl_click)
        self.canvas.draw()

    # ── Viewer panel ──────────────────────────────────────────────────────────

    def _build_viewer_panel(self, v: tk.Frame) -> None:
        """Build the right-hand media viewer with image canvas and action buttons.

        The canvas row (row 1) is configured with ``weight=1`` so the image
        area fills all available vertical space.

        Args:
            v: The parent ``tk.Frame`` for the viewer panel.
        """
        v.columnconfigure(0, weight=1)
        v.rowconfigure(1, weight=1)  # canvas row expands

        self._section_label(v, 0, "MEDIA VIEWER")

        self.img_canvas = tk.Canvas(v, bg="#111111", highlightthickness=0)
        self.img_canvas.grid(row=1, column=0, padx=12, pady=(4, 0), sticky="nsew")
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)
        self._draw_placeholder()

        self.viewer_name_lbl = self._viewer_label(v, 2, font=("", 9, "bold"), fg=TEXT_COLOR)
        self.viewer_type_lbl = self._viewer_label(v, 3)
        self.viewer_meta_lbl = self._viewer_label(v, 4)
        self.viewer_cluster_lbl = self._viewer_label(
            v, 5, text="Click a point or cluster\nto preview media", fg="#444"
        )

        # Navigation
        nav = tk.Frame(v, bg=PANEL_BG)
        nav.grid(row=6, column=0, pady=(10, 0))
        self.prev_btn = self._icon_btn(nav, "◀", self._viewer_prev, col=0)
        self.nav_lbl = tk.Label(
            nav, text="", bg=PANEL_BG, fg=DIM_COLOR, font=("", 9), width=8
        )
        self.nav_lbl.grid(row=0, column=1)
        self.next_btn = self._icon_btn(nav, "▶", self._viewer_next, col=2)
        for b in (self.prev_btn, self.next_btn):
            b.config(state="disabled")

        # Actions
        act = tk.Frame(v, bg=PANEL_BG)
        act.grid(row=7, column=0, pady=(8, 0))
        self.play_btn = self._action_btn(act, "▶  Play", self._play_media, col=0)
        self.open_btn = self._action_btn(act, "↗  Open", self._open_media, col=1)
        self.folder_btn = self._action_btn(act, "⌂  Folder", self._open_in_folder, col=2)
        for b in (self.play_btn, self.open_btn, self.folder_btn):
            b.config(state="disabled")

        self.viewer_path_lbl = self._viewer_label(
            v, 8, fg="#383838", font=("", 7), pady=(6, 12)
        )

    # ── Widget factory helpers ────────────────────────────────────────────────

    def _section_label(self, parent: tk.Widget, row: int, text: str) -> None:
        """Add a small uppercase section heading label to *parent* at *row*.

        Args:
            parent: Container widget.
            row: Grid row index.
            text: Heading text (displayed uppercase, dim colour).
        """
        tk.Label(
            parent, text=text, bg=PANEL_BG, fg=DIM_COLOR, font=("", 8, "bold")
        ).grid(row=row, column=0, sticky="w", padx=12, pady=(14, 2))

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
            parent: Container widget (expected to be a nav row frame).
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

    def _action_btn(
        self, parent: tk.Widget, text: str, command: Any, col: int
    ) -> tk.Button:
        """Create, grid, and return a viewer action button (Play / Open / Folder).

        Args:
            parent: Container widget (action row frame).
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
            The value ``tk.Label`` widget (useful for storing a reference).
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

        Used inside the t-SNE options frame.

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

        Convenience wrapper used when building the dynamic cluster params frame.

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

        Rescheduled every 50 ms via ``after()``.  The poll ID is stored in
        ``_poll_id`` so it can be cancelled on window close.
        """
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg["kind"]
                if kind == "status":
                    self._set_status(msg["text"], msg.get("error", False))
                elif kind == "pipeline_done":
                    self._on_pipeline_done(msg["projection"])
                elif kind == "cluster_done":
                    self._on_cluster_done(msg["clusters"])
                elif kind == "error":
                    self._on_error(msg["text"])
        except queue.Empty:
            pass
        self._poll_id = self.after(50, self._poll)

    # ── Pipeline ──────────────────────────────────────────────────────────────

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
            "starting pipeline — folder=%s  algo=%s  proj=%s",
            folder,
            self.algo_var.get(),
            self.proj_var.get(),
        )
        self._running = True
        self._projection_data = None
        self._cluster_data = None
        self.run_btn.config(state="disabled")
        self.progress.start(10)
        self._clear_viewer()
        self._reset_axes()
        self.canvas.draw()
        threading.Thread(target=self._pipeline_thread, daemon=True).start()

    def _pipeline_thread(self) -> None:
        """Background thread: run ``sift hash`` then ``sift project``.

        Puts ``pipeline_done`` or ``error`` messages onto ``_q`` when done.
        Each sift subprocess's stderr is forwarded to the logger line-by-line
        so progress and timing appear in the terminal.
        """
        try:
            folder = self.folder_var.get().strip()
            algo = self.algo_var.get()
            size = int(self.hash_size_var.get())
            proj_m = self.proj_var.get()

            # Hash
            self._q.put({"kind": "status", "text": f"Hashing with {algo} {size}×{size}…"})
            r = subprocess.run(
                [
                    self.binary,
                    "hash",
                    folder,
                    f"--algo={algo}",
                    f"--size={size}",
                    f"--output={self._hashes_json}",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "sift hash failed")
            _log_stderr(r.stderr, "hash")

            # Project
            self._q.put({"kind": "status", "text": f"Running {proj_m.upper()} projection…"})
            cmd = [
                self.binary,
                "project",
                self._hashes_json,
                f"--method={proj_m}",
                "--dims=2",
                f"--output={self._proj_json}",
            ]
            if proj_m == "tsne":
                cmd += [
                    f"--perplexity={int(self.perplexity_var.get())}",
                    f"--iterations={int(self.iterations_var.get())}",
                ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "sift project failed")
            _log_stderr(r.stderr, "project")

            with open(self._proj_json) as fh:
                proj = json.load(fh)
            self._q.put({"kind": "pipeline_done", "projection": proj})

        except Exception as exc:
            log.error("pipeline error: %s", exc)
            self._q.put({"kind": "error", "text": str(exc)})

    def _on_pipeline_done(self, projection: dict) -> None:
        """Handle successful pipeline completion — store data and trigger clustering.

        Args:
            projection: Parsed JSON dict from ``sift project``.
        """
        self._running = False
        self._projection_data = projection
        n = len(projection["files"])
        self.progress.stop()
        self.run_btn.config(state="normal")
        method = projection.get("method", "?").upper()
        msg = f"{method} done — {n} images. Adjust clustering below."
        log.info(msg)
        self._set_status(msg)
        self._run_cluster()

    def _on_error(self, msg: str) -> None:
        """Handle a pipeline or cluster error reported via the queue.

        Args:
            msg: Human-readable error description.
        """
        self._running = False
        self.progress.stop()
        self.run_btn.config(state="normal")
        self._set_status(f"Error: {msg}", error=True)

    # ── Clustering ────────────────────────────────────────────────────────────

    def _schedule_recluster(self) -> None:
        """Debounce cluster requests: wait 350 ms after the last slider move."""
        if not self._projection_data:
            return
        if self._cluster_timer:
            self.after_cancel(self._cluster_timer)
        self._cluster_timer = self.after(350, self._run_cluster)

    def _run_cluster(self) -> None:
        """Build the ``sift cluster`` command from current UI state and run it."""
        if not self._projection_data:
            return
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
            self._q.put({"kind": "status", "text": str(exc), "error": True})

    def _on_cluster_done(self, clusters: dict) -> None:
        """Store cluster results, update the status bar, and redraw the plot.

        Args:
            clusters: Parsed JSON dict from ``sift cluster``.
        """
        self._cluster_data = clusters
        n_groups = len(clusters.get("groups", []))
        n_ungrouped = len(clusters.get("ungrouped", []))
        msg = f"{n_groups} groups · {n_ungrouped} ungrouped"
        log.info("cluster done — %s", msg)
        self._set_status(msg)
        self._redraw()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _reset_axes(self) -> None:
        """Clear the axes and apply the idle/placeholder style."""
        self.ax.cla()
        self.ax.set_facecolor(BG_COLOR)
        self.ax.set_title("Run the pipeline to begin", color=DIM_COLOR, fontsize=11)
        self.ax.tick_params(colors=DIM_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color("#333")
        self.fig.tight_layout()

    def _redraw(self) -> None:
        """Replot all scatter points and hull polygons from current data.

        Rebuilds ``_hull_paths`` and ``_hull_verts`` for hit-testing, then
        calls ``_apply_selection_overlay`` to restore any active selection.
        """
        if not self._projection_data:
            return

        proj = self._projection_data
        files = proj["files"]
        pts = np.array(proj["points"])
        n = len(files)

        colors: list[Any] = [UNGROUPED_COLOR] * n
        membership: dict[int, list[int]] = {}

        if self._cluster_data:
            groups = self._cluster_data.get("groups", [])
            palette = cluster_palette(len(groups))
            for gi, group in enumerate(groups):
                for idx in group["members"]:
                    if 0 <= idx < n:
                        colors[idx] = palette[gi]
                        membership.setdefault(gi, []).append(idx)

        self.ax.cla()
        self.ax.set_facecolor(BG_COLOR)

        self._hull_paths = []
        self._hull_verts = []
        if self._cluster_data:
            groups = self._cluster_data.get("groups", [])
            palette = cluster_palette(len(groups))  # same palette, O(1) per call
            for gi, indices in membership.items():
                color = palette[gi]
                verts = build_hull_verts(pts[indices])
                if verts is not None:
                    self._hull_paths.append((mpath.Path(verts), gi))
                    self._hull_verts.append((verts, gi, color))
                    self.ax.add_patch(
                        Polygon(
                            verts,
                            closed=True,
                            facecolor=color,
                            alpha=0.10,
                            edgecolor=color,
                            linewidth=1.2,
                            linestyle="--",
                            zorder=1,
                        )
                    )

        self.ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=colors,
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
        else:
            xl, yl = "t-SNE 1", "t-SNE 2"

        self.ax.set_xlabel(xl, color=DIM_COLOR, fontsize=9)
        self.ax.set_ylabel(yl, color=DIM_COLOR, fontsize=9)
        self.ax.tick_params(colors="#555", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#2e2e2e")

        n_groups = len(membership)
        self.ax.set_title(
            f"{n} images — {n_groups} cluster{'s' if n_groups != 1 else ''}"
            if self._cluster_data
            else f"{n} images",
            color=TEXT_COLOR,
            fontsize=11,
            pad=10,
        )
        self.fig.tight_layout()
        self._overlay_artists = []
        self._apply_selection_overlay()

    # ── Selection overlay ─────────────────────────────────────────────────────

    def _apply_selection_overlay(self) -> None:
        """Remove stale overlay artists and draw fresh ones for the current selection.

        For a group selection: brightens the hull polygon and draws a ring on
        the currently viewed member point.
        For a single-point selection: draws a ring on that point only.
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
        if sel.group_idx is not None:
            # Solid white hull border + brighter fill for the selected group
            for verts, gi, color in self._hull_verts:
                if gi == sel.group_idx:
                    poly = Polygon(
                        verts,
                        closed=True,
                        facecolor=color,
                        alpha=0.22,
                        edgecolor="white",
                        linewidth=2.0,
                        linestyle="-",
                        zorder=2,
                    )
                    self.ax.add_patch(poly)
                    self._overlay_artists.append(poly)
                    break
            # Ring on the currently viewed member
            if sel.member_indices and 0 <= self._viewer.index < len(sel.member_indices):
                pt_idx = sel.member_indices[self._viewer.index]
                if 0 <= pt_idx < len(pts):
                    self._draw_selection_ring(pts[pt_idx])

        elif sel.point_idx is not None and 0 <= sel.point_idx < len(pts):
            self._draw_selection_ring(pts[sel.point_idx])

        self.canvas.draw()

    def _draw_selection_ring(self, xy: np.ndarray) -> None:
        """Draw a white ring + accent dot at scatter coordinate *xy*.

        The ring and dot are added to ``_overlay_artists`` so they can be
        removed on the next selection change without a full redraw.

        Args:
            xy: 2-element array of data-space coordinates.
        """
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
        """Handle a matplotlib mouse click: hull bubbles take priority over points.

        If the click lands inside a hull polygon ``_show_group`` is called.
        Otherwise the nearest scatter point within ``POINT_HIT_RADIUS_PX``
        pixels is selected via ``_show_single``.

        Args:
            event: Matplotlib ``MouseEvent`` from the canvas.
        """
        if event.inaxes is not self.ax or self._point_positions is None:
            return

        click_pt = np.array([event.xdata, event.ydata])
        click_disp = self.ax.transData.transform(click_pt.reshape(1, 2))[0]
        pts_disp = self.ax.transData.transform(self._point_positions)
        dists = np.linalg.norm(pts_disp - click_disp, axis=1)

        # Bubbles take priority over individual points
        for hull_path, gi in self._hull_paths:
            if hull_path.contains_point(click_pt):
                self._show_group(gi)
                return

        nearest = int(np.argmin(dists))
        if dists[nearest] <= POINT_HIT_RADIUS_PX:
            self._show_single(nearest)

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

    def _show_group(self, group_idx: int) -> None:
        """Select a cluster group and load all its member files into the viewer.

        Args:
            group_idx: Index into ``_cluster_data["groups"]``.
        """
        if not self._cluster_data:
            return
        groups = self._cluster_data.get("groups", [])
        if group_idx >= len(groups):
            return
        members = groups[group_idx]["members"]
        files = [self._point_files[i] for i in members if 0 <= i < len(self._point_files)]
        if not files:
            return
        valid_members = [i for i in members if 0 <= i < len(self._point_files)]
        log.debug("selected group %d (%d files)", group_idx, len(files))
        self._viewer = ViewerState(files=files, group_id=group_idx)
        self._selection = SelectionState(group_idx=group_idx, member_indices=valid_members)
        self._update_viewer()
        self._apply_selection_overlay()

    # ── Viewer: display ───────────────────────────────────────────────────────

    def _update_viewer(self) -> None:
        """Refresh all viewer widgets (nav buttons, labels, image) for the current file."""
        v = self._viewer
        if not v.files:
            return

        path = v.current_path
        is_video = Path(path).suffix.lower() in VIDEO_EXTS

        self.prev_btn.config(state="normal" if v.is_multi else "disabled")
        self.next_btn.config(state="normal" if v.is_multi else "disabled")
        self.nav_lbl.config(text=f"{v.index + 1} / {len(v.files)}" if v.is_multi else "")

        self.play_btn.config(state="normal" if is_video else "disabled")
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

    def _display_image(
        self, img: Image.Image | None, path: str, type_line: str, meta: str
    ) -> None:
        """Apply a loaded image to the canvas (called on the main thread via ``after``).

        Silently discards the result if the viewer has navigated to a different
        file while the background load was in flight.

        Args:
            img: Decoded ``PIL.Image``, or ``None`` if loading failed.
            path: The file path that was loaded (used for staleness check).
            type_line: First info line (e.g. ``"IMAGE · PNG"``).
            meta: Second info line (dimensions, size, duration).
        """
        if not self._viewer.files or self._viewer.current_path != path:
            return  # navigated away before load finished

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
            return  # not yet mapped

        if self._current_img is None:
            # Contextual placeholder for unloadable files
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
        self._thumb_ref = photo  # prevent GC
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
        for btn in (self.prev_btn, self.next_btn, self.play_btn, self.open_btn, self.folder_btn):
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

    # ── Viewer: actions ───────────────────────────────────────────────────────

    def _play_media(self) -> None:
        """Open the current file in the system default media player."""
        if path := self._viewer.current_path:
            log.info("opening in system player: %s", path)
            system_open(path)

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
