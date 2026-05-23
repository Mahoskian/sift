from __future__ import annotations

import json
import logging
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from sift_viz import widgets
from sift_viz.cache import CacheManager
from sift_viz.constants import (
    ACCENT_COLOR,
    BG_COLOR,
    DIM_COLOR,
    HAS_FFMPEG,
    PANEL_BG,
    TEXT_COLOR,
    THUMB_H,
    THUMB_W,
    VIDEO_EXTS,
    VIEWER_W,
)
from sift_viz.media import (
    find_sift_binary,
    fit_image,
    fmt_ago,
    fmt_duration,
    fmt_hash_stats,
    get_file_info,
    load_media,
    probe_video_info,
    system_open,
)
from sift_viz.models import (
    ClusterGroup,
    ClusterResult,
    HashSettings,
    ProjectionResult,
    SelectionState,
    ViewerState,
)
from sift_viz.pipeline import PipelineRunner
from sift_viz.player import VideoPlayer
from sift_viz.scatter import ScatterController

log = logging.getLogger("sift-viz")


class SiftViz(tk.Tk):
    """Main application window for the sift interactive visualizer.

    Arranges three resizable panels (controls | scatter plot | media viewer)
    inside a tk.PanedWindow. Background threads communicate back to the UI
    via a queue.Queue polled every 50 ms with after().
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("sift viz")
        self.minsize(1100, 620)
        self.configure(bg=BG_COLOR)

        binary = find_sift_binary()
        if binary:
            log.info("sift binary: %s", binary)
        else:
            log.warning("sift binary not found — build the project first")

        self.tmpdir = tempfile.mkdtemp(prefix="sift_viz_")
        self._q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._cache = CacheManager()

        if binary:
            self._pipeline = PipelineRunner(binary, Path(self.tmpdir), self._q)
        else:
            self._pipeline = None  # type: ignore[assignment]

        self._projection_2d: ProjectionResult | None = None
        self._projection_3d: ProjectionResult | None = None
        self._cluster_data: ClusterResult | None = None

        self._running = False
        self._cluster_timer: str | None = None
        self._proj_timer: str | None = None
        self._poll_id: str | None = None
        self._loading_from_cache = False

        self._hash_start = 0.0
        self._hash_n = 0
        self._hash_total = 0
        self._proj_start = 0.0
        self._proj_running = False
        self._proj_n = 0
        self._proj_total = 0
        self._cluster_start = 0.0
        self._cluster_running = False
        self._cluster_n = 0
        self._cluster_total = 0

        self._viewer = ViewerState()
        self._selection = SelectionState()
        self._current_img: Image.Image | None = None
        self._thumb_ref: Any = None
        self._video_player: VideoPlayer | None = None  # set in _build_viewer_panel

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

        if not binary:
            self._set_status("sift binary not found — build with: cmake --build build/", error=True)
            self.run_btn.config(state="disabled")

    def _on_close(self) -> None:
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
        return self.dims_var.get() == "3"

    def _settings_from_ui(self) -> HashSettings:
        """Build a HashSettings model from current tk var values."""
        return HashSettings(
            algo=self.algo_var.get(),  # type: ignore[arg-type]
            hash_size=self.hash_size_var.get(),
            media=self.media_var.get(),  # type: ignore[arg-type]
            frames=self.frames_var.get(),
            proj=self.proj_var.get(),  # type: ignore[arg-type]
            dims=self.dims_var.get(),  # type: ignore[arg-type]
            perplexity=self.perplexity_var.get(),
            iterations=self.iterations_var.get(),
            cluster_method=self.cluster_method_var.get(),  # type: ignore[arg-type]
            threshold=self.threshold_var.get(),
            cut_height=self.cut_height_var.get(),
            linkage=self.linkage_var.get(),  # type: ignore[arg-type]
            min_group=self.min_group_var.get(),
            min_filter=self.min_filter_var.get(),
        )

    def _restore_settings(self, settings: HashSettings) -> None:
        """Restore tk vars from a HashSettings model (e.g. from cache)."""
        self.algo_var.set(settings.algo)
        self.hash_size_var.set(settings.hash_size)
        self.media_var.set(settings.media)
        self.frames_var.set(settings.frames)
        self.proj_var.set(settings.proj)
        self.dims_var.set(settings.dims)
        self.perplexity_var.set(settings.perplexity)
        self.iterations_var.set(settings.iterations)
        self.cluster_method_var.set(settings.cluster_method)
        self.threshold_var.set(settings.threshold)
        self.cut_height_var.set(settings.cut_height)
        self.linkage_var.set(settings.linkage)
        self.min_group_var.set(settings.min_group)
        self.min_filter_var.set(settings.min_filter)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
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

    def _build_left_panel(self, ctrl: tk.Frame) -> None:
        ctrl.columnconfigure(0, weight=1)
        r = 0

        widgets.section_label(ctrl, r, "INPUT")
        r += 1
        ff = tk.Frame(ctrl, bg=PANEL_BG)
        ff.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 4))
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
        widgets.flat_btn(ff, "Browse", self._browse).grid(row=0, column=1, padx=(4, 0))

        widgets.section_label(ctrl, r, "MEDIA TYPE")
        r += 1
        mf = tk.Frame(ctrl, bg=PANEL_BG)
        mf.grid(row=r, column=0, sticky="ew", padx=12)
        r += 1
        for i, (value, label) in enumerate(
            (("images", "Images only"), ("videos", "Videos only"), ("all", "Both"))
        ):
            tk.Radiobutton(
                mf,
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
            ).grid(row=i, column=0, sticky="w")

        self.frames_frame = tk.Frame(ctrl, bg=PANEL_BG)
        self.frames_frame.grid(row=r, column=0, sticky="ew")
        r += 1
        self.frames_frame.columnconfigure(0, weight=1)
        tk.Label(
            self.frames_frame,
            text="Frames per video",
            bg=PANEL_BG,
            fg=DIM_COLOR,
            font=("", 8),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(4, 0))
        widgets.slider(
            self.frames_frame,
            1,
            self.frames_var,
            1,
            32,
            label_fn=lambda v: f"{v} frame{'s' if v != 1 else ''}",
        )

        widgets.divider(ctrl, r)
        r += 1

        widgets.section_label(ctrl, r, "HASH ALGORITHM")
        r += 1
        for algo in ("dhash", "phash", "whash"):
            widgets.radio(ctrl, r, algo, self.algo_var, algo)
            r += 1

        widgets.section_label(ctrl, r, "HASH SIZE")
        r += 1
        widgets.slider(
            ctrl,
            r,
            self.hash_size_var,
            4,
            32,
            label_fn=lambda v: f"{v}×{v} · {v * v} bits",
        )
        r += 1

        self.run_btn = widgets.run_button(ctrl, self._on_run)
        self.run_btn.grid(row=r, column=0, sticky="ew", padx=12, pady=(6, 4))
        r += 1

        self.progress = ttk.Progressbar(ctrl, mode="determinate", maximum=100, value=0, length=100)
        self.progress.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 2))
        r += 1
        self.hash_stats_lbl = tk.Label(
            ctrl, text="", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8), anchor="w"
        )
        self.hash_stats_lbl.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 4))
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

        widgets.divider(ctrl, r)
        r += 1
        widgets.section_label(ctrl, r, "PROJECTION")
        r += 1

        self.proj_progress = ttk.Progressbar(
            ctrl, mode="determinate", maximum=100, value=0, length=100
        )
        self.proj_progress.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 2))
        r += 1
        self.proj_stats_lbl = tk.Label(
            ctrl, text="", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8), anchor="w"
        )
        self.proj_stats_lbl.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 4))
        r += 1

        for proj, label in (("pca", "PCA"), ("tsne", "t-SNE")):
            widgets.radio(ctrl, r, label, self.proj_var, proj, command=self._on_proj_method_change)
            r += 1

        self.tsne_frame = tk.Frame(ctrl, bg=PANEL_BG)
        self.tsne_frame.grid(row=r, column=0, sticky="ew")
        r += 1
        self.tsne_frame.columnconfigure(0, weight=1)
        widgets.labeled_slider(
            self.tsne_frame,
            0,
            "Perplexity",
            self.perplexity_var,
            5,
            100,
            on_change=self._schedule_reprojection,
        )
        widgets.labeled_slider(
            self.tsne_frame,
            2,
            "Iterations",
            self.iterations_var,
            250,
            3000,
            on_change=self._schedule_reprojection,
        )

        widgets.divider(ctrl, r)
        r += 1
        widgets.section_label(ctrl, r, "CLUSTERING")
        r += 1

        self.cluster_progress = ttk.Progressbar(
            ctrl, mode="determinate", maximum=100, value=0, length=100
        )
        self.cluster_progress.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 2))
        r += 1
        self.cluster_stats_lbl = tk.Label(
            ctrl, text="", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8), anchor="w"
        )
        self.cluster_stats_lbl.grid(row=r, column=0, sticky="ew", padx=12, pady=(0, 4))
        r += 1

        for method in ("threshold", "hierarchical", "hdbscan"):
            widgets.radio(
                ctrl, r, method, self.cluster_method_var, method, command=self._on_method_change
            )
            r += 1

        self.cluster_params_frame = tk.Frame(ctrl, bg=PANEL_BG)
        self.cluster_params_frame.grid(row=r, column=0, sticky="ew")
        r += 1
        self.cluster_params_frame.columnconfigure(0, weight=1)

        widgets.section_label(ctrl, r, "MIN GROUP SIZE")
        r += 1
        widgets.slider(ctrl, r, self.min_filter_var, 2, 20, on_change=self._on_slider, label_fn=str)
        r += 1

        widgets.divider(ctrl, r)
        r += 1
        widgets.section_label(ctrl, r, "VIEW DIMENSIONS")
        r += 1
        dims_row = tk.Frame(ctrl, bg=PANEL_BG)
        dims_row.grid(row=r, column=0, sticky="w", padx=12, pady=(0, 8))
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

    def _build_scatter_panel(self, pf: tk.Frame) -> None:
        pf.columnconfigure(0, weight=1)
        pf.rowconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(7, 6))
        self.fig.patch.set_facecolor(BG_COLOR)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas_widget.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._scatter = ScatterController(self.fig, self.canvas_widget)
        self.canvas_widget.mpl_connect("button_press_event", self._on_mpl_click)
        self.canvas_widget.draw()

    def _build_viewer_panel(self, v: tk.Frame) -> None:
        v.columnconfigure(0, weight=1)
        v.rowconfigure(1, weight=1)

        widgets.section_label(v, 0, "MEDIA VIEWER")

        self.img_canvas = tk.Canvas(v, bg="#111111", highlightthickness=0)
        self.img_canvas.grid(row=1, column=0, padx=12, pady=(4, 0), sticky="nsew")
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)
        self._draw_placeholder()
        self._video_player = VideoPlayer(self.img_canvas, self._on_video_end)

        self.viewer_name_lbl = widgets.viewer_label(v, 2, font=("", 9, "bold"), fg=TEXT_COLOR)
        self.viewer_type_lbl = widgets.viewer_label(v, 3)
        self.viewer_meta_lbl = widgets.viewer_label(v, 4)
        self.viewer_cluster_lbl = widgets.viewer_label(
            v,
            5,
            text="Click a point or cluster\nto preview media",
            fg="#444",
        )

        # ── Seek / scrub bar ──────────────────────────────────────────────────
        self.seek_frame = tk.Frame(v, bg=PANEL_BG)
        self.seek_frame.columnconfigure(1, weight=1)
        self.seek_frame.grid(row=6, column=0, sticky="ew", padx=12, pady=(6, 0))
        self.seek_frame.grid_remove()

        self.seek_pos_lbl = tk.Label(
            self.seek_frame,
            text="0:00",
            bg=PANEL_BG,
            fg=DIM_COLOR,
            font=("", 7),
            width=5,
            anchor="w",
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
            self.seek_frame,
            text="0:00",
            bg=PANEL_BG,
            fg=DIM_COLOR,
            font=("", 7),
            width=5,
            anchor="e",
        )
        self.seek_dur_lbl.grid(row=0, column=2, sticky="e")

        self._user_scrubbing = False
        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)

        # ── Playback controls ─────────────────────────────────────────────────
        pb = tk.Frame(v, bg=PANEL_BG)
        pb.grid(row=7, column=0, pady=(6, 0))
        self.skip_back_btn = widgets.icon_btn(pb, "⏮", self._skip_back, col=0)
        self.play_pause_btn = widgets.action_btn(pb, "▶  Play", self._toggle_play_pause, col=1)
        self.skip_fwd_btn = widgets.icon_btn(pb, "⏭", self._skip_forward, col=2)
        for b in (self.skip_back_btn, self.play_pause_btn, self.skip_fwd_btn):
            b.config(state="disabled")

        # ── Group navigation + file actions ───────────────────────────────────
        act = tk.Frame(v, bg=PANEL_BG)
        act.grid(row=8, column=0, pady=(8, 0))
        self.prev_btn = widgets.icon_btn(act, "◀", self._viewer_prev, col=0)
        self.nav_lbl = tk.Label(act, text="", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8))
        self.nav_lbl.grid(row=0, column=1, padx=2)
        self.delete_btn = widgets.action_btn(act, "✗  Del", self._delete_current_media, col=2)
        self.delete_btn.config(bg="#3a1a1a", activebackground="#5a2020")
        self.open_btn = widgets.action_btn(act, "↗  Open", self._open_media, col=3)
        self.folder_btn = widgets.action_btn(act, "⌂  Folder", self._open_in_folder, col=4)
        self.next_btn = widgets.icon_btn(act, "▶", self._viewer_next, col=5)
        for b in (
            self.prev_btn,
            self.next_btn,
            self.delete_btn,
            self.open_btn,
            self.folder_btn,
        ):
            b.config(state="disabled")

        self.viewer_path_lbl = widgets.viewer_label(v, 9, fg="#383838", font=("", 7), pady=(6, 12))

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _save_cache(self) -> None:
        folder = self.folder_var.get().strip()
        if not folder or self._projection_2d is None or self._cluster_data is None:
            return
        self._cache.save(
            folder=folder,
            settings=self._settings_from_ui(),
            proj_2d=self._projection_2d,
            proj_3d=self._projection_3d,
            clusters=self._cluster_data,
            hashes_src=Path(self.tmpdir) / "hashes.json",
        )

    def _load_cache(self, folder: str) -> bool:
        meta = self._cache.load(folder)
        if meta is None:
            return False
        results = self._cache.load_results(folder)
        if results is None:
            return False

        proj_2d, proj_3d, clusters = results

        # Copy hashes to tmpdir so hot-swap projection/cluster still works
        self._cache.copy_hashes_to(folder, Path(self.tmpdir) / "hashes.json")

        self._restore_settings(meta.settings)
        self._update_tsne_visibility()
        self._update_frames_visibility()
        self._rebuild_cluster_params()

        self._projection_2d = proj_2d
        self._projection_3d = proj_3d
        self._loading_from_cache = True
        self._on_cluster_done(clusters)
        self._loading_from_cache = False

        saved_ago = time.time() - meta.saved_at
        self._set_status(f"Loaded from cache · saved {fmt_ago(saved_ago)} ago")
        log.info("cache loaded (%s ago)", fmt_ago(saved_ago))
        return True

    # ── Control callbacks ─────────────────────────────────────────────────────

    def _browse(self) -> None:
        if d := filedialog.askdirectory(title="Select image folder"):
            self.folder_var.set(d)
            self._projection_2d = None
            self._projection_3d = None
            self._cluster_data = None
            self._clear_viewer()
            self._scatter.rebuild_axes(self._is_3d())
            self._scatter.style_idle()
            self.canvas_widget.draw()
            self.hash_stats_lbl.config(text="")
            self.proj_stats_lbl.config(text="")
            self.cluster_stats_lbl.config(text="")
            self.progress.config(mode="determinate", value=0)
            self.proj_progress.config(mode="determinate", value=0)
            self.cluster_progress.stop()
            self.cluster_progress.config(mode="determinate", value=0)
            if not self._load_cache(d):
                self._set_status("No cache found — click Run to process")

    def _update_tsne_visibility(self) -> None:
        if self.proj_var.get() == "tsne":
            self.tsne_frame.grid()
        else:
            self.tsne_frame.grid_remove()

    def _update_frames_visibility(self) -> None:
        if self.media_var.get() in ("videos", "all"):
            self.frames_frame.grid()
        else:
            self.frames_frame.grid_remove()

    def _on_media_type_change(self) -> None:
        self._update_frames_visibility()

    def _on_dims_change(self) -> None:
        if self._projection_2d is not None or self._projection_3d is not None:
            self._redraw()

    def _on_method_change(self) -> None:
        self._rebuild_cluster_params()
        self._schedule_recluster()

    def _rebuild_cluster_params(self) -> None:
        """Destroy and recreate cluster parameter widgets for the current method."""
        for w in self.cluster_params_frame.winfo_children():
            w.destroy()
        f = self.cluster_params_frame
        method = self.cluster_method_var.get()

        if method == "threshold":
            tk.Label(f, text="Threshold", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)).grid(
                row=0, column=0, sticky="w", padx=12, pady=(6, 0)
            )
            widgets.slider(f, 1, self.threshold_var, 1, 100, on_change=self._on_slider)

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
            tk.Label(f, text="Cut Height", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)).grid(
                row=4, column=0, sticky="w", padx=12, pady=(6, 0)
            )
            widgets.slider(f, 5, self.cut_height_var, 1, 100, on_change=self._on_slider)

        elif method == "hdbscan":
            tk.Label(f, text="Min Cluster Size", bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)).grid(
                row=0, column=0, sticky="w", padx=12, pady=(6, 0)
            )
            widgets.slider(f, 1, self.min_group_var, 2, 30, on_change=self._on_slider)

    def _on_slider(self) -> None:
        self._schedule_recluster()

    def _set_status(self, msg: str, error: bool = False) -> None:
        self.status_lbl.config(text=msg, fg="#e05252" if error else DIM_COLOR)

    # ── Message queue poll ────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg["kind"]
                if kind == "status":
                    self._set_status(msg["text"], msg.get("error", False))
                elif kind == "hash_start":
                    self._hash_start = time.monotonic()
                    self._hash_n = 0
                    self._hash_total = 0
                    self.hash_stats_lbl.config(text="starting…")
                elif kind == "progress":
                    n, total = msg["value"], msg["total"]
                    if total > 0:
                        self.progress.config(mode="determinate", value=(n / total) * 100)
                        self._hash_n = n
                        self._hash_total = total
                        elapsed = time.monotonic() - self._hash_start if self._hash_start else 0.0
                        self.hash_stats_lbl.config(text=fmt_hash_stats(n, total, elapsed))
                elif kind == "progress_indeterminate":
                    self.progress.config(mode="indeterminate")
                    self.progress.start(12)
                elif kind == "proj_progress_indeterminate":
                    self._proj_start = time.monotonic()
                    self._proj_running = True
                    self._proj_n = 0
                    self._proj_total = 0
                    self.proj_progress.config(mode="indeterminate")
                    self.proj_progress.start(12)
                elif kind == "proj_progress":
                    n, total = msg["value"], msg["total"]
                    if total > 0:
                        self.proj_progress.stop()
                        self.proj_progress.config(mode="determinate", value=(n / total) * 100)
                        self._proj_n = n
                        self._proj_total = total
                        elapsed = time.monotonic() - self._proj_start if self._proj_start else 0.0
                        self.proj_stats_lbl.config(text=fmt_hash_stats(n, total, elapsed))
                elif kind == "cluster_progress":
                    n, total = msg["value"], msg["total"]
                    if total > 0:
                        self.cluster_progress.stop()
                        self.cluster_progress.config(mode="determinate", value=(n / total) * 100)
                        self._cluster_n = n
                        self._cluster_total = total
                        elapsed = (
                            time.monotonic() - self._cluster_start if self._cluster_start else 0.0
                        )
                        self.cluster_stats_lbl.config(text=fmt_hash_stats(n, total, elapsed))
                elif kind == "pipeline_done":
                    self._on_pipeline_done(msg["projection_2d"], msg["projection_3d"])
                elif kind == "projection_done":
                    self._on_projection_done(msg["projection_2d"], msg["projection_3d"])
                elif kind == "projection_error":
                    self._proj_running = False
                    self.proj_progress.stop()
                    self.proj_progress.config(mode="determinate", value=0)
                    self.proj_stats_lbl.config(text="error")
                    self._set_status(f"Projection error: {msg['text']}", error=True)
                elif kind == "cluster_done":
                    clusters_raw = msg["clusters"]
                    try:
                        clusters = ClusterResult.model_validate(clusters_raw)
                    except Exception:
                        clusters = ClusterResult()
                    self._on_cluster_done(clusters)
                elif kind == "cancelled":
                    self._running = False
                    self._proj_running = False
                    self._cluster_running = False
                    self._reset_run_button()
                    self._set_status("Cancelled.")
                elif kind == "cluster_error":
                    self._cluster_running = False
                    self.cluster_progress.stop()
                    self.cluster_progress.config(mode="determinate", value=0)
                    self.cluster_stats_lbl.config(text="error")
                    self._set_status(msg["text"], error=True)
                elif kind == "error":
                    self._proj_running = False
                    self._on_error(msg["text"])
        except queue.Empty:
            pass

        now = time.monotonic()
        if self._proj_running and self._proj_start and self._proj_total == 0:
            self.proj_stats_lbl.config(text=f"elapsed  {fmt_duration(now - self._proj_start)}")
        if self._cluster_running and self._cluster_start and self._cluster_total == 0:
            self.cluster_stats_lbl.config(
                text=f"elapsed  {fmt_duration(now - self._cluster_start)}"
            )

        self._poll_id = self.after(50, self._poll)

    # ── Pipeline orchestration ────────────────────────────────────────────────

    def _reset_run_button(self) -> None:
        has_binary = self._pipeline is not None
        self.run_btn.config(
            text="Run",
            bg=ACCENT_COLOR,
            activebackground="#3a8ee6",
            command=self._on_run,
            state="normal" if has_binary else "disabled",
        )
        self.progress.stop()
        self.progress.config(mode="determinate", value=0)
        self.proj_progress.stop()
        self.proj_progress.config(mode="determinate", value=0)

    def _on_cancel(self) -> None:
        if not self._running or self._pipeline is None:
            return
        self._pipeline.cancel()

    def _on_run(self) -> None:
        if self._running or self._pipeline is None:
            return
        folder = self.folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            self._set_status("Select a valid input folder first.", error=True)
            return

        settings = self._settings_from_ui()
        log.info(
            "starting pipeline — folder=%s  algo=%s  proj=%s  (2D+3D)",
            folder,
            settings.algo,
            settings.proj,
        )
        if self._proj_timer:
            self.after_cancel(self._proj_timer)
            self._proj_timer = None

        self._running = True
        self._projection_2d = None
        self._projection_3d = None
        self._cluster_data = None
        self._hash_start = 0.0
        self._hash_n = 0
        self._hash_total = 0
        self._proj_running = False
        self._proj_n = 0
        self._proj_total = 0
        self._cluster_running = False
        self._cluster_n = 0
        self._cluster_total = 0
        self.hash_stats_lbl.config(text="")
        self.proj_stats_lbl.config(text="")
        self.cluster_stats_lbl.config(text="")
        self.run_btn.config(
            text="Cancel",
            bg="#c0392b",
            activebackground="#a93226",
            command=self._on_cancel,
        )
        self.progress.config(mode="determinate", value=0)
        self._clear_viewer()
        self._scatter.rebuild_axes(self._is_3d())
        self._scatter.style_idle()
        self.canvas_widget.draw()
        self._pipeline.run_pipeline(settings, folder)

    def _on_pipeline_done(self, proj_2d_raw: dict[str, Any], proj_3d_raw: dict[str, Any]) -> None:
        self._running = False
        self._proj_running = False
        try:
            proj_2d = ProjectionResult.model_validate(proj_2d_raw)
            proj_3d = ProjectionResult.model_validate(proj_3d_raw)
        except Exception as exc:
            self._on_error(f"Invalid projection data: {exc}")
            return
        self._projection_2d = proj_2d
        self._projection_3d = proj_3d
        n = len(proj_2d.files)
        self._reset_run_button()
        if self._proj_start:
            self.proj_stats_lbl.config(
                text=f"done in  {fmt_duration(time.monotonic() - self._proj_start)}"
            )
        method = proj_2d.method.upper()
        msg = f"{method} done — {n} files, 2D+3D ready. Adjust clustering below."
        log.info(msg)
        self._set_status(msg)
        self._run_cluster()

    def _on_error(self, msg: str) -> None:
        self._running = False
        self._reset_run_button()
        self._set_status(f"Error: {msg}", error=True)

    # ── Projection (hot-swap) ─────────────────────────────────────────────────

    def _on_proj_method_change(self) -> None:
        self._update_tsne_visibility()
        self._schedule_reprojection()

    def _schedule_reprojection(self) -> None:
        if self._pipeline is None:
            return
        if not self._pipeline.hashes_path.exists() or self._running:
            return
        if self._proj_timer:
            self.after_cancel(self._proj_timer)
        self._proj_timer = self.after(350, self._run_projection)

    def _run_projection(self) -> None:
        self._proj_timer = None
        if self._pipeline is None or self._running:
            return
        if not self._pipeline.hashes_path.exists():
            return
        self._proj_start = time.monotonic()
        self._proj_running = True
        self._proj_n = 0
        self._proj_total = 0
        self.proj_stats_lbl.config(text="")
        self.proj_progress.config(mode="indeterminate")
        self.proj_progress.start(12)
        self._pipeline.run_projection(self._settings_from_ui())

    def _on_projection_done(self, proj_2d_raw: dict[str, Any], proj_3d_raw: dict[str, Any]) -> None:
        if self._running:
            return
        try:
            proj_2d = ProjectionResult.model_validate(proj_2d_raw)
            proj_3d = ProjectionResult.model_validate(proj_3d_raw)
        except Exception as exc:
            self._set_status(f"Projection error: {exc}", error=True)
            return
        self._proj_running = False
        self._projection_2d = proj_2d
        self._projection_3d = proj_3d
        self.proj_progress.stop()
        self.proj_progress.config(mode="determinate", value=0)
        if self._proj_start:
            self.proj_stats_lbl.config(
                text=f"done in  {fmt_duration(time.monotonic() - self._proj_start)}"
            )
        self._redraw()

    # ── Clustering ────────────────────────────────────────────────────────────

    def _schedule_recluster(self) -> None:
        if self._projection_2d is None:
            return
        if self._cluster_timer:
            self.after_cancel(self._cluster_timer)
        self._cluster_timer = self.after(350, self._run_cluster)

    def _run_cluster(self) -> None:
        if self._projection_2d is None or self._pipeline is None:
            return
        self._cluster_start = time.monotonic()
        self._cluster_running = True
        self._cluster_n = 0
        self._cluster_total = 0
        self.cluster_stats_lbl.config(text="")
        self.cluster_progress.config(mode="indeterminate")
        self.cluster_progress.start(12)
        self._pipeline.run_cluster(self._settings_from_ui())

    def _on_cluster_done(self, clusters: ClusterResult) -> None:
        self._cluster_running = False
        self.cluster_progress.stop()
        self.cluster_progress.config(mode="determinate", value=0)
        self._cluster_data = clusters
        n_groups = len(clusters.groups)
        n_ungrouped = len(clusters.ungrouped)
        elapsed_str = (
            f"done in  {fmt_duration(time.monotonic() - self._cluster_start)}  ·  "
            if self._cluster_start
            else ""
        )
        self.cluster_stats_lbl.config(
            text=f"{elapsed_str}{n_groups} groups  ·  {n_ungrouped} ungrouped"
        )
        log.info("cluster done — %d groups, %d ungrouped", n_groups, n_ungrouped)
        self._set_status(f"{n_groups} groups · {n_ungrouped} ungrouped")
        self._redraw()
        if not self._loading_from_cache:
            self._save_cache()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        """Replot scatter points coloured by cluster membership."""
        is_3d = self._is_3d()
        proj: ProjectionResult | None
        if is_3d and self._projection_3d is not None:
            proj = self._projection_3d
        else:
            proj = self._projection_2d or self._projection_3d
        if proj is None:
            return

        self._scatter.redraw(proj, self._cluster_data, is_3d)
        self._scatter.apply_selection_overlay(self._selection, self._viewer)

    # ── Click detection ───────────────────────────────────────────────────────

    def _on_mpl_click(self, event: Any) -> None:
        if event.inaxes is not self._scatter.ax or self._projection_2d is None:
            return
        proj = self._projection_3d if self._is_3d() else self._projection_2d
        if proj is None:
            proj = self._projection_2d
        nearest = self._scatter.hit_test(event, proj)
        if nearest is not None:
            self._select_nearest(nearest)

    def _select_nearest(self, nearest: int) -> None:
        gi = self._scatter.point_group.get(nearest)
        if gi is not None:
            self._show_group(gi, start_point=nearest)
        else:
            self._show_single(nearest)

    # ── Viewer: selection ─────────────────────────────────────────────────────

    def _show_single(self, point_idx: int) -> None:
        files = self._scatter.point_files
        log.debug("selected point %d: %s", point_idx, files[point_idx])
        self._viewer = ViewerState(files=[files[point_idx]])
        self._selection = SelectionState(point_idx=point_idx)
        self._update_viewer()
        self._scatter.apply_selection_overlay(self._selection, self._viewer)

    def _show_group(self, group_idx: int, start_point: int | None = None) -> None:
        if self._cluster_data is None:
            return
        groups = self._cluster_data.groups
        if group_idx >= len(groups):
            return
        members = groups[group_idx].members
        files = self._scatter.point_files
        valid_members = [i for i in members if 0 <= i < len(files)]
        group_files = [files[i] for i in valid_members]
        if not group_files:
            return
        initial_index = 0
        if start_point is not None and start_point in valid_members:
            initial_index = valid_members.index(start_point)
        log.debug(
            "selected group %d (%d files), starting at member %d",
            group_idx,
            len(group_files),
            initial_index,
        )
        self._viewer = ViewerState(files=group_files, index=initial_index, group_id=group_idx)
        self._selection = SelectionState(group_idx=group_idx, member_indices=valid_members)
        self._update_viewer()
        self._scatter.apply_selection_overlay(self._selection, self._viewer)

    # ── Viewer: display ───────────────────────────────────────────────────────

    def _update_viewer(self) -> None:
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
        assert path is not None
        is_video = Path(path).suffix.lower() in VIDEO_EXTS

        self.prev_btn.config(state="normal" if v.is_multi else "disabled")
        self.next_btn.config(state="normal" if v.is_multi else "disabled")
        self.nav_lbl.config(text=f"{v.index + 1} / {len(v.files)}" if v.is_multi else "")
        self.play_pause_btn.config(state="normal" if is_video else "disabled")
        self.skip_back_btn.config(state="disabled")
        self.skip_fwd_btn.config(state="disabled")
        self.delete_btn.config(state="normal")
        self.open_btn.config(state="normal")
        self.folder_btn.config(state="normal")

        self.viewer_name_lbl.config(text=Path(path).name)
        self.viewer_type_lbl.config(text="")
        self.viewer_meta_lbl.config(text="")
        self.viewer_cluster_lbl.config(
            text=(
                f"Cluster {v.group_id + 1}  ·  {len(v.files)} files"
                if v.group_id is not None
                else ""
            )
        )
        self.viewer_path_lbl.config(text=str(Path(path).parent))

        threading.Thread(target=self._load_and_display, args=(path,), daemon=True).start()

    def _load_and_display(self, path: str) -> None:
        img = load_media(path)
        type_line, meta = get_file_info(path)
        self.after(0, self._display_image, img, path, type_line, meta)

    def _display_image(
        self,
        img: Image.Image | None,
        path: str,
        type_line: str,
        meta: str,
    ) -> None:
        if not self._viewer.files or self._viewer.current_path != path:
            return
        self.viewer_type_lbl.config(text=type_line)
        self.viewer_meta_lbl.config(text=meta)
        self._current_img = img
        self._render_image()

    def _render_image(self) -> None:
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
                cw // 2,
                ch // 2,
                text=msg,
                fill=DIM_COLOR,
                font=("", 12),
                justify="center",
            )
            return

        pad = 8
        fitted = fit_image(self._current_img.copy(), cw - pad, ch - pad)
        iw, ih = fitted.size
        photo = ImageTk.PhotoImage(fitted)
        self._thumb_ref = photo
        self.img_canvas.create_image((cw - iw) // 2, (ch - ih) // 2, anchor="nw", image=photo)

    def _on_canvas_resize(self, _: Any = None) -> None:
        self._render_image()

    def _draw_placeholder(self) -> None:
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
            self.prev_btn,
            self.next_btn,
            self.play_pause_btn,
            self.skip_back_btn,
            self.skip_fwd_btn,
            self.delete_btn,
            self.open_btn,
            self.folder_btn,
        ):
            btn.config(state="disabled")

    # ── Viewer: navigation ────────────────────────────────────────────────────

    def _viewer_prev(self) -> None:
        if not self._viewer.files:
            return
        self._viewer.index = (self._viewer.index - 1) % len(self._viewer.files)
        self._update_viewer()
        self._scatter.apply_selection_overlay(self._selection, self._viewer)

    def _viewer_next(self) -> None:
        if not self._viewer.files:
            return
        self._viewer.index = (self._viewer.index + 1) % len(self._viewer.files)
        self._update_viewer()
        self._scatter.apply_selection_overlay(self._selection, self._viewer)

    # ── Viewer: playback controls ─────────────────────────────────────────────

    def _toggle_play_pause(self) -> None:
        player = self._video_player
        if not player:
            return

        if player.is_playing:
            player.pause()
            self.play_pause_btn.config(text="▶  Play")

        elif player.is_paused:
            player.resume()
            self.play_pause_btn.config(text="⏸  Pause")

        else:
            path = self._viewer.current_path
            if not path:
                return
            if not HAS_FFMPEG:
                log.warning("ffmpeg not found — falling back to system open")
                system_open(path)
                return
            log.info("starting inline playback: %s", path)
            info = probe_video_info(path)
            ok = player.play(path, info, position_cb=self._on_position_update)
            if not ok:
                log.warning("could not start playback for %s", path)
                return
            dur = player.duration
            self.seek_scale.config(to=max(1.0, dur))
            self.seek_var.set(0.0)
            self.seek_pos_lbl.config(text="0:00")
            self.seek_dur_lbl.config(text=fmt_duration(dur))
            self.seek_frame.grid()
            self.play_pause_btn.config(text="⏸  Pause")
            self.skip_back_btn.config(state="normal")
            self.skip_fwd_btn.config(state="normal")

    def _skip_back(self) -> None:
        player = self._video_player
        if not player:
            return
        new_pos = max(0.0, player.position - 1.0)
        if player.is_paused:
            player.seek_paused(new_pos)
        elif player.is_playing:
            player.seek(new_pos)
            self.play_pause_btn.config(text="⏸  Pause")

    def _skip_forward(self) -> None:
        player = self._video_player
        if not player:
            return
        dur = player.duration
        new_pos = player.position + 1.0
        if dur > 0:
            new_pos = min(new_pos, dur)
        if player.is_paused:
            player.seek_paused(new_pos)
        elif player.is_playing:
            player.seek(new_pos)
            self.play_pause_btn.config(text="⏸  Pause")

    def _on_seek_press(self, _: Any) -> None:
        self._user_scrubbing = True

    def _on_seek_release(self, _: Any) -> None:
        self._user_scrubbing = False
        player = self._video_player
        if not player:
            return
        t = self.seek_var.get()
        if player.is_paused:
            player.seek_paused(t)
        elif player.is_playing:
            player.seek(t)
            self.play_pause_btn.config(text="⏸  Pause")

    def _on_seek_drag(self, val: Any) -> None:
        self.seek_pos_lbl.config(text=fmt_duration(float(val)))

    def _on_position_update(self, pos: float) -> None:
        if not self._user_scrubbing:
            self.seek_var.set(pos)
            self.seek_pos_lbl.config(text=fmt_duration(pos))

    def _on_video_end(self) -> None:
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

    def _delete_current_media(self) -> None:
        path = self._viewer.current_path
        if not path:
            return

        player = self._video_player
        if player and (player.is_playing or player.is_paused):
            player.stop()
            self.play_pause_btn.config(text="▶  Play")
            self.seek_frame.grid_remove()

        try:
            try:
                import send2trash  # type: ignore[import]

                send2trash.send2trash(path)
            except ImportError:
                import subprocess

                result = subprocess.run(["gio", "trash", path], capture_output=True, timeout=5)
                if result.returncode != 0:
                    msg = result.stderr.decode().strip() or "gio trash failed"
                    raise RuntimeError(msg) from None
            log.info("trashed: %s", path)
        except Exception as exc:
            log.warning("could not trash %s: %s", path, exc)
            self._set_status(f"Could not delete: {exc}", error=True)
            return

        if self._projection_2d is None:
            return
        proj_files = self._projection_2d.files
        if path not in proj_files:
            return
        idx = proj_files.index(path)

        # Remove from both projections in-memory
        for proj in (self._projection_2d, self._projection_3d):
            if proj is None:
                continue
            if idx < len(proj.files):
                proj.files.pop(idx)
            if idx < len(proj.points):
                proj.points.pop(idx)

        self._remove_file_from_cluster_data(idx)

        # Patch hashes JSON on disk so hot-swap stays consistent
        hashes_path = Path(self.tmpdir) / "hashes.json"
        if hashes_path.exists():
            try:
                with open(hashes_path) as fh:
                    hd = json.load(fh)
                hd.get("files", {}).pop(path, None)
                with open(hashes_path, "w") as fh:
                    json.dump(hd, fh)
            except Exception as exc:
                log.warning("could not patch hashes JSON: %s", exc)

        v = self._viewer
        if path in v.files:
            removed_at = v.files.index(path)
            v.files.remove(path)
            if not v.files:
                self._clear_viewer()
                self._selection.clear()
            else:
                v.index = min(removed_at, len(v.files) - 1)
                sel = self._selection
                if sel.member_indices:
                    sel.member_indices = [
                        m - (1 if m > idx else 0) for m in sel.member_indices if m != idx
                    ]
                self._update_viewer()

        if self._projection_2d and self._projection_2d.files:
            self._redraw()
        else:
            self._scatter.rebuild_axes(self._is_3d())
            self._scatter.style_idle()
            self.canvas_widget.draw()

        self._save_cache()

    def _remove_file_from_cluster_data(self, idx: int) -> None:
        """Remove file at *idx* from cluster data, shifting subsequent indices."""
        cd = self._cluster_data
        if cd is None:
            return

        if idx < len(cd.files):
            cd.files.pop(idx)

        new_groups: list[ClusterGroup] = []
        for group in cd.groups:
            new_members = [m - (1 if m > idx else 0) for m in group.members if m != idx]
            if new_members:
                new_member_files = [cd.files[m] for m in new_members if m < len(cd.files)]
                new_groups.append(ClusterGroup(members=new_members, member_files=new_member_files))
        cd.groups = new_groups
        cd.ungrouped = [m - (1 if m > idx else 0) for m in cd.ungrouped if m != idx]

    def _open_media(self) -> None:
        if path := self._viewer.current_path:
            log.info("opening: %s", path)
            system_open(path)

    def _open_in_folder(self) -> None:
        if path := self._viewer.current_path:
            log.info("revealing in folder: %s", path)
            system_open(path, reveal=True)
