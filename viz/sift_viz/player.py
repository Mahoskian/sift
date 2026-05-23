from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
import tkinter as tk
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageTk

from sift_viz.constants import HAS_FFPLAY
from sift_viz.models import VideoInfo

log = logging.getLogger("sift-viz")


class VideoPlayer:
    """Renders ffmpeg-decoded video frames onto a tkinter Canvas with audio.

    The video ffmpeg process pipes raw RGB frames into a queue consumed by
    the main-thread tick loop.  Audio is played via a separate ffplay process
    which handles device output independently.

    Frame timing uses elapsed wall-clock time so the video self-corrects for
    drift.  A/V sync is wall-clock based: both ffmpeg and ffplay are started
    from the same seek offset at nearly the same instant.
    """

    def __init__(self, canvas: tk.Canvas, on_end: Callable[[], None]) -> None:
        self._canvas = canvas
        self._on_end = on_end

        self._path = ""
        self._duration = 0.0
        self._has_audio = False
        self._frame_w = 0
        self._frame_h = 0
        self._fps = 25.0
        self._position_cb: Callable[[float], None] | None = None

        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._frame_q: queue.Queue[tuple[int, bytes | None]] = queue.Queue(maxsize=16)
        self._frames_shown = 0
        self._seek_pos = 0.0
        self._play_start = 0.0

        self._audio_proc: subprocess.Popen[bytes] | None = None

        self._active = False
        self._paused = False
        self._pause_pos = 0.0

        self._after_id: str | None = None
        self._photo: Any = None  # ImageTk.PhotoImage — held to prevent GC

    @property
    def is_playing(self) -> bool:
        return self._active

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def position(self) -> float:
        if self._paused:
            return self._pause_pos
        if not self._active:
            return 0.0
        return self._seek_pos + (time.monotonic() - self._play_start)

    @property
    def duration(self) -> float:
        return self._duration

    def play(
        self,
        path: str,
        info: VideoInfo,
        seek: float = 0.0,
        position_cb: Callable[[float], None] | None = None,
    ) -> bool:
        """Start playback of *path* using pre-probed *info*.

        Returns False if the video info is invalid.
        """
        self.stop()

        if not info.is_valid:
            return False

        cw = max(self._canvas.winfo_width(), 1)
        ch = max(self._canvas.winfo_height(), 1)
        scale = min((cw - 8) / info.width, (ch - 8) / info.height)
        self._frame_w = max(1, int(info.width * scale))
        self._frame_h = max(1, int(info.height * scale))
        self._fps = info.fps
        self._path = path
        self._duration = info.duration
        self._has_audio = info.has_audio
        self._position_cb = position_cb

        self._start_at(seek)
        return True

    def pause(self) -> None:
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
        if not self._paused:
            return
        self._paused = False
        self._start_at(self._pause_pos)

    def seek(self, t: float) -> None:
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

    def seek_paused(self, t: float) -> None:
        """Seek to *t* and show a still frame without resuming playback."""
        if not self._path or not self._paused:
            return
        if self._duration > 0:
            t = max(0.0, min(t, self._duration))
        else:
            t = max(0.0, t)
        self._pause_pos = t
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._drain_queue()
        seek_args = ["-ss", f"{t:.3f}"] if t > 0 else []
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                *seek_args,
                "-i",
                self._path,
                "-vf",
                f"scale={self._frame_w}:{self._frame_h}:flags=fast_bilinear",
                "-vframes",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        frame_bytes = self._frame_w * self._frame_h * 3
        threading.Thread(
            target=self._fetch_preview_frame,
            args=(self._proc, frame_bytes, t),
            daemon=True,
        ).start()

    def stop(self) -> None:
        self._active = False
        self._paused = False
        self._cancel_tick()
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self._drain_queue()
        self._teardown_audio()

    def _start_at(self, t: float) -> None:
        if self._duration > 0:
            t = max(0.0, min(t, self._duration))
        else:
            t = max(0.0, t)
        self._seek_pos = t
        self._frames_shown = 0
        self._active = True
        self._generation += 1
        gen = self._generation

        seek_args = ["-ss", f"{t:.3f}"] if t > 0 else []

        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "quiet",
                *seek_args,
                "-i",
                self._path,
                "-vf",
                f"scale={self._frame_w}:{self._frame_h}:flags=fast_bilinear",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._read_frames, args=(gen,), daemon=True)
        self._thread.start()

        if self._has_audio and HAS_FFPLAY:
            try:
                self._audio_proc = subprocess.Popen(
                    [
                        "ffplay",
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "quiet",
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

    def _fetch_preview_frame(
        self, proc: subprocess.Popen[bytes], frame_bytes: int, t: float
    ) -> None:
        raw = b""
        try:
            while len(raw) < frame_bytes:
                assert proc.stdout is not None
                chunk = proc.stdout.read(frame_bytes - len(raw))
                if not chunk:
                    break
                raw += chunk
        except Exception:
            pass
        finally:
            try:
                assert proc.stdout is not None
                proc.stdout.close()
            except Exception:
                pass
            proc.wait()
        if self._proc is proc:
            self._proc = None
        if len(raw) == frame_bytes and self._paused and self._pause_pos == t:
            self._canvas.after(0, self._show_preview_frame, raw)

    def _show_preview_frame(self, raw: bytes) -> None:
        if not self._paused:
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
            anchor="nw",
            image=photo,
        )
        if self._position_cb is not None:
            self._position_cb(self._pause_pos)

    def _cancel_tick(self) -> None:
        if self._after_id is not None:
            try:
                self._canvas.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _drain_queue(self) -> None:
        while True:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break

    def _teardown_audio(self) -> None:
        if self._audio_proc is not None:
            self._audio_proc.terminate()
            self._audio_proc = None

    def _read_frames(self, gen: int) -> None:
        frame_bytes = self._frame_w * self._frame_h * 3
        proc = self._proc
        while self._active and proc and proc.stdout:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                try:
                    self._frame_q.put((gen, None), timeout=2.0)
                except queue.Full:
                    pass
                break
            try:
                self._frame_q.put((gen, raw), timeout=2.0)
            except queue.Full:
                pass  # drop frame — display is lagging

    def _tick(self) -> None:
        if not self._active:
            return

        elapsed = time.monotonic() - self._play_start
        target = int(elapsed * self._fps)
        cur = self._generation

        while self._frames_shown < target:
            try:
                gen, stale = self._frame_q.get_nowait()
            except queue.Empty:
                break
            if gen != cur:
                continue
            if stale is None:
                self._active = False
                self._on_end()
                return
            self._frames_shown += 1

        while True:
            try:
                gen, raw = self._frame_q.get_nowait()
            except queue.Empty:
                self._after_id = self._canvas.after(5, self._tick)
                return
            if gen == cur:
                break

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
            anchor="nw",
            image=photo,
        )
        self._frames_shown += 1

        if self._position_cb is not None:
            pos = self._seek_pos + self._frames_shown / self._fps
            self._position_cb(pos)

        next_due = self._frames_shown / self._fps
        delay = max(1, int((next_due - (time.monotonic() - self._play_start)) * 1000))
        self._after_id = self._canvas.after(delay, self._tick)
