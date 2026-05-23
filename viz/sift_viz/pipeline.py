from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from sift_viz.media import log_stderr
from sift_viz.models import HashSettings

log = logging.getLogger("sift-viz")

# Queue message dict sent back to the main thread.
QueueMsg = dict[str, Any]


def _parse_progress(line: str) -> tuple[int, int] | None:
    """Parse 'sift: progress N/TOTAL' → (n, total), or None if not a progress line."""
    if not line.startswith("sift: progress "):
        return None
    try:
        frac = line.split()[-1]
        n, total = map(int, frac.split("/"))
        return n, total
    except Exception:
        return None


class PipelineRunner:
    """Runs the sift hash/project/cluster subprocess pipeline in daemon threads."""

    def __init__(
        self,
        binary: str,
        work_dir: Path,
        event_queue: queue.Queue[QueueMsg],
    ) -> None:
        self.binary = binary
        self.work_dir = work_dir
        self._q = event_queue
        self._cancel_requested = False
        self._current_proc: subprocess.Popen[str] | None = None

        self._hashes_json = str(work_dir / "hashes.json")
        self._proj_2d_json = str(work_dir / "projection_2d.json")
        self._proj_3d_json = str(work_dir / "projection_3d.json")
        self._clusters_json = str(work_dir / "clusters.json")

    @property
    def hashes_path(self) -> Path:
        return Path(self._hashes_json)

    def cancel(self) -> None:
        log.info("cancelling pipeline")
        self._cancel_requested = True
        if self._current_proc is not None:
            self._current_proc.terminate()

    def run_pipeline(self, settings: HashSettings, folder: str) -> None:
        """Start hash → project (2D+3D) in a daemon thread."""
        self._cancel_requested = False
        threading.Thread(
            target=self._pipeline_thread,
            args=(settings, folder),
            daemon=True,
        ).start()

    def run_projection(self, settings: HashSettings) -> None:
        """Re-run projection (2D+3D) from existing hashes in a daemon thread."""
        threading.Thread(
            target=self._projection_thread,
            args=(settings,),
            daemon=True,
        ).start()

    def run_cluster(self, settings: HashSettings) -> None:
        """Run clustering from existing hashes in a daemon thread."""
        threading.Thread(
            target=self._cluster_thread,
            args=(settings,),
            daemon=True,
        ).start()

    # ── Private thread targets ────────────────────────────────────────────────

    def _run_subprocess(
        self,
        cmd: list[str],
        stage: str,
        *,
        stream_progress: bool = False,
        progress_kind: str = "progress",
        progress_transform: Any = None,
    ) -> tuple[int, str]:
        """Run *cmd*, streaming stderr line by line. Returns (returncode, stderr)."""
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self._current_proc = proc
        stderr_lines: list[str] = []

        for raw in proc.stderr:  # type: ignore[union-attr]
            if self._cancel_requested:
                proc.terminate()
                break
            line = raw.strip()
            if not line:
                continue
            parsed = _parse_progress(line) if stream_progress else None
            if parsed is not None:
                n, total = parsed
                if progress_transform:
                    n, total = progress_transform(n, total)
                self._q.put({"kind": progress_kind, "value": n, "total": total})
            else:
                log_stderr(line, stage)
            stderr_lines.append(line)

        proc.wait()
        if self._current_proc is proc:
            self._current_proc = None
        return proc.returncode, "\n".join(stderr_lines)

    def _pipeline_thread(self, settings: HashSettings, folder: str) -> None:
        try:
            # ── Hash ──────────────────────────────────────────────────────────
            media_label = {"images": "images", "videos": "videos", "all": "images+videos"}[
                settings.media
            ]
            self._q.put(
                {
                    "kind": "status",
                    "text": f"Hashing {media_label} with {settings.algo} "
                    f"{settings.hash_size}×{settings.hash_size}…",
                }
            )
            self._q.put({"kind": "hash_start"})

            hash_cmd = [
                self.binary,
                "hash",
                folder,
                f"--algo={settings.algo}",
                f"--size={settings.hash_size}",
                f"--media={settings.media}",
                f"--output={self._hashes_json}",
            ]
            if settings.media in ("videos", "all"):
                hash_cmd.append(f"--frames={settings.frames}")

            rc, stderr = self._run_subprocess(hash_cmd, "hash", stream_progress=True)

            if self._cancel_requested:
                self._q.put({"kind": "cancelled"})
                return
            if rc != 0:
                raise RuntimeError(stderr.splitlines()[-1] if stderr else "sift hash failed")

            # ── Project (both 2D and 3D) ──────────────────────────────────────
            self._q.put({"kind": "proj_progress_indeterminate"})
            phase_info: dict[str, int] = {"phase": 0, "total": 0}

            def make_transform() -> Any:
                phase = phase_info["phase"]

                def transform(n: int, total: int) -> tuple[int, int]:
                    if phase == 0:
                        phase_info["total"] = total
                    pt = phase_info["total"]
                    return phase * pt + n, max(1, pt) * 2

                return transform

            def run_projection(dims: int, out_path: str) -> dict[str, Any]:
                self._q.put(
                    {
                        "kind": "status",
                        "text": f"Running {settings.proj.upper()} projection ({dims}D)…",
                    }
                )
                cmd = [
                    self.binary,
                    "project",
                    self._hashes_json,
                    f"--method={settings.proj}",
                    f"--dims={dims}",
                    f"--output={out_path}",
                ]
                if settings.proj == "tsne":
                    cmd += [
                        f"--perplexity={settings.perplexity}",
                        f"--iterations={settings.iterations}",
                    ]
                rc, stderr = self._run_subprocess(
                    cmd,
                    "project",
                    stream_progress=True,
                    progress_kind="proj_progress",
                    progress_transform=make_transform(),
                )
                phase_info["phase"] += 1
                if self._cancel_requested:
                    return {}
                if rc != 0:
                    raise RuntimeError(stderr.splitlines()[-1] if stderr else "sift project failed")
                with open(out_path) as fh:
                    return json.load(fh)  # type: ignore[no-any-return]

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

    def _projection_thread(self, settings: HashSettings) -> None:
        try:
            phase_total = 0
            proj_cmds = [
                (self._build_proj_cmd(settings, 2, self._proj_2d_json), "2D"),
                (self._build_proj_cmd(settings, 3, self._proj_3d_json), "3D"),
            ]
            for phase, (cmd, label) in enumerate(proj_cmds):
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
                )
                for raw in proc.stderr:  # type: ignore[union-attr]
                    line = raw.strip()
                    if not line:
                        continue
                    parsed = _parse_progress(line)
                    if parsed is not None:
                        pn, ptotal = parsed
                        if phase == 0:
                            phase_total = ptotal
                        combined_n = phase * phase_total + pn
                        combined_total = max(1, phase_total) * 2
                        self._q.put(
                            {
                                "kind": "proj_progress",
                                "value": combined_n,
                                "total": combined_total,
                            }
                        )
                    else:
                        log_stderr(line, "project")
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"sift project {label} failed")

            with open(self._proj_2d_json) as fh:
                proj_2d = json.load(fh)
            with open(self._proj_3d_json) as fh:
                proj_3d = json.load(fh)
            self._q.put(
                {"kind": "projection_done", "projection_2d": proj_2d, "projection_3d": proj_3d}
            )
        except Exception as exc:
            log.warning("projection error: %s", exc)
            self._q.put({"kind": "projection_error", "text": str(exc)})

    def _cluster_thread(self, settings: HashSettings) -> None:
        try:
            cmd = self._build_cluster_cmd(settings)
            log.debug("cluster cmd: %s", " ".join(cmd))
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            for raw in proc.stderr:  # type: ignore[union-attr]
                line = raw.strip()
                if not line:
                    continue
                parsed = _parse_progress(line)
                if parsed is not None:
                    cn, ctotal = parsed
                    self._q.put({"kind": "cluster_progress", "value": cn, "total": ctotal})
                else:
                    log_stderr(line, "cluster")
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError("sift cluster failed")
            with open(self._clusters_json) as fh:
                clusters = json.load(fh)
            self._q.put({"kind": "cluster_done", "clusters": clusters})
        except Exception as exc:
            log.warning("cluster error: %s", exc)
            self._q.put({"kind": "cluster_error", "text": str(exc)})

    def _build_proj_cmd(self, settings: HashSettings, dims: int, out: str) -> list[str]:
        cmd = [
            self.binary,
            "project",
            self._hashes_json,
            f"--method={settings.proj}",
            f"--dims={dims}",
            f"--output={out}",
        ]
        if settings.proj == "tsne":
            cmd += [
                f"--perplexity={settings.perplexity}",
                f"--iterations={settings.iterations}",
            ]
        return cmd

    def _build_cluster_cmd(self, settings: HashSettings) -> list[str]:
        cmd = [
            self.binary,
            "cluster",
            self._hashes_json,
            f"--method={settings.cluster_method}",
            f"--min-filter={settings.min_filter}",
            f"--output={self._clusters_json}",
        ]
        if settings.cluster_method == "threshold":
            cmd.append(f"--threshold={settings.threshold}")
        elif settings.cluster_method == "hierarchical":
            cmd += [
                f"--linkage={settings.linkage}",
                f"--cut-height={settings.cut_height}",
            ]
        elif settings.cluster_method == "hdbscan":
            cmd.append(f"--min-group={settings.min_group}")
        return cmd
