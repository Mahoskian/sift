from __future__ import annotations

import hashlib
import logging
import shutil
import time
from pathlib import Path

from pydantic import ValidationError

from sift_viz.models import CacheMeta, ClusterResult, HashSettings, ProjectionResult

log = logging.getLogger("sift-viz")

_REQUIRED_FILES = [
    "hashes.json",
    "projection_2d.json",
    "projection_3d.json",
    "clusters.json",
    "meta.json",
]


class CacheManager:
    """Reads and writes pipeline results to ~/.cache/sift/<folder_hash>/."""

    def cache_dir(self, folder: str) -> Path:
        key = hashlib.md5(folder.encode()).hexdigest()
        return Path.home() / ".cache" / "sift" / key

    def save(
        self,
        folder: str,
        settings: HashSettings,
        proj_2d: ProjectionResult,
        proj_3d: ProjectionResult | None,
        clusters: ClusterResult,
        hashes_src: Path,
    ) -> None:
        """Persist pipeline results to disk."""
        if not folder:
            return
        cache = self.cache_dir(folder)
        cache.mkdir(parents=True, exist_ok=True)

        if hashes_src.exists():
            shutil.copy2(hashes_src, cache / "hashes.json")

        (cache / "projection_2d.json").write_text(proj_2d.model_dump_json())
        (cache / "projection_3d.json").write_text(
            proj_3d.model_dump_json() if proj_3d is not None else "{}"
        )
        (cache / "clusters.json").write_text(clusters.model_dump_json())

        meta = CacheMeta(saved_at=time.time(), folder=folder, settings=settings)
        (cache / "meta.json").write_text(meta.model_dump_json(indent=2))
        log.info("cache saved: %s", cache)

    def load(self, folder: str) -> CacheMeta | None:
        """Return parsed CacheMeta if a valid cache exists, else None."""
        cache = self.cache_dir(folder)
        if not all((cache / f).exists() for f in _REQUIRED_FILES):
            return None
        try:
            return CacheMeta.model_validate_json((cache / "meta.json").read_text())
        except (ValidationError, Exception) as exc:
            log.warning("cache meta load failed: %s", exc)
            return None

    def load_results(
        self, folder: str
    ) -> tuple[ProjectionResult, ProjectionResult | None, ClusterResult] | None:
        """Return (proj_2d, proj_3d, clusters) from cache, or None on miss/corrupt."""
        cache = self.cache_dir(folder)
        if not all((cache / f).exists() for f in _REQUIRED_FILES):
            return None
        try:
            proj_2d = ProjectionResult.model_validate_json(
                (cache / "projection_2d.json").read_text()
            )
            proj_3d_text = (cache / "projection_3d.json").read_text()
            proj_3d: ProjectionResult | None = None
            if proj_3d_text.strip() not in ("{}", ""):
                proj_3d = ProjectionResult.model_validate_json(proj_3d_text)
            clusters = ClusterResult.model_validate_json((cache / "clusters.json").read_text())
            return proj_2d, proj_3d, clusters
        except (ValidationError, Exception) as exc:
            log.warning("cache results load failed: %s", exc)
            return None

    def copy_hashes_to(self, folder: str, dest: Path) -> bool:
        """Copy cached hashes.json to *dest*. Returns True on success."""
        src = self.cache_dir(folder) / "hashes.json"
        if not src.exists():
            return False
        try:
            shutil.copy2(src, dest)
            return True
        except Exception as exc:
            log.warning("could not copy hashes: %s", exc)
            return False
