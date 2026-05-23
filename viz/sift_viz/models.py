from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class HashSettings(BaseModel):
    algo: Literal["dhash", "phash", "whash"] = "dhash"
    hash_size: int = Field(default=8, ge=4, le=32)
    media: Literal["images", "videos", "all"] = "images"
    frames: int = Field(default=8, ge=1, le=32)
    proj: Literal["pca", "tsne"] = "pca"
    dims: Literal["2", "3"] = "2"
    perplexity: int = Field(default=30, ge=5, le=100)
    iterations: int = Field(default=1000, ge=250, le=3000)
    cluster_method: Literal["threshold", "hierarchical", "hdbscan"] = "threshold"
    threshold: int = Field(default=10, ge=1, le=100)
    cut_height: int = Field(default=10, ge=1, le=100)
    linkage: Literal["single", "complete", "average"] = "complete"
    min_group: int = Field(default=3, ge=2, le=30)
    min_filter: int = Field(default=2, ge=2, le=20)


class CacheMeta(BaseModel):
    saved_at: float
    folder: str
    settings: HashSettings


class ProjectionResult(BaseModel):
    files: list[str]
    points: list[list[float]]
    method: Literal["pca", "tsne"]
    variance_explained: list[float] = Field(default_factory=list)


class ClusterGroup(BaseModel):
    members: list[int]
    member_files: list[str] = Field(default_factory=list)


class ClusterResult(BaseModel):
    files: list[str] = Field(default_factory=list)
    groups: list[ClusterGroup] = Field(default_factory=list)
    ungrouped: list[int] = Field(default_factory=list)


class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float
    has_audio: bool
    duration: float

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass
class ViewerState:
    """Tracks which file(s) are currently loaded in the right-hand viewer panel."""

    files: list[str] = field(default_factory=list)
    index: int = 0
    group_id: int | None = None

    @property
    def current_path(self) -> str | None:
        return self.files[self.index] if self.files else None

    @property
    def is_multi(self) -> bool:
        return len(self.files) > 1


@dataclass
class SelectionState:
    """Tracks which scatter point or cluster is currently selected."""

    point_idx: int | None = None
    group_idx: int | None = None
    member_indices: list[int] = field(default_factory=list)

    def clear(self) -> None:
        self.point_idx = None
        self.group_idx = None
        self.member_indices = []
