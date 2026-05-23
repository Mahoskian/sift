from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from matplotlib.artist import Artist
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d  # noqa: F401

from sift_viz.constants import (
    ACCENT_COLOR,
    BG_COLOR,
    DIM_COLOR,
    POINT_HIT_RADIUS_PX,
    TEXT_COLOR,
    UNGROUPED_COLOR,
)
from sift_viz.media import cluster_palette
from sift_viz.models import ClusterResult, ProjectionResult, SelectionState, ViewerState

if TYPE_CHECKING:
    pass

log = logging.getLogger("sift-viz")


class ScatterController:
    """Manages the matplotlib figure, axes, scatter rendering, and hit-testing."""

    def __init__(self, fig: Figure, canvas: FigureCanvasTkAgg) -> None:
        self._fig = fig
        self._canvas = canvas
        self._ax: Any = fig.add_subplot(111)  # replaced on first rebuild
        self._is_3d = False

        self._point_positions: npt.NDArray[np.float64] | None = None
        self._point_files: list[str] = []
        self._point_group: dict[int, int] = {}  # point_idx → group_idx
        self._overlay_artists: list[Artist] = []

        self._style_idle()

    @property
    def ax(self) -> Any:
        return self._ax

    @property
    def point_files(self) -> list[str]:
        return self._point_files

    @property
    def point_group(self) -> dict[int, int]:
        return self._point_group

    def rebuild_axes(self, is_3d: bool) -> None:
        """Clear the figure and recreate axes for the given dimension mode."""
        self._fig.clear()
        self._is_3d = is_3d
        if is_3d:
            self._ax = self._fig.add_subplot(111, projection="3d")
            self._style_3d_idle()
        else:
            self._ax = self._fig.add_subplot(111)
            self._style_2d_idle()

    def style_idle(self) -> None:
        """Apply the dark idle style to the current axes."""
        if self._is_3d:
            self._style_3d_idle()
        else:
            self._style_2d_idle()

    def _style_idle(self) -> None:
        """Apply idle style (called during __init__ before first rebuild)."""
        self._style_2d_idle()

    def _style_2d_idle(self) -> None:
        self._ax.set_facecolor(BG_COLOR)
        self._ax.set_title("Run the pipeline to begin", color=DIM_COLOR, fontsize=11)
        self._ax.tick_params(colors=DIM_COLOR)
        for spine in self._ax.spines.values():
            spine.set_color("#333")
        self._fig.tight_layout()

    def _style_3d_idle(self) -> None:
        self._ax.set_facecolor(BG_COLOR)
        for pane in (self._ax.xaxis.pane, self._ax.yaxis.pane, self._ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#2e2e2e")
        self._ax.tick_params(colors=DIM_COLOR, labelsize=7)
        self._ax.set_title("Run the pipeline to begin", color=DIM_COLOR, fontsize=11)
        self._fig.tight_layout()

    def redraw(
        self,
        proj: ProjectionResult,
        clusters: ClusterResult | None,
        is_3d: bool,
    ) -> None:
        """Replot all scatter points coloured by cluster membership."""
        files = proj.files
        pts = np.array(proj.points)
        n = len(files)
        actual_3d = is_3d and pts.ndim == 2 and pts.shape[1] == 3

        colors: list[Any] = [UNGROUPED_COLOR] * n
        self._point_group = {}
        n_groups = 0

        if clusters is not None:
            n_groups = len(clusters.groups)
            palette = cluster_palette(n_groups)
            for gi, group in enumerate(clusters.groups):
                for idx in group.members:
                    if 0 <= idx < n:
                        colors[idx] = palette[gi]
                        self._point_group[idx] = gi

        self.rebuild_axes(actual_3d)

        c_rgba = mcolors.to_rgba_array(colors)

        if actual_3d:
            self._ax.scatter(
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
            self._ax.scatter(
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

        method = proj.method
        if method == "pca":
            ve = proj.variance_explained
            xl = f"PC1  ({ve[0] * 100:.1f}%)" if ve else "PC1"
            yl = f"PC2  ({ve[1] * 100:.1f}%)" if len(ve) > 1 else "PC2"
            zl = f"PC3  ({ve[2] * 100:.1f}%)" if len(ve) > 2 else "PC3"
        else:
            xl, yl, zl = "t-SNE 1", "t-SNE 2", "t-SNE 3"

        self._ax.set_xlabel(xl, color=DIM_COLOR, fontsize=9)
        self._ax.set_ylabel(yl, color=DIM_COLOR, fontsize=9)

        if actual_3d:
            self._ax.set_zlabel(zl, color=DIM_COLOR, fontsize=9)
            for pane in (self._ax.xaxis.pane, self._ax.yaxis.pane, self._ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor("#2e2e2e")
            self._ax.tick_params(colors="#555", labelsize=7)
        else:
            self._ax.tick_params(colors="#555", labelsize=8)
            for spine in self._ax.spines.values():
                spine.set_color("#2e2e2e")

        dim_label = "3D" if actual_3d else "2D"
        self._ax.set_title(
            (
                f"{n} files — {n_groups} cluster{'s' if n_groups != 1 else ''}  [{dim_label}]"
                if clusters is not None
                else f"{n} files  [{dim_label}]"
            ),
            color=TEXT_COLOR,
            fontsize=11,
            pad=10,
        )
        self._fig.tight_layout()
        self._overlay_artists = []

    def apply_selection_overlay(self, selection: SelectionState, viewer: ViewerState) -> None:
        """Remove stale overlays and draw a ring on the currently viewed point."""
        for a in self._overlay_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._overlay_artists = []

        pts = self._point_positions
        if pts is None:
            self._canvas.draw()
            return

        pt_idx: int | None = None
        if selection.group_idx is not None:
            if selection.member_indices and 0 <= viewer.index < len(selection.member_indices):
                pt_idx = selection.member_indices[viewer.index]
        elif selection.point_idx is not None:
            pt_idx = selection.point_idx

        is_3d = pts.ndim == 2 and pts.shape[1] == 3
        if is_3d:
            xlim = self._ax.get_xlim3d()
            ylim = self._ax.get_ylim3d()
            zlim = self._ax.get_zlim3d()

        if pt_idx is not None and 0 <= pt_idx < len(pts):
            self._draw_ring(pts[pt_idx])

        if is_3d:
            self._ax.set_xlim3d(xlim)
            self._ax.set_ylim3d(ylim)
            self._ax.set_zlim3d(zlim)

        self._canvas.draw()

    def _draw_ring(self, xy: npt.NDArray[np.float64]) -> None:
        """Draw a white ring + accent dot at the given scatter coordinate."""
        if len(xy) == 3:
            x, y, z = float(xy[0]), float(xy[1]), float(xy[2])
            outer = self._ax.scatter(
                [x], [y], [z], s=220, c="none", edgecolors="white", linewidths=2.5
            )
            inner = self._ax.scatter(
                [x], [y], [z], s=70, c=[ACCENT_COLOR], edgecolors="none", alpha=0.9
            )
        else:
            x, y = float(xy[0]), float(xy[1])
            outer = self._ax.scatter(
                [x], [y], s=220, c="none", edgecolors="white", linewidths=2.5, zorder=7
            )
            inner = self._ax.scatter(
                [x], [y], s=70, c=[ACCENT_COLOR], edgecolors="none", alpha=0.9, zorder=8
            )
        self._overlay_artists.extend([outer, inner])

    def hit_test(self, event: Any, proj: ProjectionResult) -> int | None:
        """Return the index of the nearest point within POINT_HIT_RADIUS_PX, or None."""
        pts = self._point_positions
        if pts is None:
            return None

        is_3d = pts.ndim == 2 and pts.shape[1] == 3

        if is_3d:
            return self._hit_test_3d(event, pts)
        return self._hit_test_2d(event, pts)

    def _hit_test_2d(self, event: Any, pts: npt.NDArray[np.float64]) -> int | None:
        click_pt = np.array([event.xdata, event.ydata])
        click_disp = self._ax.transData.transform(click_pt.reshape(1, 2))[0]
        pts_disp = self._ax.transData.transform(pts)
        dists = np.linalg.norm(pts_disp - click_disp, axis=1)
        nearest = int(np.argmin(dists))
        return nearest if dists[nearest] <= POINT_HIT_RADIUS_PX else None

    def _hit_test_3d(self, event: Any, pts: npt.NDArray[np.float64]) -> int | None:
        try:
            proj_mat = self._ax.get_proj()
            x2d, y2d, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], proj_mat)
            pts_ax = np.column_stack([x2d, y2d])
            pts_disp = self._ax.transData.transform(pts_ax)
            click_disp = np.array([event.x, event.y])
            dists = np.linalg.norm(pts_disp - click_disp, axis=1)
            nearest = int(np.argmin(dists))
            return nearest if dists[nearest] <= POINT_HIT_RADIUS_PX else None
        except Exception as exc:
            log.debug("3D hit test failed: %s", exc)
            return None

    def clear(self) -> None:
        """Reset point tracking state without rebuilding axes."""
        self._point_positions = None
        self._point_files = []
        self._point_group = {}
        self._overlay_artists = []
