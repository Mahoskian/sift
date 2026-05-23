"""Module-level widget factory functions shared across UI panels.

All functions are pure constructors — they take a parent widget and
configuration, create the widget, and return or grid it. None reference
app state.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any

from sift_viz.constants import ACCENT_COLOR, DIM_COLOR, PANEL_BG, TEXT_COLOR, VIEWER_W


def section_label(parent: tk.Widget, row: int, text: str) -> None:
    """Add a small uppercase section heading at *row*."""
    tk.Label(parent, text=text, bg=PANEL_BG, fg=DIM_COLOR, font=("", 8, "bold")).grid(
        row=row, column=0, sticky="w", padx=12, pady=(14, 2)
    )


def radio(
    parent: tk.Widget,
    row: int,
    text: str,
    var: tk.Variable,
    value: str,
    command: Callable[[], None] | None = None,
) -> None:
    """Add a dark-themed radiobutton at *row*."""
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


def divider(parent: tk.Widget, row: int) -> None:
    """Insert a 1-pixel horizontal divider line at *row*."""
    tk.Frame(parent, bg="#333", height=1).grid(
        row=row, column=0, sticky="ew", padx=12, pady=(10, 0)
    )


def flat_btn(parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
    """Create and return a flat dark button (not yet gridded)."""
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


def icon_btn(parent: tk.Widget, text: str, command: Callable[[], None], col: int) -> tk.Button:
    """Create, grid, and return a small icon/navigation button."""
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


def action_btn(parent: tk.Widget, text: str, command: Callable[[], None], col: int) -> tk.Button:
    """Create, grid, and return a viewer action button."""
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


def viewer_label(
    parent: tk.Widget,
    row: int,
    *,
    text: str = "",
    fg: str = DIM_COLOR,
    font: tuple[str, int] | tuple[str, int, str] = ("", 8),
    pady: tuple[int, int] | int = (1, 0),
) -> tk.Label:
    """Create, grid, and return a centred label for the viewer info area."""
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


def slider(
    parent: tk.Widget,
    row: int,
    var: tk.IntVar,
    lo: int,
    hi: int,
    on_change: Callable[[], None] | None = None,
    label_fn: Callable[[int], str] | None = None,
) -> tk.Label:
    """Add a horizontal slider with an adjacent value label.

    Returns the value Label widget.
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

    # Keep label in sync when var is set programmatically (e.g. cache restore).
    def _sync(*_: Any) -> None:
        try:
            v = int(var.get())
            lbl.config(text=label_fn(v) if label_fn else str(v))
        except Exception:
            pass

    tid = var.trace_add("write", _sync)
    f.bind("<Destroy>", lambda _e, _t=tid: var.trace_remove("write", _t), add=True)

    return lbl


def labeled_slider(
    parent: tk.Widget,
    row: int,
    label: str,
    var: tk.IntVar,
    lo: int,
    hi: int,
    on_change: Callable[[], None] | None = None,
) -> None:
    """Add a text label followed by a slider on the next row."""
    tk.Label(parent, text=label, bg=PANEL_BG, fg=DIM_COLOR, font=("", 8)).grid(
        row=row, column=0, sticky="w", padx=12, pady=(4, 0)
    )
    slider(parent, row + 1, var, lo, hi, on_change=on_change)


def run_button(parent: tk.Widget, command: Callable[[], None]) -> tk.Button:
    """Create and return the primary Run button (not yet gridded)."""
    return tk.Button(
        parent,
        text="Run",
        command=command,
        bg=ACCENT_COLOR,
        fg="white",
        relief="flat",
        font=("", 10, "bold"),
        pady=6,
        cursor="hand2",
        activebackground="#3a8ee6",
        activeforeground="white",
    )
