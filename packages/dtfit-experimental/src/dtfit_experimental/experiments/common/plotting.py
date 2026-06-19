"""Matplotlib setup and shared figure helpers (headless / Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True})


def fit_overlay(ax, x, y, yhat, *, truth=None, title="", label="fit",
                color="tab:blue", train_split=None):
    """Scatter samples + fitted curve (+ optional ground truth / holdout split)."""
    if train_split is not None:
        ax.scatter(x[:train_split], y[:train_split], s=12, c="0.6",
                   label="train")
        ax.scatter(x[train_split:], y[train_split:], s=12, c="tab:orange",
                   label="held-out")
        ax.axvline(x[train_split], color="0.7", ls=":")
    else:
        ax.scatter(x, y, s=12, c="0.6", label="data")
    if truth is not None:
        ax.plot(x, truth, "k--", lw=1, label="ground truth")
    ax.plot(x, yhat, color=color, lw=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)


def residuals(ax, x, y, yhat, *, title="Residuals", color="tab:blue"):
    ax.plot(x, np.asarray(yhat) - np.asarray(y), color=color, lw=1)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("residual")


def error_bars(ax, names, values, *, ylabel="", title="", colors=None,
               annotate="{:.3g}"):
    """A labelled bar chart of one metric per method (lower-is-better style)."""
    colors = colors or [f"C{i}" for i in range(len(names))]
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v, annotate.format(v),
                    ha="center", va="bottom", fontsize=8)
    return bars


def savefig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
