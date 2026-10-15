"""Matplotlib figures used by the PDF report and the evaluation/XAI scripts."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GRID_MAJOR = "#f2b8b8"
GRID_MINOR = "#fbe3e3"
TRACE = "#1f2933"
LAYOUT = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]


def _ecg_grid(ax, t_max: float, y_min: float, y_max: float) -> None:
    for x in np.arange(0, t_max + 1e-9, 0.04):
        ax.axvline(x, color=GRID_MINOR, lw=0.3, zorder=0)
    for x in np.arange(0, t_max + 1e-9, 0.2):
        ax.axvline(x, color=GRID_MAJOR, lw=0.6, zorder=0)
    for y in np.arange(np.floor(y_min / 0.1) * 0.1, y_max + 1e-9, 0.1):
        ax.axhline(y, color=GRID_MINOR, lw=0.3, zorder=0)
    for y in np.arange(np.floor(y_min / 0.5) * 0.5, y_max + 1e-9, 0.5):
        ax.axhline(y, color=GRID_MAJOR, lw=0.6, zorder=0)


def plot_12lead(signal_mv: np.ndarray, fs: float, leads: list[str], title: str = "",
                seconds: float = 10.0, attribution: np.ndarray | None = None) -> plt.Figure:
    """Standard 3x4 layout (2.5 s per panel) on ECG paper, plus a full-length lead II strip.
    If `attribution` [T, 12] is given, |attribution| is shaded behind each trace."""
    n = min(signal_mv.shape[0], int(seconds * fs))
    sig = signal_mv[:n]
    seg = int(2.5 * fs)
    fig = plt.figure(figsize=(11, 6.6), dpi=130)
    gs = fig.add_gridspec(4, 4, hspace=0.08, wspace=0.02)
    att = None
    if attribution is not None:
        att = np.abs(attribution[:n])
        att = att / att.max() if att.max() > 0 else att
    for r, row in enumerate(LAYOUT):
        for c, lead in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            j = leads.index(lead)
            start = c * seg
            y = sig[start:start + seg, j]
            t = np.arange(len(y)) / fs
            lo, hi = min(-1.0, y.min() - 0.2), max(1.0, y.max() + 0.2)
            _ecg_grid(ax, 2.5, lo, hi)
            if att is not None:
                a = att[start:start + seg, j]
                ax.imshow(a[None, :], aspect="auto", cmap="Reds", alpha=0.55, vmin=0, vmax=1,
                          extent=(0, len(a) / fs, lo, hi), zorder=1)
            ax.plot(t, y, color=TRACE, lw=0.8, zorder=2)
            ax.set_xlim(0, 2.5)
            ax.set_ylim(lo, hi)
            ax.text(0.03, 0.82, lead, transform=ax.transAxes, fontsize=8, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
    ax = fig.add_subplot(gs[3, :])
    j = leads.index("II")
    y = sig[:, j]
    t = np.arange(len(y)) / fs
    lo, hi = min(-1.0, y.min() - 0.2), max(1.0, y.max() + 0.2)
    _ecg_grid(ax, n / fs, lo, hi)
    if att is not None:
        ax.imshow(att[:, j][None, :], aspect="auto", cmap="Reds", alpha=0.55, vmin=0, vmax=1,
                  extent=(0, n / fs, lo, hi), zorder=1)
    ax.plot(t, y, color=TRACE, lw=0.8, zorder=2)
    ax.set_xlim(0, n / fs)
    ax.set_ylim(lo, hi)
    ax.text(0.005, 0.82, "II (rhythm)", transform=ax.transAxes, fontsize=8, weight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Time (s) — grid: 0.2 s × 0.5 mV", fontsize=8)
    ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.07)
    return fig


def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_probability_bars(results: dict, classes: list[str], model_order: list[str],
                          names: dict[str, str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=130)
    present = [m for m in model_order if m in results]
    width = 0.8 / max(1, len(present))
    colors = ["#0f766e", "#2563eb", "#7c3aed", "#b45309", "#be123c"]
    x = np.arange(len(classes))
    for i, m in enumerate(present):
        vals = [results[m]["probs"][c] for c in classes]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=names.get(m, m), color=colors[i % len(colors)])
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Model probability")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
