#!/usr/bin/env python3
"""Section 4.8 unique-token figure: val loss and unique-data multiplier.

75M points are Protocol A (Tables 1-3). Other unique-token budgets are
Protocol B 100-epoch minima. Open markers = multiplier extrapolated past 300M.

    PYTHONPATH=src python scripts/plot_scaling.py
    PYTHONPATH=src python scripts/plot_scaling.py --out figures
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

COL = {
    "baseline": "#222222",
    "rand15": "#E6A817",
    "combo": "#8E7CC3",
}
LBL = {
    "baseline": "Baseline",
    "rand15": "Rand. 15%",
    "combo": r"Rand. 5% + R2L + $i\leq5$ exp.",
}

# 75M column uses Tables 1-3. 150M/300M 3-cat and 300M Random 15% are
# 100-epoch continued Protocol B minima.
RATIO_X = [19, 37, 75, 150, 300]
RATIO = {
    "baseline": [4.7139, 4.2924, 4.015, 3.7377, 3.5035],
    "rand15": [4.5129, 4.1514, 3.841, 3.6035, 3.3815],
    "combo": [4.3625, 4.0568, 3.805, 3.5814, 3.4511],
}


def apply_style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(colors="#222222", labelsize=10)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def plot_series(ax, xs, ys, key, z=3):
    ax.plot(
        xs, ys, color=COL[key], marker="o", ms=6.5, lw=2.0,
        label=LBL[key], zorder=z, clip_on=False,
    )


def unique_data_multipliers(xs, base, method):
    """Piecewise log-linear interpolation of the baseline (U, L) curve.

    The last segment is used even if the method loss lies below the last
    baseline point (short extrapolation past 300M).
    """
    out, extra = [], []
    for u, loss in zip(xs, method):
        ueq = None
        extrapolated = False
        for j in range(len(xs) - 1):
            y0, y1 = base[j], base[j + 1]
            last = j == len(xs) - 2
            if (not last) and not (y0 >= loss >= y1):
                continue
            x0, x1 = math.log(xs[j]), math.log(xs[j + 1])
            t = (loss - y0) / (y1 - y0)
            ueq = math.exp(x0 + t * (x1 - x0))
            extrapolated = last and (loss < y1)
            break
        if ueq is None:
            ueq = xs[0]
        out.append(ueq / u)
        extra.append(extrapolated)
    return out, extra


def plot_multiplier(ax, xs, mults, extra, key, z=3):
    ax.plot(xs, mults, color=COL[key], lw=2.0, zorder=z, clip_on=False)
    xs_in = [x for x, e in zip(xs, extra) if not e]
    ys_in = [y for y, e in zip(mults, extra) if not e]
    xs_ex = [x for x, e in zip(xs, extra) if e]
    ys_ex = [y for y, e in zip(mults, extra) if e]
    ax.plot(
        xs_in, ys_in, color=COL[key], marker="o", ms=6.5, lw=0,
        label=LBL[key], zorder=z + 1, clip_on=False,
    )
    if xs_ex:
        ax.plot(
            xs_ex, ys_ex, color=COL[key], marker="o", ms=6.5,
            mfc="white", mew=1.6, lw=0, zorder=z + 1, clip_on=False,
        )


def fig_unique(out_dir: Path):
    xs = RATIO_X
    base = RATIO["baseline"]
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.55), sharex=True)
    for ax in axes:
        apply_style(ax)
        ax.axvline(75, color="#bbbbbb", ls=":", lw=1.1, zorder=1)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())

    plot_series(axes[0], xs, base, "baseline")
    plot_series(axes[0], xs, RATIO["rand15"], "rand15", z=4)
    plot_series(axes[0], xs, RATIO["combo"], "combo", z=5)
    axes[0].set_ylabel("Validation loss")
    axes[0].set_ylim(3.25, 4.85)
    axes[0].legend(frameon=True, fancybox=False, edgecolor="#cccccc", fontsize=8.0, loc="upper right")

    axes[1].axhline(1.0, color="#222222", ls="--", lw=1.1, zorder=1)
    for key, z in (("rand15", 3), ("combo", 4)):
        mults, extra = unique_data_multipliers(xs, base, RATIO[key])
        plot_multiplier(axes[1], xs, mults, extra, key, z=z)
    axes[1].set_ylabel("Unique-data multiplier")
    axes[1].set_xlabel("Unique training tokens (M)")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(["19", "37", "75", "150", "300"])
    axes[1].set_ylim(0.9, 2.15)
    axes[1].legend(frameon=True, fancybox=False, edgecolor="#cccccc", fontsize=8.0, loc="upper right")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "paper_scaling_unique.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig_unique(a.out)


if __name__ == "__main__":
    main()
