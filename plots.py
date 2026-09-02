#!./.venv/bin/python
"""Figures for the reasoning-replay runs, saved as png under figures/. Run: uv run python plots.py <figure name>"""

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from run import RUNS, load_records, run_dir

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # fixed categorical order, one hue per behavior across every panel


def tint(hex_color: str, amount: float = 0.5) -> tuple[float, float, float]:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(c + (1 - c) * amount for c in rgb)


def rates(records: list[dict]) -> dict[str, tuple[float, float, int, int]]:
    out = {}
    for b in sorted({r["behavior_id"] for r in records}):
        off = [r["judge_match"] for r in records if r["behavior_id"] == b and not r["reasoning_enabled"]]
        on = [r["judge_match"] for r in records if r["behavior_id"] == b and r["reasoning_enabled"]]
        out[b] = (sum(off) / len(off), sum(on) / len(on), len(off), len(on))
    return out


def plot_rates(names: list[str], out: str) -> None:
    per_run = {name: rates(load_records(name)) for name in names}
    behaviors = sorted({b for r in per_run.values() for b in r})
    assert len(behaviors) <= len(COLORS), f"{len(behaviors)} behaviors but only {len(COLORS)} colors"
    color = dict(zip(behaviors, COLORS))
    ncols = 3
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        r = per_run[name]
        for i, (b, (off, on, n_off, n_on)) in enumerate(r.items()):
            ax.bar(i - 0.2, off, 0.38, color=tint(color[b]))
            ax.bar(i + 0.2, on, 0.38, color=color[b])
            ax.text(i - 0.2, off + 0.01, f"{off:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
            ax.text(i + 0.2, on + 0.01, f"{on:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
        ax.set_xticks(range(len(r)), [b.replace("-", "\n") for b in r], fontsize=7)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("elicitation rate", fontsize=8)
        n_off, n_on = next(iter(r.values()))[2:]
        ax.set_title(f"{name}  ({RUNS[name]['model']})  samples per behavior: off={n_off} on={n_on}", fontsize=9)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.legend(handles=[Patch(color=tint("#52514e"), label="reasoning off"), Patch(color="#52514e", label="reasoning on")], loc="upper right", ncols=2, fontsize=9, frameon=False)
    fig.suptitle("Per-behavior elicitation rate, reasoning off vs on", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"figures/{out}.png", dpi=160)
    print(f"saved figures/{out}.png ({len(names)} runs)")


FIGURES = {
    "smoke_combined": lambda: plot_rates(["dv4f_smoke", "q36_27b_smoke", "q36_27b_elo", "q36_35b_smoke", "gemma_elo", "inkling_smoke"], "smoke_combined"),
}

if __name__ == "__main__":
    FIGURES[sys.argv[1]]()
