#!./.venv/bin/python
"""Figures for the reasoning-replay runs, saved as png under figures/. Run: uv run python plots.py [figure name]  (no name regenerates every figure)"""

import json
import math
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from run import RUNS, load_records, run_dir, run_key
from variants import VARIANTS, dating_equivalent_variants, resolve

# fixed categorical order, one hue per behavior across every panel. First 8 are the validated palette; the rest cover the 18-behavior "all" runs
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194"]
INK = "#52514e"


def tint(hex_color: str, amount: float = 0.5) -> tuple[float, float, float]:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(c + (1 - c) * amount for c in rgb)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials"""
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half


def rates(name: str) -> dict[str, dict[bool, tuple[float | None, int, int]]]:
    """behavior -> reasoning_enabled -> (rate or None if no samples, n matched, n sampled, n quota)"""
    cfg = json.load(open(os.path.join(run_dir(name), "config.json")))
    n_targets = Counter(t["behavior_id"] for t in json.load(open(os.path.join(run_dir(name), "targets.json"))))
    records = load_records(run_key(name))
    out = {}
    for b in sorted(n_targets):
        out[b] = {}
        for on, quota in ((False, cfg["n_off"]), (True, cfg["n_on"])):
            matches = [r["judge_match"] for r in records if r["behavior_id"] == b and r["reasoning_enabled"] == on]
            out[b][on] = (sum(matches) / len(matches) if matches else None, sum(matches), len(matches), quota * n_targets[b])
    return out


def draw_panel(ax, name: str, color: dict[str, str]) -> None:
    if not os.path.exists(os.path.join(run_dir(name), "targets.json")):
        ax.set_title(f"{name}  ({RUNS[name]['model']})", fontsize=9, loc="left")
        ax.text(0.5, 0.5, "no records yet", ha="center", va="center", transform=ax.transAxes, color=INK)
        ax.set_xticks([])
        return
    r = rates(name)
    for i, (b, conds) in enumerate(r.items()):
        for on, x in ((False, i - 0.2), (True, i + 0.2)):
            rate, k, n, quota = conds[on]
            if rate is None:
                ax.text(x, 0.01, "no\ndata", ha="center", va="bottom", fontsize=6, color=INK, rotation=90)
                continue
            partial = n < quota
            lo, hi = wilson(k, n)
            ax.bar(x, rate, 0.38, color=color[b] if on else tint(color[b]), hatch="////" if partial else None, edgecolor="white" if partial else None, linewidth=0)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=INK, linewidth=0.8, capsize=2)
            ax.text(x, hi + 0.01, f"{rate:.2f}" + (f"\n{n}/{quota}" if partial else ""), ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(r)), [b.replace("-", "\n") for b in r], fontsize=7)
    cfg = json.load(open(os.path.join(run_dir(name), "config.json")))
    ax.set_title(f"{name}  ({RUNS[name]['model']})\nquota per prompt: off={cfg['n_off']} on={cfg['n_on']}, {cfg['n_prompts']} prompts per behavior", fontsize=9, loc="left")


def plot_rates(names: list[str], out: str) -> None:
    behaviors = sorted({b for name in RUNS if os.path.exists(os.path.join(run_dir(name), "targets.json")) for b in rates(name)})  # union over every run so a behavior keeps its color across figures
    assert len(behaviors) <= len(COLORS), f"{len(behaviors)} behaviors but only {len(COLORS)} colors"
    color = dict(zip(behaviors, COLORS))
    wide = [name for name in names if RUNS[name]["behaviors"] == "all"]  # full-width row
    narrow = [name for name in names if name not in wide]
    nrows = -(-len(narrow) // 3) + len(wide)
    fig = plt.figure(figsize=(18, 4.2 * nrows))
    axes = [fig.add_subplot(nrows, 3, i + 1) for i in range(len(narrow))] + [fig.add_subplot(nrows, 1, -(-len(narrow) // 3) + 1 + i) for i in range(len(wide))]
    for ax, name in zip(axes, narrow + wide):
        draw_panel(ax, name, color)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("elicitation rate", fontsize=8)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Patch(color=tint(INK), label="reasoning off"), Patch(color=INK, label="reasoning on"), Patch(facecolor=tint(INK), hatch="////", edgecolor="white", linewidth=0, label="incomplete (sampled/quota shown)")]
    fig.legend(handles=handles, loc="upper right", ncols=3, fontsize=9, frameon=False)
    fig.suptitle("Per-behavior elicitation rate, reasoning off vs on (95% Wilson CIs)", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"figures/{out}.png", dpi=160)
    print(f"saved figures/{out}.png ({len(names)} runs)")


def plot_variants(names: list[str], out: str) -> None:
    """One bar pair (reasoning off/on) per prompt variant, with the base prompt's pooled rate over every non-variant run of the model as baseline."""
    model = VARIANTS[names[0]]["model"].split("/")[1]
    base_run, base_idx = VARIANTS[names[0]]["base"].rsplit("/", 1)
    prompt_id = load_records(base_run)[int(base_idx)]["prompt_id"]
    baseline = [r for d in os.listdir(f"results/{model}") if not d.startswith("v_") and os.path.exists(f"results/{model}/{d}/records.jsonl") for r in load_records(f"{model}/{d}") if r.get("prompt_id") == prompt_id]
    columns = [("baseline", baseline, None)] + [(name.split("_")[-1], load_records(f"{model}/{name}"), VARIANTS[name]) for name in names]
    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 2, 4.5))
    for i, (label, records, cfg) in enumerate(columns):
        for on, x in ((False, i - 0.2), (True, i + 0.2)):
            matches = [r["judge_match"] for r in records if r["reasoning_enabled"] == on]
            if not matches:
                ax.text(x, 0.01, "no\ndata", ha="center", va="bottom", fontsize=6, color=INK, rotation=90)
                continue
            rate, k, n = sum(matches) / len(matches), sum(matches), len(matches)
            quota = cfg["n_on" if on else "n_off"] if cfg else n
            lo, hi = wilson(k, n)
            ax.bar(x, rate, 0.38, color=COLORS[0] if on else tint(COLORS[0]), hatch="////" if n < quota else None, edgecolor="white" if n < quota else None, linewidth=0)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=INK, linewidth=0.8, capsize=2)
            ax.text(x, hi + 0.01, f"{rate:.2f}\n{n}/{quota}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axvline(0.5, color=INK, linewidth=0.6, linestyle="--")
    ax.set_xticks(range(len(columns)), [c[0] for c in columns], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("elicitation rate", fontsize=8)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(color=tint(COLORS[0]), label="reasoning off"), Patch(color=COLORS[0], label="reasoning on"), Patch(facecolor=tint(COLORS[0]), hatch="////", edgecolor="white", linewidth=0, label="incomplete")]
    ax.legend(handles=handles, loc="upper right", ncols=3, fontsize=8, frameon=False)
    ax.set_title(f"{resolve(VARIANTS[names[0]])[0]} on {model}: base prompt (pooled over unpinned runs) vs meaning-preserving paraphrases (provider={VARIANTS[names[0]]['provider']}), 95% Wilson CIs", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(f"figures/{out}.png", dpi=160)
    print(f"saved figures/{out}.png ({len(names)} variants)")


ELO_RUNS = [name for name, cfg in RUNS.items() if cfg.get("rank_by", "elo") == "elo"]
FIGURES = {
    "smoke_combined": lambda: plot_rates(["dv4f_smoke", "q36_27b_smoke", "q36_27b_elo", "q36_35b_smoke", "gemma_elo", "inkling_smoke"], "smoke_combined"),
    "plots_elo": lambda: plot_rates(ELO_RUNS, "plots_elo"),
    "dating_equivalents": lambda: plot_variants(list(dating_equivalent_variants), "dating_equivalents"),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
