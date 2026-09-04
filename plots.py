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
from utils import BG, INK
from variants import dating_custom_variants, dating_equivalent_variants, inkling_support_equivalent_variants, resolve

# fixed categorical order, one hue per behavior across every panel. First 8 are the validated palette; the rest cover the 18-behavior "all" runs
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194"]
GRID = "#33343a"
LABELS = {"no_dating_interrogative": "no_dating + interrogative", "named_plain_interrogative": "named_plain + interrogative"}  # variant name suffix -> bar label, when the suffix itself won't do


def tint(hex_color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """blend a color toward the background"""
    rgb, bg = ([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in (hex_color, BG))
    return tuple(c + (b - c) * amount for c, b in zip(rgb, bg))


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
            ax.bar(x, rate, 0.38, color=color[b] if on else tint(color[b]), hatch="////" if partial else None, edgecolor=BG if partial else None, linewidth=0)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=INK, linewidth=0.8, capsize=2)
            ax.text(x, hi + 0.01, f"{rate:.3f}" + (f"\n{n}/{quota}" if partial else ""), ha="center", va="bottom", fontsize=6.5, color=INK)
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
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Patch(color=tint(INK), label="reasoning off"), Patch(color=INK, label="reasoning on"), Patch(facecolor=tint(INK), hatch="////", edgecolor=BG, linewidth=0, label="incomplete (sampled/quota shown)")]
    fig.legend(handles=handles, loc="upper right", ncols=3, fontsize=9, frameon=False)
    fig.suptitle("Per-behavior elicitation rate, reasoning off vs on (95% Wilson CIs)", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"figures/{out}.png", dpi=160)
    print(f"saved figures/{out}.png ({len(names)} runs)")


def variant_rates(variants: dict[str, dict]) -> dict[bool, list[tuple[float, bool]]]:
    """reasoning_enabled -> per-variant (match rate, at quota), over the variants with any samples in that condition"""
    out: dict[bool, list[tuple[float, bool]]] = {False: [], True: []}
    for name, cfg in variants.items():
        records = load_records(f"{cfg['model'].split('/')[1]}/{name}")
        for on in (False, True):
            matches = [r["judge_match"] for r in records if r["reasoning_enabled"] == on]
            if matches:
                out[on].append((sum(matches) / len(matches), len(matches) >= cfg["n_on" if on else "n_off"]))
    return out


def draw_mean(ax, i: int, variants: dict[str, dict]) -> None:
    """bar pair at column i: mean of the per-variant rates, CI = 1.96 * sem over them, hatched if any variant is under quota"""
    for on, pairs in variant_rates(variants).items():
        vals = [v for v, _ in pairs]
        x = i + 0.2 if on else i - 0.2
        m = sum(vals) / len(vals)
        half = 1.96 * math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
        partial = len(vals) < len(variants) or not all(full for _, full in pairs)
        ax.bar(x, m, 0.38, color=COLORS[0] if on else tint(COLORS[0]), hatch="////" if partial else None, edgecolor=BG if partial else None, linewidth=0)
        ax.errorbar(x, m, yerr=half, color=INK, linewidth=0.8, capsize=2)
        ax.text(x, m + half + 0.01, f"{m:.3f}\n{len(vals)} prompts", ha="center", va="bottom", fontsize=6.5, color=INK)


def draw_pair(ax, i: int, records: list[dict], cfg: dict | None) -> None:
    """reasoning off/on bar pair at column i with Wilson CIs, hatched when under cfg's quota (no cfg: never hatched)"""
    for on, x in ((False, i - 0.2), (True, i + 0.2)):
        matches = [r["judge_match"] for r in records if r["reasoning_enabled"] == on]
        if not matches:
            ax.text(x, 0.01, "no\ndata", ha="center", va="bottom", fontsize=6, color=INK, rotation=90)
            continue
        rate, k, n = sum(matches) / len(matches), sum(matches), len(matches)
        quota = cfg["n_on" if on else "n_off"] if cfg else n
        lo, hi = wilson(k, n)
        ax.bar(x, rate, 0.38, color=COLORS[0] if on else tint(COLORS[0]), hatch="////" if n < quota else None, edgecolor=BG if n < quota else None, linewidth=0)
        ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=INK, linewidth=0.8, capsize=2)
        ax.text(x, hi + 0.01, f"{rate:.3f}\n{k}/{n}", ha="center", va="bottom", fontsize=6.5, color=INK)


def finish(fig, ax, labels: list[str], title: str) -> None:
    """ticks, axes, legend, title shared by the per-prompt bar figures"""
    ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("elicitation rate", fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(color=tint(COLORS[0]), label="reasoning off"), Patch(color=COLORS[0], label="reasoning on"), Patch(facecolor=tint(COLORS[0]), hatch="////", edgecolor=BG, linewidth=0, label="incomplete")]
    ax.legend(handles=handles, loc="upper right", ncols=3, fontsize=8, frameon=False)
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()


def plot_prompts(run: str, behavior: str, names: dict[str, str], out: str) -> None:
    """one bar pair per prompt of `behavior` in replay run `run`; `names` maps a prompt_id prefix to its bar label and fixes the order"""
    records = [r for r in load_records(run) if r["behavior_id"] == behavior]
    cfg = json.load(open(f"results/{run}/config.json"))
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 5))
    for i, prefix in enumerate(names):
        draw_pair(ax, i, [r for r in records if r["prompt_id"].startswith(prefix)], cfg)
    finish(fig, ax, list(names.values()), f"{behavior} on {run}: the run's {len(names)} prompts, 95% Wilson CIs")
    fig.savefig(f"figures/{out}.png", dpi=160)
    print(f"saved figures/{out}.png")


def plot_variants(variants: dict[str, dict], out: str, mean: bool, paraphrases: dict[str, dict] | None = None) -> None:
    """One bar pair (reasoning off/on) per prompt variant, with the base prompt's pooled rate over every non-variant run of the model as baseline.
    `mean` appends the mean of the variant rates (95% CI over the variant means); `paraphrases` inserts the same kind of mean over another variant set after the baseline."""
    names = list(variants)
    first = variants[names[0]]
    prefix = os.path.commonprefix(names).rsplit("_", 1)[0] + "_"
    model = first["model"].split("/")[1]
    base_run, base_idx = first["base"].rsplit("/", 1)
    prompt_id = load_records(base_run)[int(base_idx)]["prompt_id"]
    baseline = [r for d in os.listdir(f"results/{model}") if not d.startswith("v_") and os.path.exists(f"results/{model}/{d}/records.jsonl") for r in load_records(f"{model}/{d}") if r.get("prompt_id") == prompt_id]
    columns: list[tuple] = [("baseline", baseline, None)] + ([("paraphrase mean", paraphrases)] if paraphrases else []) + [(LABELS.get(name[len(prefix):], name[len(prefix):]), load_records(f"{model}/{name}"), variants[name]) for name in names] + ([("mean", variants)] if mean else [])
    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 3, 5))
    for i, column in enumerate(columns):
        if len(column) == 2:
            draw_mean(ax, i, column[1])
            continue
        draw_pair(ax, i, column[1], column[2])
    for xline in [1.5 if paraphrases else 0.5] + [len(columns) - 1.5] * mean:
        ax.axvline(xline, color=INK, linewidth=0.6, linestyle="--")
    finish(fig, ax, [c[0] for c in columns], f"{resolve(first)[0]} on {model}: base prompt (pooled over unpinned runs) vs prompt variants (provider={first['provider']}), 95% Wilson CIs; mean bars: 95% CI over the per-prompt rates")
    fig.savefig(f"figures/{out}.png", dpi=160)
    print(f"saved figures/{out}.png ({len(names)} variants)")


ELO_RUNS = [name for name, cfg in RUNS.items() if cfg.get("rank_by", "elo") == "elo"]
Q27B_911_PROMPTS = {"648552fd": "ladder fall", "640da66c": "shot", "6431e95c": "dog attack", "6455e5d0": "overdose"}
INKLING_DENYING_PROMPTS = {"64aae94e": "grief", "64753fcc": "refund", "6454e031": "lonely"}
DENYING_PROMPTS = {"64ec0ce1": "dating app", "64b563fb": "grief", "64b527e1": "work crisis", "64d0d877": "moral dilemma", "6484264e": "poem", "648d52b6": "lonely"}

FIGURES = {
    "smoke_combined": lambda: plot_rates(["dv4f_smoke", "q36_27b_smoke", "q36_27b_elo", "q36_35b_smoke", "gemma_elo", "inkling_smoke"], "smoke_combined"),
    "plots_elo": lambda: plot_rates(ELO_RUNS, "plots_elo"),
    "dating_equivalents": lambda: plot_variants(dating_equivalent_variants, "dating_equivalents", mean=True),
    "support_equivalents": lambda: plot_variants(inkling_support_equivalent_variants, "support_equivalents", mean=True),
    "dating_targeted": lambda: plot_variants({k: v for k, v in dating_custom_variants.items() if k != "v_dv4f_dating_pretending_one"}, "dating_targeted", mean=False, paraphrases=dating_equivalent_variants),
    "dv4f_denying_prompts": lambda: plot_prompts("deepseek-v4-flash/dv4f_full_elo", "denying-ai-identity", DENYING_PROMPTS, "dv4f_denying_prompts"),
    "q27b_911_prompts": lambda: plot_prompts("qwen3.6-27b/q36_27b_z", "claims-called-911", Q27B_911_PROMPTS, "q27b_911_prompts"),
    "inkling_denying_prompts": lambda: plot_prompts("inkling/inkling_full_elo", "denying-ai-identity", INKLING_DENYING_PROMPTS, "inkling_denying_prompts"),
    "main": lambda: plot_rates(["dv4f_full_elo", "inkling_full_elo", "q36_27b_z", "gemma_elo"], "main"),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
