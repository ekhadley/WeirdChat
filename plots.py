#!./.venv/bin/python
"""Figures for the reasoning-replay runs, saved as png under figures/. Run: uv run python plots.py [figure name]  (no name regenerates every figure)"""

import difflib
import json
import math
import os
import re
import sys
from collections import Counter
from html import escape

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from run import RUNS, load_records, run_dir, run_key
from utils import BG, INK, record_messages
from variants import dating_custom_variants, dating_equivalent_variants, inkling_support_equivalent_variants, q27_dating_custom_variants, q27_dating_equivalent_variants, support_custom_variants, resolve

# fixed categorical order, one hue per behavior across every panel. First 8 are the validated palette; the rest cover the 18-behavior "all" runs
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194"]
GRID = "#33343a"
LABELS = {"no_dating_interrogative": "no_dating + interrogative", "named_plain_interrogative": "named_plain + interrogative", "direct_human_or": "direct_address + human_or"}  # variant name suffix -> bar label, when the suffix itself won't do


def tint(hex_color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """blend a color toward the background"""
    rgb, bg = ([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in (hex_color, BG))
    return tuple(c + (b - c) * amount for c, b in zip(rgb, bg))


def css(rgb: tuple[float, float, float]) -> str:
    return "rgb(%d,%d,%d)" % tuple(round(c * 255) for c in rgb)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials"""
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return min(max(center - half, 0.0), p), max(min(center + half, 1.0), p)  # clamped: at k=0 or k=n rounding can put a bound a float epsilon outside [0, 1] or on the wrong side of p


def rates(name: str) -> dict[str, dict[bool, tuple[float | None, int, int, int]]]:
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


def draw_panel(ax, name: str, color: dict[str, str], sep: str = "\n", head: bool = True) -> None:
    """one run's bar pairs; `head=False` leaves the run name and model out of the panel title (the figure title already says it)"""
    if not os.path.exists(os.path.join(run_dir(name), "targets.json")):
        ax.set_title(f"{name}  ({RUNS[name]['model']})", fontsize=13.5, loc="left")
        ax.text(0.5, 0.5, "no records yet", ha="center", va="center", transform=ax.transAxes, color=INK)
        ax.set_xticks([])
        return
    r = rates(name)
    for i, (b, conds) in enumerate(r.items()):
        for on, x in ((False, i - 0.2), (True, i + 0.2)):
            rate, k, n, quota = conds[on]
            if rate is None:
                ax.text(x, 0.01, "no\ndata", ha="center", va="bottom", fontsize=9, color=INK, rotation=90)
                continue
            partial = n < quota
            lo, hi = wilson(k, n)
            ax.bar(x, rate, 0.38, color=color[b] if on else tint(color[b]), hatch="////" if partial else None, edgecolor=BG if partial else None, linewidth=0)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=INK, linewidth=0.8, capsize=2)
            ax.text(x, hi + 0.01, f"{rate:.3f}" + (f"\n{n}/{quota}" if partial else ""), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(r)), [b.replace("-", "\n") for b in r], fontsize=10.5)
    cfg = json.load(open(os.path.join(run_dir(name), "config.json")))
    ax.set_title((f"{name}  ({RUNS[name]['model']}){sep}" if head else "") + f"quota per prompt: off={cfg['n_off']} on={cfg['n_on']}, {cfg['n_prompts']} prompts per behavior", fontsize=13.5, loc="left")


def plot_rates(names: list[str], out: str, ncols: int = 3) -> None:
    behaviors = sorted({b for name in RUNS if os.path.exists(os.path.join(run_dir(name), "targets.json")) for b in rates(name)})  # union over every run so a behavior keeps its color across figures
    assert len(behaviors) <= len(COLORS), f"{len(behaviors)} behaviors but only {len(COLORS)} colors"
    color = dict(zip(behaviors, COLORS))
    wide = [name for name in names if RUNS[name]["behaviors"] == "all"] if ncols > 1 else []  # full-width row; at ncols=1 every panel is already full width, so keep the given order
    narrow = [name for name in names if name not in wide]
    nrows = -(-len(narrow) // ncols) + len(wide)
    fig = plt.figure(figsize=(18, 4.2 * nrows + 0.6 * (len(names) == 1)))  # a single-run figure gets room for its subtitle
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(len(narrow))] + [fig.add_subplot(nrows, 1, -(-len(narrow) // ncols) + 1 + i) for i in range(len(wide))]
    for ax, name in zip(axes, narrow + wide):
        draw_panel(ax, name, color, sep="  " if ncols == 1 or name in wide else "\n", head=len(names) > 1)  # full-width panels fit the run and quota on one title line
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("elicitation rate", fontsize=12)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Patch(color=tint(INK), label="reasoning off"), Patch(color=INK, label="reasoning on")]
    generic = "Per-behavior elicitation rate, reasoning off vs on (95% Wilson CIs)"
    if len(names) == 1:  # one run: its name and model head the figure, the generic title becomes the subtitle
        fig.suptitle(f"{names[0]}  ({RUNS[names[0]]['model']})", fontsize=18, x=0.02, y=0.99, ha="left")
        fig.text(0.02, 0.905, generic, fontsize=13.5, ha="left", color=INK)
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, 0.94), ncols=2, fontsize=13.5, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
    else:
        fig.suptitle(generic, fontsize=18, x=0.02, y=0.995, ha="left")
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, 0.978), ncols=2, fontsize=13.5, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        ax.errorbar(x, rate, yerr=[[max(rate - lo, 0)], [max(hi - rate, 0)]], color=INK, linewidth=0.8, capsize=2)
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
    handles = [Patch(color=tint(COLORS[0]), label="reasoning off"), Patch(color=COLORS[0], label="reasoning on")]
    ax.legend(handles=handles, loc="upper right", ncols=2, fontsize=8, frameon=False)
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


def by_rate(variants: dict[str, dict]) -> list[str]:
    """variant names rising left to right by their measured reasoning-off rate; variants with no samples yet come first"""
    def key(name: str) -> tuple[bool, float]:
        matches = [r["judge_match"] for r in load_records(f"{variants[name]['model'].split('/')[1]}/{name}") if not r["reasoning_enabled"]]
        return (bool(matches), sum(matches) / len(matches) if matches else 0.0)
    return sorted(variants, key=key)


def plot_variants(variants: dict[str, dict], out: str, mean: bool, paraphrases: dict[str, dict] | None = None) -> None:
    """One bar pair (reasoning off/on) per prompt variant, with the base prompt's pooled rate over every non-variant run of the model as baseline.
    `mean` appends the mean of the variant rates (95% CI over the variant means); `paraphrases` inserts the same kind of mean over another variant set after the baseline."""
    names = by_rate(variants)
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


def plot_picked_variants(variants: dict[str, dict], picked: list[str], paraphrases: dict[str, dict], out: str) -> None:
    """plot_variants restricted to the variant name suffixes in `picked`, in that order, after the baseline and paraphrase mean."""
    first = next(iter(variants.values()))
    model = first["model"].split("/")[1]
    prefix = os.path.commonprefix(list(variants)).rsplit("_", 1)[0] + "_"
    base_run, base_idx = first["base"].rsplit("/", 1)
    prompt_id = load_records(base_run)[int(base_idx)]["prompt_id"]
    baseline = [r for d in os.listdir(f"results/{model}") if not d.startswith("v_") and os.path.exists(f"results/{model}/{d}/records.jsonl") for r in load_records(f"{model}/{d}") if r.get("prompt_id") == prompt_id]
    fig, ax = plt.subplots(figsize=(1.1 * (2 + len(picked)) + 3, 5))
    draw_pair(ax, 0, baseline, None)
    draw_mean(ax, 1, paraphrases)
    for i, name in enumerate(picked, 2):
        draw_pair(ax, i, load_records(f"{model}/{prefix}{name}"), variants[prefix + name])
    ax.axvline(1.5, color=INK, linewidth=0.6, linestyle="--")
    finish(fig, ax, ["baseline", "paraphrase mean"] + [LABELS.get(n, n) for n in picked], f"{resolve(first)[0]} on {model}: base prompt (pooled over unpinned runs) vs mean over {len(paraphrases)} meaning-preserving paraphrases, and prompt ablations (provider={first['provider']}), 95% Wilson CIs")
    fig.savefig(f"figures/{out}.png", dpi=160)
    print(f"saved figures/{out}.png ({len(picked)} variants)")


PAGE = """<!doctype html>
<meta charset="utf-8"><title>{out}</title>
<style>
body {{ background: {bg}; color: {ink}; font: 13px system-ui, sans-serif; margin: 24px; }}
h1 {{ font-size: 13px; font-weight: 400; margin: 0 0 14px; }}
#copy {{ position: fixed; top: 14px; right: 16px; width: 30px; height: 30px; padding: 6px; background: {bg}; color: {ink}; border: 1px solid {grid}; border-radius: 6px; cursor: pointer; }}
#copy:hover {{ border-color: {on}; }}
#copy.done {{ color: {green}; border-color: {green}; }}
h2 {{ font-size: 12px; font-weight: 400; margin: 22px 0 0 40px; white-space: pre-line; }}
.legend {{ display: flex; gap: 18px; font-size: 11px; margin: 0 0 10px 40px; }}
.key {{ width: 22px; height: 10px; display: inline-block; vertical-align: -1px; margin-right: 5px; }}
.chart {{ display: flex; align-items: flex-start; padding: 30px 0 16px; overflow-x: auto; }}  /* top room for the caption over a near-1.0 bar; bottom room for a horizontal scrollbar, so a narrow window cannot also trigger a vertical one */  /* the bottom padding leaves room for a horizontal scrollbar, so a narrow window cannot also trigger a vertical one */
.yaxis {{ position: relative; width: 40px; height: 340px; flex: none; }}
.yaxis span {{ position: absolute; right: 6px; bottom: -0.6em; font-size: 11px; }}
.plot {{ position: relative; flex: 1; display: flex; }}
.grid {{ position: absolute; left: 0; right: 0; top: 0; height: 340px; border-bottom: 1px solid {ink}; }}
.tick {{ position: absolute; left: 0; right: 0; border-top: 1px solid {grid}; }}
.yaxis .tick {{ border: 0; }}
.col {{ flex: 1; min-width: 64px; display: flex; flex-direction: column; }}
.col.sep {{ border-left: 1px dashed {ink}; }}
.bars {{ position: relative; height: 340px; display: flex; justify-content: center; align-items: flex-end; gap: 8%; }}
.slot {{ position: relative; width: 36%; height: 100%; }}
.bar {{ position: absolute; left: 0; right: 0; bottom: 0; background: {off}; }}
.bar.on {{ background: {on}; }}
.bar.partial {{ background-image: repeating-linear-gradient(45deg, transparent 0 3px, {bg} 3px 5px); }}
.err {{ position: absolute; left: 50%; width: 1px; background: {ink}; }}
.err::before, .err::after {{ content: ""; position: absolute; left: -3px; width: 7px; height: 1px; background: {ink}; }}
.err::before {{ top: 0; }}
.err::after {{ bottom: 0; }}
.cap {{ position: absolute; left: 50%; transform: translateX(-50%); margin-bottom: 5px; font-size: 9.5px; line-height: 1.15; text-align: center; white-space: nowrap; }}
.labcell {{ position: relative; height: 96px; }}
.lab {{ position: absolute; top: 6px; right: 50%; font-size: 11px; white-space: nowrap; transform: rotate(-30deg); transform-origin: 100% 0; }}
.lab[data-prompt] {{ cursor: pointer; text-decoration: underline dotted {grid}; }}
.lab[data-prompt]:hover {{ color: #fff; }}
.lab.pin {{ color: #fff; text-decoration: underline solid {on}; }}
#tip {{ position: fixed; display: none; max-width: 480px; padding: 9px 11px; background: #000; border: 1px solid {grid}; border-radius: 5px; font-size: 12px; line-height: 1.45; white-space: pre-wrap; pointer-events: none; }}
#tip.pin {{ pointer-events: auto; user-select: text; border-color: {on}; }}
#tip del {{ color: {red}; }}
#tip ins {{ color: {green}; text-decoration: none; }}
</style>
<button id="copy" title="copy link"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
<h1>{title}</h1>
<div class="legend"><span><i class="key" style="background:{off}"></i>reasoning off</span><span><i class="key" style="background:{on}"></i>reasoning on</span></div>
{charts}
<div id="tip"></div>
<script>
const copy = document.getElementById("copy");
copy.onclick = () => navigator.clipboard.writeText("https://ekhadley.net/matsmatsmats/figures/{out}.html").then(() => {{ copy.classList.add("done"); setTimeout(() => copy.classList.remove("done"), 1200); }});

for (const chart of document.querySelectorAll(".chart")) {{  // the angled labels are absolutely positioned, so give their row exactly the height they render at: any less and the chart box grows a scrollbar
    const bottom = Math.max(...[...chart.querySelectorAll(".lab")].map(l => l.getBoundingClientRect().bottom));
    for (const cell of chart.querySelectorAll(".labcell")) cell.style.height = Math.ceil(bottom - cell.getBoundingClientRect().top) + 1 + "px";
}}

const tip = document.getElementById("tip");
let pinned = null;
const show = (lab, e) => {{
    tip.innerHTML = lab.dataset.prompt;
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 12) + "px";
    tip.style.top = Math.min(e.clientY + 16, innerHeight - tip.offsetHeight - 12) + "px";
}};
const unpin = () => {{
    if (pinned) pinned.classList.remove("pin");
    pinned = null;
    tip.classList.remove("pin");
    tip.style.display = "none";
}};
for (const lab of document.querySelectorAll(".lab[data-prompt]")) {{
    lab.onmousemove = e => pinned || show(lab, e);
    lab.onmouseleave = () => pinned || (tip.style.display = "none");
    lab.onclick = e => {{  // click pins the tooltip where it is, so it can be read and copied; clicking it again, another label, or anywhere else unpins
        e.stopPropagation();
        const same = pinned === lab;
        unpin();
        if (!same) {{
            pinned = lab;
            lab.classList.add("pin");
            tip.classList.add("pin");
            show(lab, e);
        }}
    }};
}}
tip.onclick = e => e.stopPropagation();
document.onclick = unpin;
</script>
"""


def diff_html(base: str, other: str) -> str:
    """`other` word by word against `base`: the words it drops struck through in red, the words it adds in green"""
    a, b = (re.findall(r"\s*\S+", t) for t in (base, other))
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            out.append(f"<del>{escape(''.join(a[i1:i2]))}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins>{escape(''.join(b[j1:j2]))}</ins>")
        if op == "equal":
            out.append(escape("".join(b[j1:j2])))
    return "".join(out)


def bar_cells(records: list[dict], cfg: dict | None) -> list[dict]:
    """the HTML twin of draw_pair: per reasoning condition, the rate, its Wilson interval, the caption and whether it is under quota"""
    cells = []
    for on in (False, True):
        matches = [r["judge_match"] for r in records if r["reasoning_enabled"] == on]
        if not matches:
            cells.append(dict(on=on, rate=0.0, lo=0.0, hi=0.0, caption="no<br>data", partial=False))
            continue
        rate, k, n = sum(matches) / len(matches), sum(matches), len(matches)
        lo, hi = wilson(k, n)
        cells.append(dict(on=on, rate=rate, lo=lo, hi=hi, caption=f"{rate:.3f}<br>{k}/{n}", partial=n < (cfg["n_on" if on else "n_off"] if cfg else n)))
    return cells


def mean_cells(variants: dict[str, dict]) -> list[dict]:
    """the HTML twin of draw_mean: the mean of the per-variant rates with a 1.96*sem interval"""
    cells = []
    for on, pairs in variant_rates(variants).items():
        vals = [v for v, _ in pairs]
        m = sum(vals) / len(vals)
        half = 1.96 * math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
        cells.append(dict(on=on, rate=m, lo=m - half, hi=m + half, caption=f"{m:.3f}<br>{len(vals)} prompts", partial=len(vals) < len(variants) or not all(full for _, full in pairs)))
    return cells


def html_column(label: str, cells: list[dict], prompt: str | None, sep: bool = False, color: str | None = None) -> str:
    """one bar-pair column; hovering its label shows `prompt` (already-escaped HTML), `sep` draws the divider on its left edge, and `color` overrides the page's bar hue (the off bar gets its tint)"""
    slots = []
    for c in cells:
        fill = f";background-color:{color if c['on'] else css(tint(color))}" if color else ""  # background-color, not the background shorthand, so a partial bar keeps its hatch
        slots.append(f'<div class="slot"><div class="bar{" on" * c["on"]}{" partial" * c["partial"]}" style="height:{c["rate"]:.1%}{fill}"></div>'
                     f'<i class="err" style="bottom:{c["lo"]:.1%};height:{c["hi"] - c["lo"]:.1%}"></i>'
                     f'<span class="cap" style="bottom:{c["hi"]:.1%}">{c["caption"]}</span></div>')
    tip = f' data-prompt="{escape(prompt, quote=True)}"' if prompt else ""
    return f'<div class="col{" sep" * sep}"><div class="bars">{"".join(slots)}</div><div class="labcell"><span class="lab"{tip}>{escape(label)}</span></div></div>'


def html_chart(columns: list[str], title: str = "") -> str:
    """one bar chart -- y axis, gridlines, the given columns -- under an optional panel title"""
    ys = [i / 5 for i in range(6)]
    ticks = "".join(f'<div class="tick" style="bottom:{y:.0%}"><span>{y:.1f}</span></div>' for y in ys)
    lines = "".join(f'<div class="tick" style="bottom:{y:.0%}"></div>' for y in ys)
    return (f'<h2>{escape(title)}</h2>' if title else "") + f'<div class="chart"><div class="yaxis">{ticks}</div><div class="plot"><div class="grid">{lines}</div>{"".join(columns)}</div></div>'


def write_page(out: str, title: str, charts: str, hue: str = COLORS[0]) -> None:
    """render PAGE to figures/<out>.html; `hue` is the reasoning-on bar and legend color, reasoning off its tint"""
    open(f"figures/{out}.html", "w").write(PAGE.format(out=out, title=escape(title), charts=charts, bg=BG, ink=INK, grid=GRID, off=css(tint(hue)), on=hue, red=COLORS[7], green=COLORS[2]))
    print(f"saved figures/{out}.html")


def plot_variants_html(variants: dict[str, dict], out: str, mean: bool = False, paraphrases: dict[str, dict] | None = None, picked: list[str] | None = None) -> None:
    """HTML twin of plot_variants: the same bars and columns, and hovering a bar's label shows that prompt as a word diff against the base prompt. `picked` restricts to those variant name suffixes, in that order (the twin of plot_picked_variants)."""
    names = [next(n for n in variants if n.endswith("_" + p)) for p in picked] if picked else by_rate(variants)
    first = variants[names[0]]
    prefix = os.path.commonprefix(names).rsplit("_", 1)[0] + "_"
    model = first["model"].split("/")[1]
    base_run, base_idx = first["base"].rsplit("/", 1)
    base = load_records(base_run)[int(base_idx)]
    baseline = [r for d in os.listdir(f"results/{model}") if not d.startswith("v_") and os.path.exists(f"results/{model}/{d}/records.jsonl") for r in load_records(f"{model}/{d}") if r.get("prompt_id") == base["prompt_id"]]
    base_text = "\n\n".join(m.content for m in record_messages(base))
    columns = [html_column("baseline", bar_cells(baseline, None), escape(base_text))]
    if paraphrases:
        columns.append(html_column("paraphrase mean", mean_cells(paraphrases), escape(f"mean over the {len(paraphrases)} meaning-preserving paraphrases of the base prompt")))
    columns += [html_column(LABELS.get(name[len(prefix):], name[len(prefix):]), bar_cells(load_records(f"{model}/{name}"), variants[name]), diff_html(base_text, variants[name]["prompt"]), sep=i == 0) for i, name in enumerate(names)]
    if mean:
        columns.append(html_column("mean", mean_cells(variants), escape(f"mean over the {len(variants)} variant prompts"), sep=True))
    what = f"mean over {len(paraphrases)} meaning-preserving paraphrases, and prompt ablations" if picked else f"prompt variants"
    title = f"{resolve(first)[0]} on {model}: base prompt (pooled over unpinned runs) vs {what} (provider={first['provider']}), 95% Wilson CIs; mean bars: 95% CI over the per-prompt rates. Hover a variant label for its prompt, diffed against the base prompt (red struck-through = dropped, green = added); click to pin it."
    write_page(out, title, html_chart(columns))


def plot_prompts_html(run: str, behavior: str, names: dict[str, str], out: str) -> None:
    """HTML twin of plot_prompts: one bar pair per prompt, and hovering a label shows that prompt in full."""
    records = [r for r in load_records(run) if r["behavior_id"] == behavior]
    cfg = json.load(open(f"results/{run}/config.json"))
    columns = []
    for prefix, label in names.items():
        rs = [r for r in records if r["prompt_id"].startswith(prefix)]
        columns.append(html_column(label, bar_cells(rs, cfg), escape("\n\n".join(m.content for m in record_messages(rs[0])))))
    write_page(out, f"{behavior} on {run}: the run's {len(names)} prompts, 95% Wilson CIs. Hover a label for its prompt; click to pin it.", html_chart(columns))


def rate_cells(conds: dict[bool, tuple[float | None, int, int, int]]) -> list[dict]:
    """the HTML twin of draw_panel's bar pair: one behavior's off/on rates, captioned with sampled/quota when under quota"""
    cells = []
    for on in (False, True):
        rate, k, n, quota = conds[on]
        if rate is None:
            cells.append(dict(on=on, rate=0.0, lo=0.0, hi=0.0, caption="no<br>data", partial=False))
            continue
        lo, hi = wilson(k, n)
        cells.append(dict(on=on, rate=rate, lo=lo, hi=hi, caption=f"{rate:.3f}" + (f"<br>{n}/{quota}" if n < quota else ""), partial=n < quota))
    return cells


def plot_rates_html(names: list[str], out: str) -> None:
    """HTML twin of plot_rates: one chart per run, stacked, one hue per behavior held fixed across every run. Like the PNG, "all"-behavior runs come after the rest, and a single run's name heads the page with the generic title under it."""
    behaviors = sorted({b for name in RUNS if os.path.exists(os.path.join(run_dir(name), "targets.json")) for b in rates(name)})
    color = dict(zip(behaviors, COLORS))
    generic = "Per-behavior elicitation rate, reasoning off vs on (95% Wilson CIs). Bars are hued per behavior; incomplete bars are hatched and captioned sampled/quota."
    charts = []
    for name in sorted(names, key=lambda n: RUNS[n]["behaviors"] == "all"):
        head = f"{name}  ({RUNS[name]['model']})" if len(names) > 1 else generic
        if not os.path.exists(os.path.join(run_dir(name), "targets.json")):
            charts.append(f'<h2>{escape(head)}</h2><div class="chart">no records yet</div>')
            continue
        cfg = json.load(open(os.path.join(run_dir(name), "config.json")))
        columns = [html_column(b, rate_cells(conds), None, color=color[b]) for b, conds in rates(name).items()]
        charts.append(html_chart(columns, f"{head}\nquota per prompt: off={cfg['n_off']} on={cfg['n_on']}, {cfg['n_prompts']} prompts per behavior"))
    write_page(out, generic if len(names) > 1 else f"{names[0]}  ({RUNS[names[0]]['model']})", "".join(charts), hue=INK)


ELO_RUNS = ["dv4f_full_elo", "inkling_full_elo", "q36_27b_elo"]
SMOKE_RUNS = ["dv4f_smoke", "q36_27b_smoke", "q36_27b_elo6", "q36_35b_smoke", "gemma_elo", "inkling_smoke"]
ALL_RUNS = ["dv4f_full_elo", "inkling_full_elo", "q36_27b_z", "gemma_elo", "nemotron_elo"]
Q27B_911_PROMPTS = {"648552fd": "ladder fall", "640da66c": "shot", "6431e95c": "dog attack", "6455e5d0": "overdose"}
INKLING_DENYING_PROMPTS = {"64aae94e": "grief", "64753fcc": "refund", "6454e031": "lonely"}
DENYING_PROMPTS = {"64ec0ce1": "dating app", "64b563fb": "grief", "64b527e1": "work crisis", "64d0d877": "moral dilemma", "6484264e": "poem", "648d52b6": "lonely"}
DV4F_DATING_ABLATIONS = {k: v for k, v in dating_custom_variants.items() if k != "v_dv4f_dating_pretending_one"}

INKLING_SUPPORT_PICKED = ["named_plain", "bots_ok", "direct_address", "human_question", "less_upset", "more_upset", "no_transfer", "direct_human_or"]
DV4F_DATING_PICKED = ["no_dating", "interrogative", "imperative", "named_plain", "not_bot_valence", "bots_ok", "no_dating_interrogative"]

FIGURES = {
    "smoke_combined": lambda: plot_rates(SMOKE_RUNS, "smoke_combined"),
    "smoke_combined_html": lambda: plot_rates_html(SMOKE_RUNS, "smoke_combined"),
    "plots_elo": lambda: plot_rates(ELO_RUNS, "plots_elo", ncols=1),
    "plots_elo_html": lambda: plot_rates_html(ELO_RUNS, "plots_elo"),
    "dv4f_dating_equivalents": lambda: plot_variants(dating_equivalent_variants, "dv4f_dating_equivalents", mean=True),
    "dv4f_dating_equivalents_html": lambda: plot_variants_html(dating_equivalent_variants, "dv4f_dating_equivalents", mean=True),
    "dv4f_dating_ablations": lambda: plot_variants(DV4F_DATING_ABLATIONS, "dv4f_dating_ablations", mean=False, paraphrases=dating_equivalent_variants),
    "inkling_support_ablations_picked": lambda: plot_picked_variants(support_custom_variants, INKLING_SUPPORT_PICKED, inkling_support_equivalent_variants, "inkling_support_ablations_picked"),
    "inkling_support_ablations_picked_html": lambda: plot_variants_html(support_custom_variants, "inkling_support_ablations_picked", paraphrases=inkling_support_equivalent_variants, picked=INKLING_SUPPORT_PICKED),
    "dv4f_dating_ablations_picked_html": lambda: plot_variants_html(dating_custom_variants, "dv4f_dating_ablations_picked", paraphrases=dating_equivalent_variants, picked=DV4F_DATING_PICKED),
    "dv4f_dating_ablations_picked": lambda: plot_picked_variants(dating_custom_variants, DV4F_DATING_PICKED, dating_equivalent_variants, "dv4f_dating_ablations_picked"),
    "dv4f_dating_ablations_html": lambda: plot_variants_html(DV4F_DATING_ABLATIONS, "dv4f_dating_ablations", paraphrases=dating_equivalent_variants),
    "inkling_support_equivalents": lambda: plot_variants(inkling_support_equivalent_variants, "inkling_support_equivalents", mean=True),
    "inkling_support_equivalents_html": lambda: plot_variants_html(inkling_support_equivalent_variants, "inkling_support_equivalents", mean=True),
    "inkling_support_ablations": lambda: plot_variants(support_custom_variants, "inkling_support_ablations", mean=False, paraphrases=inkling_support_equivalent_variants),
    "inkling_support_ablations_html": lambda: plot_variants_html(support_custom_variants, "inkling_support_ablations", paraphrases=inkling_support_equivalent_variants),
    "q27b_dating_equivalents": lambda: plot_variants(q27_dating_equivalent_variants, "q27b_dating_equivalents", mean=True),
    "q27b_dating_equivalents_html": lambda: plot_variants_html(q27_dating_equivalent_variants, "q27b_dating_equivalents", mean=True),
    "q27b_dating_ablations": lambda: plot_variants(q27_dating_custom_variants, "q27b_dating_ablations", mean=False, paraphrases=q27_dating_equivalent_variants),
    "q27b_dating_ablations_html": lambda: plot_variants_html(q27_dating_custom_variants, "q27b_dating_ablations", paraphrases=q27_dating_equivalent_variants),
    "dv4f_denying_prompts": lambda: plot_prompts("deepseek-v4-flash/dv4f_full_elo", "denying-ai-identity", DENYING_PROMPTS, "dv4f_denying_prompts"),
    "dv4f_denying_prompts_html": lambda: plot_prompts_html("deepseek-v4-flash/dv4f_full_elo", "denying-ai-identity", DENYING_PROMPTS, "dv4f_denying_prompts"),
    "q27b_911_prompts": lambda: plot_prompts("qwen3.6-27b/q36_27b_z", "claims-called-911", Q27B_911_PROMPTS, "q27b_911_prompts"),
    "q27b_911_prompts_html": lambda: plot_prompts_html("qwen3.6-27b/q36_27b_z", "claims-called-911", Q27B_911_PROMPTS, "q27b_911_prompts"),
    "inkling_denying_prompts": lambda: plot_prompts("inkling/inkling_full_elo", "denying-ai-identity", INKLING_DENYING_PROMPTS, "inkling_denying_prompts"),
    "inkling_denying_prompts_html": lambda: plot_prompts_html("inkling/inkling_full_elo", "denying-ai-identity", INKLING_DENYING_PROMPTS, "inkling_denying_prompts"),
    "all": lambda: plot_rates(ALL_RUNS, "all"),
    "all_html": lambda: plot_rates_html(ALL_RUNS, "all"),
    **{f"all_{name}": (lambda name=name: plot_rates([name], f"all_{name}", ncols=1)) for name in ALL_RUNS},
    **{f"all_{name}_html": (lambda name=name: plot_rates_html([name], f"all_{name}")) for name in ALL_RUNS},
}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
