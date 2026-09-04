#!./.venv/bin/python
"""Reasoning-replay runs over WeirdChat behaviors. Each named run in RUNS replays the best prompts
of a subject model's top patterns, reasoning-off (control) then reasoning-on, and judges every
sample with the behavior's rubric judge.

Results live in results/<model>/<run>/: config.json (the RUNS entry), targets.json (the chosen
prompts, frozen on first run), records.jsonl (one judged sample per line, appended as it finishes).
Reruns sample only the per-(prompt, condition) deficit against the quotas, so raising n_off/n_on in
RUNS extends a run and provider can be swapped (each record stores the provider that served it); changing anything else for an existing run dir is an error.

Run: uv run python run.py <run name>
"""

# pyright: basic

import json
import os
import sys
from collections import Counter

import anyio

import weirdchat as wc
from weirdchat.types import Pattern, Prompt
from utils import load_records, sample_and_judge, gather_bar, judges_for, probe_reasoning, purple, cyan, yellow, red, gray, endc

RUNS = {
    "dv4f_smoke": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors=["fabricated-code-execution", "denying-ai-identity", "chemtrails-assertion"], rank_by="rate", n_prompts=1, n_off=16, n_on=50),
    "dv4f_full": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors="all", rank_by="rate", n_prompts=6, n_off=240, n_on=240),
    "dv4f_full_elo": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors="all", n_prompts=6, n_off=240, n_on=240),
    "inkling_full_elo": dict(model="thinkingmachines/inkling", provider=None, behaviors="all", n_prompts=3, n_off=240, n_on=240, max_tokens_on=32768),
    "q36_27b_smoke": dict(model="qwen/qwen3.6-27b", provider="Alibaba", behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911"], rank_by="rate", n_prompts=3, n_off=32, n_on=64),
    "q36_27b_elo": dict(model="qwen/qwen3.6-27b", provider="Alibaba", behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911"], n_prompts=6, n_off=32, n_on=64),
    "q36_35b_smoke": dict(model="qwen/qwen3.6-35b-a3b", provider=None, behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "chemtrails-assertion", "claims-called-911", "fabricated-code-execution"], rank_by="rate", n_prompts=3, n_off=32, n_on=64),
    "dv4f_unexp": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors=["cutting-instructions", "suicide-jumping"], rank_by="unexpectedness", n_prompts=3, n_off=256, n_on=256),
    "inkling_smoke": dict(model="thinkingmachines/inkling", provider=None, behaviors=["false-physical-embodiment", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911", "unsolicited-sexual-advances"], n_prompts=3, n_off=32, n_on=64, max_tokens_on=32768),
    "dv4f_elo": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911"], n_prompts=3, n_off=32, n_on=64),
    "q36_35b_elo": dict(model="qwen/qwen3.6-35b-a3b", provider=None, behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911"], n_prompts=3, n_off=32, n_on=64),
    "gemma_elo": dict(model="google/gemma-4-31b-it", provider=None, behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911"], n_prompts=3, n_off=32, n_on=64),
    "nemotron_elo": dict(model="nvidia/nemotron-3-ultra-550b-a55b", provider=None, behaviors=["cutting-instructions", "denying-ai-identity", "chemtrails-assertion", "claims-called-911"], n_prompts=3, n_off=32, n_on=64),
    "q36_27b_z": dict(model="qwen/qwen3.6-27b", provider="Alibaba", behaviors=["chemtrails-assertion", "claims-called-911", "cutting-instructions", "denying-ai-identity", "extreme-calorie-restriction", "false-physical-embodiment", "unsolicited-sexual-advances"], rank_by="z", n_prompts=4, n_off=128, n_on=256),
}
BLACKLIST = {"fabricated-code-execution"}  # never targeted, even when a run lists it or says "all"
CONCURRENCY = 96


def run_key(name: str) -> str:
    return os.path.join(RUNS[name]["model"].split("/")[1], name)


def run_dir(name: str) -> str:
    return os.path.join("results", run_key(name))


def zscore(xs: list[float]) -> list[float]:
    mean = sum(xs) / len(xs)
    std = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
    return [(x - mean) / std if std > 0 else 0.0 for x in xs]


def pick_targets(cfg: dict) -> list[tuple[Pattern, Prompt]]:
    behaviors = sorted({p.behavior_id for p in wc.patterns(subject_model=cfg["model"])}) if cfg["behaviors"] == "all" else cfg["behaviors"]
    behaviors = [b for b in behaviors if b not in BLACKLIST]
    targets = []
    for behavior_id in behaviors:
        pats = [p for p in wc.patterns(behavior_id=behavior_id, subject_model=cfg["model"]) if p.openrouter_replication and p.openrouter_replication.rate is not None]
        rank_by = cfg.get("rank_by", "elo")
        if rank_by == "z":  # z(mean Elo) + z(OR rate), each standardized across the behavior's patterns
            score = {p.pattern_id: ze + zr for p, ze, zr in zip(pats, zscore([p.elo.mean for p in pats]), zscore([p.openrouter_replication.rate for p in pats]))}
        else:
            score = {p.pattern_id: s for p in pats if (s := p.elo.mean if rank_by == "elo" else p.elo.unexpectedness.elo if rank_by == "unexpectedness" else p.openrouter_replication.rate) is not None}
        pairs = []
        for pattern in pats:
            ranked = sorted(wc.prompts(pattern), key=lambda pr: pr.match_summary.n_matched / pr.match_summary.n_samples, reverse=True)
            pairs.extend((rank, pattern, pr) for rank, pr in enumerate(ranked))
        pairs.sort(key=lambda t: (t[0], -score[t[1].pattern_id]))  # best prompt of each pattern first, patterns by mean Elo (default), unexpectedness Elo (rank_by="unexpectedness"), OR rate (rank_by="rate") or z(Elo)+z(OR rate) (rank_by="z")
        targets.extend((pattern, prompt) for _, pattern, prompt in pairs[:cfg["n_prompts"]])
    return targets


def load_targets(name: str, cfg: dict) -> list[tuple[Pattern, Prompt]]:
    path = os.path.join(run_dir(name), "targets.json")
    if not os.path.exists(path):
        return pick_targets(cfg)
    targets = []
    for t in json.load(open(path)):
        pattern = wc.pattern(t["pattern_id"])
        targets.append((pattern, next(pr for pr in wc.prompts(pattern) if pr.prompt_id == t["prompt_id"])))
    return targets


def report(name: str, cfg: dict, records: list[dict]) -> None:
    print(f"{purple}=== {name} ==={endc}")
    for behavior_id in sorted({r["behavior_id"] for r in records}):
        off = [r for r in records if r["behavior_id"] == behavior_id and not r["reasoning_enabled"]]
        on = [r for r in records if r["behavior_id"] == behavior_id and r["reasoning_enabled"]]
        off_rate = sum(r["judge_match"] for r in off) / len(off) if off else float("nan")
        on_rate = sum(r["judge_match"] for r in on) / len(on) if on else float("nan")
        ratio = f"{on_rate / off_rate:.2f}" if off and on and off_rate > 0 else "--"
        rtoks = [r["reasoning_tokens"] for r in on if r["reasoning_tokens"]]
        mean_rtoks = f"{sum(rtoks) / len(rtoks):.0f}" if rtoks else "--"
        print(f"  {cyan}{behavior_id:35s} off={off_rate:.3f} (n={len(off):3d})  on={on_rate:.3f} (n={len(on):3d})  on/off={ratio}{endc} {gray}mean_reasoning_toks={mean_rtoks}{endc}")


async def main(name: str) -> None:
    cfg = RUNS[name]
    config_path = os.path.join(run_dir(name), "config.json")
    if os.path.exists(config_path):
        saved = json.load(open(config_path))
        changed = [k for k in cfg if k not in ("n_off", "n_on", "provider") and saved.get(k) != cfg[k]]
        if changed:
            raise SystemExit(f"{red}run {name} exists with different config for {changed}; pick a new run name{endc}")

    records = load_records(run_key(name))
    done = Counter((r["prompt_id"], r["reasoning_enabled"]) for r in records)
    targets = load_targets(name, cfg)
    behavior_ids = sorted({pattern.behavior_id for pattern, _ in targets})
    print(f"  {gray}{len(targets)} target prompts over {len(behavior_ids)} behaviors; {len(records)} records already in {run_dir(name)}{endc}")

    deficit = {tag: sum(max(n - done[(prompt.prompt_id, on)], 0) for _, prompt in targets) for tag, on, n in (("off", False, cfg["n_off"]), ("on", True, cfg["n_on"]))}
    print(f"{purple}=== config ==={endc}")
    for k, v in cfg.items():
        print(f"  {cyan}{k:10s}{endc} {v}")
    print(f"  {cyan}{'to sample':10s}{endc} off={deficit['off']} on={deficit['on']}")
    print(f"{yellow}proceed? [y/n] {endc}", end="", flush=True)
    if input().strip().lower() != "y":
        raise SystemExit(f"{red}aborted{endc}")
    os.makedirs(run_dir(name), exist_ok=True)
    json.dump(cfg, open(config_path, "w"), indent=2)
    json.dump([{"behavior_id": pa.behavior_id, "pattern_id": pa.pattern_id, "prompt_id": pr.prompt_id} for pa, pr in targets], open(os.path.join(run_dir(name), "targets.json"), "w"), indent=2)

    if deficit["off"] or deficit["on"]:
        await probe_reasoning(cfg)

    judges = {b: judges_for(b) for b in behavior_ids}
    with open(os.path.join(run_dir(name), "records.jsonl"), "a") as out:
        for reasoning_enabled, n_samples in ((False, cfg["n_off"]), (True, cfg["n_on"])):
            tag = "on" if reasoning_enabled else "off"
            flat = [(pattern, prompt) for pattern, prompt in targets for _ in range(max(n_samples - done[(prompt.prompt_id, reasoning_enabled)], 0))]
            if not flat:
                continue
            print(f"{purple}=== reasoning {tag.upper()}: {len(flat)} samples to reach {n_samples}/prompt ==={endc}")
            ok = await gather_bar([sample_and_judge(cfg, list(prompt.messages), judges[pattern.behavior_id], reasoning_enabled, {"behavior_id": pattern.behavior_id, "pattern_id": pattern.pattern_id, "prompt_id": prompt.prompt_id}, out) for pattern, prompt in flat], CONCURRENCY, f"{cyan}{name} reasoning {tag}{endc}")
            if None in ok:
                print(f"{yellow}{ok.count(None)} samples dropped (sample or judge failure) -- rerun to fill in{endc}")
            records = load_records(run_key(name))
            if not reasoning_enabled and not any(r["judge_match"] for r in records if not r["reasoning_enabled"]):
                raise SystemExit(f"{red}control replay elicited nothing on any behavior -- replay/judge/provider is broken, fix before spending on reasoning-on{endc}")

    report(name, cfg, records)


if __name__ == "__main__":
    anyio.run(main, sys.argv[1])
