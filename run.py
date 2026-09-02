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
from dotenv import load_dotenv
from openai import AsyncOpenAI

import weirdchat as wc
from weirdchat.judge import JudgeError
from weirdchat.types import Message, Pattern, Prompt

load_dotenv()

RUNS = {
    "dv4f_smoke": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors=["fabricated-code-execution", "denying-ai-identity", "chemtrails-assertion"], rank_by="rate", n_prompts=1, n_off=16, n_on=50, price_in=0.086, price_out=0.171),
    "dv4f_full": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors="all", rank_by="rate", n_prompts=6, n_off=240, n_on=240, price_in=0.0763, price_out=0.1526),
    "q36_27b_smoke": dict(model="qwen/qwen3.6-27b", provider="Alibaba", behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911"], rank_by="rate", n_prompts=3, n_off=32, n_on=64, price_in=0.45, price_out=2.70),
    "q36_27b_elo": dict(model="qwen/qwen3.6-27b", provider="Alibaba", behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911"], n_prompts=6, n_off=32, n_on=64, price_in=0.45, price_out=2.70),
    "q36_35b_smoke": dict(model="qwen/qwen3.6-35b-a3b", provider=None, behaviors=["false-physical-embodiment", "cutting-instructions", "denying-ai-identity", "chemtrails-assertion", "claims-called-911", "fabricated-code-execution"], rank_by="rate", n_prompts=3, n_off=32, n_on=64, price_in=0.15, price_out=1.0),
    "dv4f_unexp": dict(model="deepseek/deepseek-v4-flash", provider=None, behaviors=["cutting-instructions", "suicide-jumping"], rank_by="unexpectedness", n_prompts=3, n_off=256, n_on=256, price_in=0.0855, price_out=0.171),
    "inkling_smoke": dict(model="thinkingmachines/inkling", provider=None, behaviors=["false-physical-embodiment", "denying-ai-identity", "fabricated-code-execution", "chemtrails-assertion", "claims-called-911", "unsolicited-sexual-advances"], n_prompts=3, n_off=32, n_on=64, price_in=1.0, price_out=4.05),
}
TEMPERATURE = 1.0
MAX_TOKENS_OFF = 1024  # the original pipeline's response cap
MAX_TOKENS_ON = 8192  # reasoning bills as completion tokens; the trace needs headroom beyond the visible 1024
CONCURRENCY = 128

purple = '\x1b[38;2;255;0;255m'
cyan = '\x1b[38;2;0;255;255m'
yellow = '\x1b[38;2;255;255;0m'
green = '\x1b[38;2;0;255;0m'
red = '\x1b[38;2;255;0;0m'
gray = '\x1b[38;2;127;127;127m'
endc = '\033[0m'

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=300.0, max_retries=3)


def run_dir(name: str) -> str:
    return os.path.join("results", RUNS[name]["model"].split("/")[1], name)


def load_records(name: str) -> list[dict]:
    path = os.path.join(run_dir(name), "records.jsonl")
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


async def sample_once(cfg: dict, messages: list[Message], reasoning_enabled: bool) -> dict | None:
    extra: dict = {"reasoning": {"enabled": reasoning_enabled}}
    if cfg["provider"]:
        extra["provider"] = {"order": [cfg["provider"]], "allow_fallbacks": False}
    last = "unknown error"
    for attempt in range(3):
        if attempt:
            await anyio.sleep(2**attempt)
        try:
            reply = await client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_ON if reasoning_enabled else MAX_TOKENS_OFF,
                extra_body=extra,
            )
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            continue
        msg = reply.choices[0].message
        if not (msg.content or "").strip():
            last = "empty response"
            continue
        details = getattr(reply.usage, "completion_tokens_details", None)
        return {
            "response": msg.content,
            "reasoning": (msg.model_extra or {}).get("reasoning"),
            "provider": (reply.model_extra or {}).get("provider"),
            "served_model": reply.model,
            "prompt_tokens": reply.usage.prompt_tokens,
            "completion_tokens": reply.usage.completion_tokens,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        }
    print(f"{red}sample failed after 3 attempts: {last}{endc}")
    return None


async def sample_and_judge(cfg: dict, pattern: Pattern, prompt: Prompt, judge: wc.RubricJudge, reasoning_enabled: bool, out) -> bool:
    s = await sample_once(cfg, prompt.messages, reasoning_enabled)
    if s is None:
        return False
    try:
        j = await judge.judge(list(prompt.messages) + [Message(role="assistant", content=s["response"])])
    except JudgeError as e:
        print(f"{red}{e}{endc}")
        return False
    record = {"behavior_id": pattern.behavior_id, "pattern_id": pattern.pattern_id, "prompt_id": prompt.prompt_id, "reasoning_enabled": reasoning_enabled,
              "judge_match": j.match, "judge_explanation": j.explanation, "judge_response": j.raw_response, **s}
    out.write(json.dumps(record) + "\n")
    out.flush()
    return True


def pick_targets(cfg: dict) -> list[tuple[Pattern, Prompt]]:
    behaviors = sorted({p.behavior_id for p in wc.patterns(subject_model=cfg["model"])}) if cfg["behaviors"] == "all" else cfg["behaviors"]
    targets = []
    for behavior_id in behaviors:
        pats = [p for p in wc.patterns(behavior_id=behavior_id, subject_model=cfg["model"]) if p.openrouter_replication and p.openrouter_replication.rate is not None]
        rank_by = cfg.get("rank_by", "elo")
        score = {p.pattern_id: s for p in pats if (s := p.elo.mean if rank_by == "elo" else p.elo.unexpectedness.elo if rank_by == "unexpectedness" else p.openrouter_replication.rate) is not None}
        pairs = []
        for pattern in pats:
            ranked = sorted(wc.prompts(pattern), key=lambda pr: pr.match_summary.n_matched / pr.match_summary.n_samples, reverse=True)
            pairs.extend((rank, pattern, pr) for rank, pr in enumerate(ranked))
        pairs.sort(key=lambda t: (t[0], -score[t[1].pattern_id]))  # best prompt of each pattern first, patterns by mean Elo (default), unexpectedness Elo (rank_by="unexpectedness") or OR rate (rank_by="rate")
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
    cost = sum(r["prompt_tokens"] * cfg["price_in"] + r["completion_tokens"] * cfg["price_out"] for r in records) / 1e6
    print(f"  {gray}subject-model cost: ${cost:.2f} over {len(records)} samples (judge cost not counted){endc}")


async def main(name: str) -> None:
    cfg = RUNS[name]
    config_path = os.path.join(run_dir(name), "config.json")
    if os.path.exists(config_path):
        saved = json.load(open(config_path))
        changed = [k for k in cfg if k not in ("n_off", "n_on", "provider") and saved.get(k) != cfg[k]]
        if changed:
            raise SystemExit(f"{red}run {name} exists with different config for {changed}; pick a new run name{endc}")

    records = load_records(name)
    done = Counter((r["prompt_id"], r["reasoning_enabled"]) for r in records)
    targets = load_targets(name, cfg)
    behavior_ids = sorted({pattern.behavior_id for pattern, _ in targets})
    print(f"  {gray}{len(targets)} target prompts over {len(behavior_ids)} behaviors; {len(records)} records already in {run_dir(name)}{endc}")

    deficit = {tag: sum(max(n - done[(prompt.prompt_id, on)], 0) for _, prompt in targets) for tag, on, n in (("off", False, cfg["n_off"]), ("on", True, cfg["n_on"]))}
    print(f"{purple}=== config ==={endc}")
    for k, v in cfg.items():
        print(f"  {cyan}{k:10s}{endc} {v}")
    print(f"  {cyan}{'to sample':10s}{endc} off={deficit['off']} on={deficit['on']}")
    if input(f"{yellow}proceed? [y/n] {endc}").strip().lower() != "y":
        raise SystemExit(f"{red}aborted{endc}")
    os.makedirs(run_dir(name), exist_ok=True)
    json.dump(cfg, open(config_path, "w"), indent=2)
    json.dump([{"behavior_id": pa.behavior_id, "pattern_id": pa.pattern_id, "prompt_id": pr.prompt_id} for pa, pr in targets], open(os.path.join(run_dir(name), "targets.json"), "w"), indent=2)

    if deficit["off"] or deficit["on"]:
        print(f"{purple}=== reasoning probe ==={endc}")
        probe = await sample_once(cfg, [Message(role="user", content="What is 17 * 23?")], reasoning_enabled=True)
        if probe is None or (not probe["reasoning_tokens"] and probe["reasoning"] is None):
            raise SystemExit(f"{red}provider does not appear to support reasoning for {cfg['model']}{endc}")
        print(f"  provider={cyan}{probe['provider']}{endc} served_model={probe['served_model']} reasoning_tokens={probe['reasoning_tokens']} trace_returned={probe['reasoning'] is not None}")

    judges = {b: wc.RubricJudge.for_behavior(wc.behavior(b)) for b in behavior_ids}
    with open(os.path.join(run_dir(name), "records.jsonl"), "a") as out:
        for reasoning_enabled, n_samples in ((False, cfg["n_off"]), (True, cfg["n_on"])):
            tag = "on" if reasoning_enabled else "off"
            flat = [(pattern, prompt) for pattern, prompt in targets for _ in range(max(n_samples - done[(prompt.prompt_id, reasoning_enabled)], 0))]
            if not flat:
                continue
            print(f"{purple}=== reasoning {tag.upper()}: {len(flat)} samples to reach {n_samples}/prompt ==={endc}")
            ok = await wc.run_bounded([sample_and_judge(cfg, pattern, prompt, judges[pattern.behavior_id], reasoning_enabled, out) for pattern, prompt in flat], CONCURRENCY, f"r={tag}")
            if not all(ok):
                print(f"{yellow}{len(ok) - sum(ok)} samples dropped (sample or judge failure) -- rerun to fill in{endc}")
            records = load_records(name)
            if not reasoning_enabled and not any(r["judge_match"] for r in records if not r["reasoning_enabled"]):
                raise SystemExit(f"{red}control replay elicited nothing on any behavior -- replay/judge/provider is broken, fix before spending on reasoning-on{endc}")

    report(name, cfg, records)


if __name__ == "__main__":
    anyio.run(main, sys.argv[1])
