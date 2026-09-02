#!./.venv/bin/python
"""Token-level CoT resampling for one judged reasoning-on record.

Cuts the record's reasoning after every token, resamples S continuations from each prefix through the raw
/completions endpoint (prompt rendered with DeepSeek's shipped encoder, so the think block stays open), judges each
visible response with the behavior's rubric judge, and reports o_t = P(match | prefix_t).
Rollouts land in results/<model>/resample/<run>_<idx>/rollouts.jsonl, one per line; reruns fill the deficit per position.
"""
import asyncio
import json
import math
import os
import sys
from collections import Counter

import httpx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import weirdchat as wc
from weirdchat.judge import JudgeError
from weirdchat.types import Message
from utils import *

load_dotenv()
sys.path.insert(0, os.path.dirname(hf_hub_download("deepseek-ai/DeepSeek-V4-Flash", "encoding/encoding_dsv4.py")))
from encoding_dsv4 import encode_messages

RUN, IDX = "deepseek-v4-flash/dv4f_smoke", 128
MODEL, PROVIDER, TOKENIZER = "deepseek/deepseek-v4-flash", "deepinfra", "deepseek-ai/DeepSeek-V4-Flash"
S, MAX_TOKENS, TEMPERATURE, CONCURRENCY = 20, 8192, 1.0, 32
OUT_DIR = os.path.join("results", RUN.split("/")[0], "resample", f"{RUN.split('/')[1]}_{IDX}")
ROLLOUTS = os.path.join(OUT_DIR, "rollouts.jsonl")
HEADERS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


async def rollout(client: httpx.AsyncClient, sem: asyncio.Semaphore, prefix: str, t: int, i: int, prompt_msgs: list[Message], judge: wc.RubricJudge, out) -> bool:
    payload = {"model": MODEL, "prompt": prefix, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "transforms": [], "provider": {"only": [PROVIDER], "allow_fallbacks": False}}
    async with sem:
        for attempt in range(6):  # providers rate-limit bursts with 429; back off and retry
            await asyncio.sleep(2**attempt - 1)
            body = None
            try:
                r = await client.post("https://openrouter.ai/api/v1/completions", json=payload, headers=HEADERS, timeout=300)
                body = r.json()
                choice = body["choices"][0]
                break
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:100]} {str(body)[:200]}"
        else:
            print(f"{red}t={t} i={i} request failed after 6 attempts: {last}{endc}")
            return False
    reasoning_cont, response = choice.get("reasoning") or "", choice.get("text") or ""
    rec = {"t": t, "i": i, "reasoning_cont": reasoning_cont, "response": response, "finish_reason": choice.get("finish_reason"), "provider": body.get("provider"),
           "prompt_tokens": body["usage"]["prompt_tokens"], "completion_tokens": body["usage"]["completion_tokens"]}
    if not response.strip():
        rec |= {"category": "other", "judge_match": None, "judge_explanation": None}
    else:
        try:
            j = await judge.judge(prompt_msgs + [Message(role="assistant", content=response)])
        except JudgeError as e:
            print(f"{red}t={t} i={i} judge failed: {e}{endc}")
            return False
        rec |= {"category": "match" if j.match else "nomatch", "judge_match": j.match, "judge_explanation": j.explanation}
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return True


async def main():
    record = load_records(RUN)[IDX]
    assert record["reasoning_enabled"] and record["reasoning"], "need a reasoning-on record with a trace"
    prompt = next(p for p in wc.prompts(record["pattern_id"]) if p.prompt_id == record["prompt_id"])
    prompt_msgs = list(prompt.messages)
    prompt_str = encode_messages([{"role": m.role, "content": m.content} for m in prompt_msgs], "thinking")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    ids = tokenizer(record["reasoning"], add_special_tokens=False)["input_ids"]
    n_prompt = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
    print(f"{purple}=== {RUN}[{IDX}] {record['behavior_id']} judge_match={record['judge_match']} ==={endc}")
    print(f"  {gray}prompt tokens local={n_prompt} record={record['prompt_tokens']}  reasoning tokens local={len(ids)} record={record['reasoning_tokens']}{endc}")
    assert n_prompt == record["prompt_tokens"] and len(ids) == record["reasoning_tokens"], "local rendering disagrees with the record's token counts"
    prefixes = {t: prompt_str + tokenizer.decode(ids[:t]) for t in range(len(ids) + 1)}
    for t in prefixes:
        assert tokenizer(tokenizer.decode(ids[:t]), add_special_tokens=False)["input_ids"] == ids[:t], f"prefix at t={t} does not retokenize identically"

    os.makedirs(OUT_DIR, exist_ok=True)
    done = Counter(r["t"] for r in (json.loads(l) for l in open(ROLLOUTS))) if os.path.exists(ROLLOUTS) else Counter()
    todo = [(t, done[t] + k) for t in prefixes for k in range(S - done[t])]
    print(f"  {gray}{sum(done.values())} rollouts on disk, {len(todo)} to sample across {len(prefixes)} positions at S={S} via {PROVIDER}{endc}")
    if todo:
        judge = wc.RubricJudge.for_behavior(wc.behavior(record["behavior_id"]))
        sem = asyncio.Semaphore(CONCURRENCY)
        out = open(ROLLOUTS, "a")
        async with httpx.AsyncClient() as client:
            ok = await wc.run_bounded([rollout(client, sem, prefixes[t], t, i, prompt_msgs, judge, out) for t, i in todo], CONCURRENCY, "resample")
        out.close()
        print(f"  {green}{sum(ok)}/{len(todo)} rollouts written{endc}" if all(ok) else f"  {yellow}{len(ok) - sum(ok)} failures, rerun to fill{endc}")

    rollouts = [json.loads(l) for l in open(ROLLOUTS)]
    scores = []
    for t in prefixes:
        cats = Counter(r["category"] for r in rollouts if r["t"] == t)
        n_judged = cats["match"] + cats["nomatch"]
        p = cats["match"] / n_judged if n_judged else float("nan")
        scores.append({"t": t, "token": tokenizer.decode(ids[t - 1]) if t else "", "n": sum(cats.values()), "match": cats["match"], "nomatch": cats["nomatch"], "other": cats["other"], "p_match": p, "ci": wilson(cats["match"], n_judged)})
    json.dump({"run": RUN, "idx": IDX, "model": MODEL, "provider": PROVIDER, "S": S, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "behavior_id": record["behavior_id"],
               "tokens": [tokenizer.decode([i]) for i in ids], "scores": scores}, open(os.path.join(OUT_DIR, "scores.json"), "w"), indent=1)
    print(f"{purple}=== o_t = P(match | prefix_t) ==={endc}")
    for s in scores:
        lo, hi = s["ci"]
        print(f"  {cyan}t={s['t']:3d} p={s['p_match']:.2f} [{lo:.2f},{hi:.2f}] n={s['n']:2d} other={s['other']}{endc} {gray}{s['token']!r}{endc}")
    cost = sum(r["prompt_tokens"] for r in rollouts) * 0.09e-6 + sum(r["completion_tokens"] for r in rollouts) * 0.18e-6
    print(f"  {gray}subject-model cost: ${cost:.2f} over {len(rollouts)} rollouts (judge not counted){endc}")

    fig, ax = plt.subplots(figsize=(max(12, len(scores) / 6), 5))
    ts = [s["t"] for s in scores]
    ax.fill_between(ts, [s["ci"][0] for s in scores], [s["ci"][1] for s in scores], alpha=0.25)
    ax.plot(ts, [s["p_match"] for s in scores], marker="o", ms=3)
    ax.set_xticks(ts)
    ax.set_xticklabels([s["token"] for s in scores], rotation=90, fontsize=6, family="monospace")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(match | prefix_t)")
    ax.set_title(f"{RUN}[{IDX}] {record['behavior_id']}  S={S} via {PROVIDER}")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/resample_{RUN.split('/')[1]}_{IDX}.png", dpi=150)
    print(f"  {green}figure saved to figures/resample_{RUN.split('/')[1]}_{IDX}.png{endc}")


if __name__ == "__main__":
    asyncio.run(main())
