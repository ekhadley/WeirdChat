"""Local-model interp helpers: TransformerBridge sampling, j-lens / template-lens loading and scoring,
record -> chat-template conversion for the replay runs under results/, and the token-level CoT resampling pipeline."""

# pyright: basic

import asyncio
import glob
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import matplotlib.pyplot as plt
import torch as t
from torch import Tensor

import IPython
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI
from safetensors import safe_open
from tabulate import tabulate
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer

import weirdchat as wc
from weirdchat.judge import JudgeError
from weirdchat.types import Message

# IPYTHON = IPython.get_ipython()
# if IPYTHON is not None:
#     IPYTHON.run_line_magic("load_ext", "autoreload")
#     IPYTHON.run_line_magic("autoreload", "2")

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

LENS_REPO = "camilablank/workspace-lenses"


def tec(): t.cuda.empty_cache()

# ============================= replay records ============================= #

def all_runs() -> list[str]:
    return sorted(p.removeprefix("results/").removesuffix("/records.jsonl") for p in glob.glob("results/*/*/records.jsonl"))


def load_records(run: str) -> list[dict]:
    path = f"results/{run}/records.jsonl"
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def record_messages(record: dict) -> list[Message]:
    """A record's prompt turns: stored inline (variants.py) or looked up in the dataset by prompt_id (run.py)."""
    if "messages" in record:
        return [Message(role=m["role"], content=m["content"]) for m in record["messages"]]
    return list(next(p for p in wc.prompts(record["pattern_id"]) if p.prompt_id == record["prompt_id"]).messages)


def record_to_conv(record: dict) -> list[dict]:
    conv = [{"role": m.role, "content": m.content} for m in record_messages(record)]
    assistant = {"role": "assistant", "content": record["response"]}
    if record["reasoning"]:
        assistant["reasoning_content"] = record["reasoning"]
    return conv + [assistant]

# ============================= OpenRouter sampling (run.py, variants.py) ============================= #

load_dotenv()
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=300.0, max_retries=3)
TEMPERATURE = 1.0
MAX_TOKENS_OFF = 1024  # the original pipeline's response cap
MAX_TOKENS_ON = 8192  # default; a run can override with max_tokens_on. Reasoning bills as completion tokens and the trace needs headroom beyond the visible 1024


async def sample_once(cfg: dict, messages: list[Message], reasoning_enabled: bool) -> dict | None:
    """One chat completion for cfg's model (pinned to cfg["provider"] if set); None after 3 failed attempts."""
    extra: dict = {"reasoning": {"enabled": reasoning_enabled}}
    if cfg["provider"]:
        extra["provider"] = {"order": [cfg["provider"]], "allow_fallbacks": False}
    last = "unknown error"
    for attempt in range(3):
        if attempt:
            await asyncio.sleep(2**attempt)
        try:
            reply = await client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=TEMPERATURE,
                max_tokens=cfg.get("max_tokens_on", MAX_TOKENS_ON) if reasoning_enabled else MAX_TOKENS_OFF,
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


async def sample_and_judge(cfg: dict, messages: list[Message], judge: wc.RubricJudge, reasoning_enabled: bool, meta: dict, out) -> bool:
    """Samples one response to messages, judges it, and appends {**meta, condition, verdict, sample} as one record line to out."""
    s = await sample_once(cfg, messages, reasoning_enabled)
    if s is None:
        return False
    try:
        j = await judge.judge(messages + [Message(role="assistant", content=s["response"])])
    except JudgeError as e:
        print(f"{red}{e}{endc}")
        return False
    out.write(json.dumps({**meta, "reasoning_enabled": reasoning_enabled, "judge_match": j.match, "judge_explanation": j.explanation, "judge_response": j.raw_response, **s}) + "\n")
    out.flush()
    return True


async def probe_reasoning(cfg: dict) -> None:
    """One reasoning-on call; exits if the provider returns neither a trace nor reasoning tokens for cfg's model."""
    print(f"{purple}=== reasoning probe ==={endc}")
    probe = await sample_once(cfg, [Message(role="user", content="What is 17 * 23?")], reasoning_enabled=True)
    if probe is None or (not probe["reasoning_tokens"] and probe["reasoning"] is None):
        raise SystemExit(f"{red}provider does not appear to support reasoning for {cfg['model']}{endc}")
    print(f"  provider={cyan}{probe['provider']}{endc} served_model={probe['served_model']} reasoning_tokens={probe['reasoning_tokens']} trace_returned={probe['reasoning'] is not None}")

# ============================= local sampling ============================= #

def stream_toks(model: TransformerBridge, inputs: Tensor, new_toks: int = 512):
    toks = inputs
    past = None
    for _ in range(new_toks):
        logits, past = model(toks, return_type="logits_and_cache", past_key_values=past, use_cache=True)
        probs = t.softmax(logits[0, -1], dim=-1)
        toks = t.multinomial(probs, num_samples=1).unsqueeze(0)
        if toks.item() == model.tokenizer.eos_token_id:
            break
        yield toks.item()

# ============================= lenses ============================= #

def load_jlens(path: str, device: str = "cpu") -> dict:
    return t.load(hf_hub_download(repo_id=LENS_REPO, filename=path), map_location=device, weights_only=False)


def load_tlens(path: str, device: str = "cpu") -> dict:
    local_path = hf_hub_download(repo_id=LENS_REPO, filename=path)
    words_path = hf_hub_download(repo_id=LENS_REPO, filename=path.replace("templates", "template_words").replace(".safetensors", ".txt"))
    with safe_open(local_path, framework="pt", device=device) as f:
        tlens = {"meta": f.metadata(), "templates": f.get_tensor("templates"), "word_ids": f.get_tensor("word_ids")}
    tlens["words"] = [line.split("\t", 1)[1] for line in open(words_path).read().splitlines()]
    return tlens


def jlens_transport(acts: Tensor, lens: dict, layer: int) -> Tensor:
    return lens["J"][layer].to(t.bfloat16) @ acts


def get_lens_logits(h: Tensor, layer: int, model: TransformerBridge, lens: dict) -> Tensor:
    return model.unembed(model.ln_final(jlens_transport(h, lens, layer)))


def get_template_idx(template: str, lens: dict) -> int:
    return lens["words"].index(template)


def get_template_vec(template: str, layer: int, lens: dict) -> Tensor:
    return lens["templates"][layer, get_template_idx(template, lens)]


def print_templates(tlens: dict, contains: str | None = None):
    for i, word in enumerate(tlens["words"]):
        if contains is None or contains.lower() in word.lower():
            print(f"{i}\t{word!r}")


def get_tlens_scores(h: Tensor, layer: int, tlens: dict) -> Tensor:
    return t.cosine_similarity(tlens["templates"][layer].to(h.device), h, dim=-1)

# ============================= printing ============================= #

def to_str_toks(inp: str | Tensor, tokenizer) -> list[str]:
    return [tokenizer.decode(tok) for tok in (tokenizer.encode(inp) if isinstance(inp, str) else inp.squeeze())]


def underline_stoks(toks: Tensor, tokenizer) -> str:
    stoks = to_str_toks(toks, tokenizer)
    return "".join(f"{underline if i % 2 else endc}{s}" for i, s in enumerate(stoks)) + endc


def print_titled_table(table_str: str, title: str | None = None):
    if title is None:
        print(table_str)
        return
    lines = table_str.splitlines()
    inner = len(lines[0]) - 2
    print(f"╭{'─' * inner}╮")
    print(f"│{bold}{title.center(inner)}{endc}│")
    print(f"├{'─' * inner}┤")
    print("\n".join(lines[1:]))


def top_toks_table(logits: Tensor, tokenizer, k: int = 10, show_negative: bool = False, title: str | None = None):
    logits = logits.flatten().float()
    probs = logits.softmax(-1)
    top = logits.topk(k)
    headers = ["Tok", "Logit", "Prob"]
    cols = [[repr(tokenizer.decode([i])) for i in top.indices.tolist()], top.values.tolist(), probs[top.indices].tolist()]
    if show_negative:
        bot = logits.topk(k, largest=False)
        headers = [f"Top {h}" for h in headers] + [f"Bot {h}" for h in headers]
        cols += [[repr(tokenizer.decode([i])) for i in bot.indices.tolist()], bot.values.tolist(), probs[bot.indices].tolist()]
    data = [(i, *(col[i] for col in cols)) for i in range(k)]
    print_titled_table(tabulate(data, headers=["Idx"] + headers, tablefmt="rounded_outline"), title)


def top_templates_table(scores: Tensor, words: list[str], k: int = 10, show_negative: bool = False, title: str | None = None):
    scores = scores.flatten().float()
    top = scores.topk(k)
    headers = ["Template", "Cos"]
    cols = [[repr(words[i]) for i in top.indices.tolist()], top.values.tolist()]
    if show_negative:
        bot = scores.topk(k, largest=False)
        headers = [f"Top {h}" for h in headers] + [f"Bot {h}" for h in headers]
        cols += [[repr(words[i]) for i in bot.indices.tolist()], bot.values.tolist()]
    data = [(i, *(col[i] for col in cols)) for i in range(k)]
    print_titled_table(tabulate(data, headers=["Idx"] + headers, tablefmt="rounded_outline"), title)


# ============================= CoT resampling ============================= #
# Cut a judged reasoning-on record's trace at token positions, resample S continuations per prefix through OpenRouter's raw
# /completions endpoint (the prompt is rendered by cfg.render_prompt so the think block stays open; chat-message prefill is
# closed by every provider's template), judge each visible response with the behavior's rubric judge, and report
# o_t = P(match | prefix_t). Rollouts land in results/<model>/resample/<run>_<idx>/rollouts.jsonl, one per line; reruns fill
# the per-position deficit, and scoring covers every position that has rollouts on disk plus the configured grid.

@dataclass
class ResampleConfig:
    run: str                                              # "<model dir>/<run name>" under results/
    idx: int                                              # record index in that run's records.jsonl
    model: str                                            # OpenRouter model id
    provider: str                                         # OpenRouter provider slug; must pass raw prompts through untouched (see CLAUDE.md)
    tokenizer: str                                        # HF tokenizer id of the served checkpoint
    render_prompt: Callable[[Any, list[Message]], str]    # (tokenizer, prompt messages) -> prompt string ending inside an open think block
    S: int = 20                                           # continuations per position
    stride: int = 1                                       # resample every stride-th token position; 0 and the final position are always included
    max_tokens: int = 8192
    temperature: float = 1.0
    concurrency: int = 32

    @property
    def out_dir(self) -> str:
        return os.path.join("results", self.run.split("/")[0], "resample", f"{self.run.split('/')[1]}_{self.idx}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


async def _rollout(client: httpx.AsyncClient, sem: asyncio.Semaphore, cfg: ResampleConfig, headers: dict, prefix: str, t_pos: int, i: int, prompt_msgs: list[Message], judge: wc.RubricJudge, out) -> bool:
    payload = {"model": cfg.model, "prompt": prefix, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature, "transforms": [], "provider": {"only": [cfg.provider], "allow_fallbacks": False}}
    async with sem:
        for attempt in range(6):  # providers rate-limit bursts with 429; back off and retry
            await asyncio.sleep(2**attempt - 1)
            body = None
            try:
                r = await client.post("https://openrouter.ai/api/v1/completions", json=payload, headers=headers, timeout=300)
                body = r.json()
                choice = body["choices"][0]
                break
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:100]} {str(body)[:200]}"
        else:
            print(f"{red}t={t_pos} i={i} request failed after 6 attempts: {last}{endc}")
            return False
    reasoning_cont, response = choice.get("reasoning") or "", choice.get("text") or ""
    rec = {"t": t_pos, "i": i, "reasoning_cont": reasoning_cont, "response": response, "finish_reason": choice.get("finish_reason"), "provider": body.get("provider"),
           "prompt_tokens": body["usage"]["prompt_tokens"], "completion_tokens": body["usage"]["completion_tokens"]}
    if not response.strip():
        rec |= {"category": "other", "judge_match": None, "judge_explanation": None}
    else:
        try:
            j = await judge.judge(prompt_msgs + [Message(role="assistant", content=response)])
        except JudgeError as e:
            print(f"{red}t={t_pos} i={i} judge failed: {e}{endc}")
            return False
        rec |= {"category": "match" if j.match else "nomatch", "judge_match": j.match, "judge_explanation": j.explanation}
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return True


async def _resample(cfg: ResampleConfig):
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    record = load_records(cfg.run)[cfg.idx]
    assert record["reasoning_enabled"] and record["reasoning"], "need a reasoning-on record with a trace"
    prompt_msgs = record_messages(record)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    prompt_str = cfg.render_prompt(tokenizer, prompt_msgs)
    ids = tokenizer(record["reasoning"], add_special_tokens=False)["input_ids"]
    n_prompt = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
    print(f"{purple}=== {cfg.run}[{cfg.idx}] {record['behavior_id']} judge_match={record['judge_match']} ==={endc}")
    n_response = len(tokenizer(record["response"], add_special_tokens=False)["input_ids"])
    print(f"  {gray}prompt tokens local={n_prompt} record={record['prompt_tokens']}  reasoning+response tokens local={len(ids) + n_response} record completion={record['completion_tokens']}{endc}")
    # providers' reasoning_tokens is unreliable, but completion_tokens = reasoning + response + </think> (+ eos) holds everywhere; a larger gap means a truncated trace
    assert n_prompt == record["prompt_tokens"] and record["completion_tokens"] - len(ids) - n_response in (1, 2), "local rendering disagrees with the record's token counts"
    grid = sorted(set(range(0, len(ids) + 1, cfg.stride)) | {len(ids)})
    # a cut inside a multi-byte character (byte-level BPE) decodes to a replacement char and does not retokenize to the same ids; skip those positions
    invalid = [t_pos for t_pos in grid if tokenizer(tokenizer.decode(ids[:t_pos]), add_special_tokens=False)["input_ids"] != ids[:t_pos]]
    if invalid:
        print(f"  {yellow}skipping {len(invalid)} positions whose text prefix does not retokenize identically: {invalid[:20]}{endc}")
    grid = [t_pos for t_pos in grid if t_pos not in invalid]
    prefixes = {t_pos: prompt_str + tokenizer.decode(ids[:t_pos]) for t_pos in grid}

    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, "rollouts.jsonl")
    done = Counter(r["t"] for r in (json.loads(l) for l in open(path))) if os.path.exists(path) else Counter()
    todo = [(t_pos, done[t_pos] + k) for t_pos in grid for k in range(cfg.S - done[t_pos])]
    print(f"  {gray}{sum(done.values())} rollouts on disk, {len(todo)} to sample across {len(grid)} positions (stride {cfg.stride}) at S={cfg.S} via {cfg.provider}{endc}")
    if todo:
        judge = wc.RubricJudge.for_behavior(wc.behavior(record["behavior_id"]))
        sem = asyncio.Semaphore(cfg.concurrency)
        out = open(path, "a")
        async with httpx.AsyncClient() as client:
            ok = await wc.run_bounded([_rollout(client, sem, cfg, headers, prefixes[t_pos], t_pos, i, prompt_msgs, judge, out) for t_pos, i in todo], cfg.concurrency, "resample")
        out.close()
        print(f"  {green}{sum(ok)}/{len(todo)} rollouts written{endc}" if all(ok) else f"  {yellow}{len(ok) - sum(ok)} failures, rerun to fill{endc}")

    rollouts = [json.loads(l) for l in open(path)]
    assert all(r["prompt_tokens"] == n_prompt + r["t"] for r in rollouts), "provider tokenized a prefix into a different number of tokens than the cut position"
    scores = []
    for t_pos in sorted(set(grid) | {r["t"] for r in rollouts}):
        cats = Counter(r["category"] for r in rollouts if r["t"] == t_pos)
        n_judged = cats["match"] + cats["nomatch"]
        scores.append({"t": t_pos, "token": tokenizer.decode(ids[t_pos - 1]) if t_pos else "", "n": sum(cats.values()), "match": cats["match"], "nomatch": cats["nomatch"], "other": cats["other"],
                       "p_match": cats["match"] / n_judged if n_judged else float("nan"), "ci": wilson(cats["match"], n_judged)})
    json.dump({"run": cfg.run, "idx": cfg.idx, "model": cfg.model, "provider": cfg.provider, "S": cfg.S, "stride": cfg.stride, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
               "behavior_id": record["behavior_id"], "tokens": [tokenizer.decode([i]) for i in ids], "scores": scores}, open(os.path.join(cfg.out_dir, "scores.json"), "w"), indent=1)
    print(f"{purple}=== o_t = P(match | prefix_t) ==={endc}")
    for s in scores:
        print(f"  {cyan}t={s['t']:4d} p={s['p_match']:.2f} [{s['ci'][0]:.2f},{s['ci'][1]:.2f}] n={s['n']:2d} other={s['other']}{endc} {gray}{s['token']!r}{endc}")

    fig, ax = plt.subplots(figsize=(max(12, len(scores) / 6), 5))
    ts = [s["t"] for s in scores]
    ax.fill_between(ts, [s["ci"][0] for s in scores], [s["ci"][1] for s in scores], alpha=0.25)
    ax.plot(ts, [s["p_match"] for s in scores], marker="o", ms=3)
    if len(scores) <= 400:
        ax.set_xticks(ts)
        ax.set_xticklabels([s["token"] for s in scores], rotation=90, fontsize=6, family="monospace")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(match | prefix_t)")
    ax.set_title(f"{cfg.run}[{cfg.idx}] {record['behavior_id']}  S={cfg.S} stride={cfg.stride} via {cfg.provider}")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig_path = f"figures/resample_{cfg.run.split('/')[1]}_{cfg.idx}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  {green}figure saved to {fig_path}{endc}")


def resample(cfg: ResampleConfig):
    asyncio.run(_resample(cfg))
