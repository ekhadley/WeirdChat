"""Snapshots the full-quota replay runs to the HF hub as two parquet configs: `records` (one judged sample per row, prompt text inlined) and `runs` (one row per run with its config and frozen targets). `uv run python export_hf.py` builds and prints both tables; `--push` also uploads them."""

import json
import sys

from datasets import Dataset
from dotenv import load_dotenv

import weirdchat as wc
from utils import load_records

load_dotenv()
REPO = "eekay/weirdchat-replays"
RUNS = ["deepseek-v4-flash/dv4f_full_elo", "deepseek-v4-flash/dv4f_unexp", "inkling/inkling_full_elo", "qwen3.6-27b/q36_27b_elo", "qwen3.6-27b/q36_27b_z", "qwen3-8b/q3_8b_dating"]


def prompt_text(pattern_id: str, prompt_id: str) -> str:
    (msg,) = next(p for p in wc.prompts(pattern_id) if p.prompt_id == prompt_id).messages  # every target is a single user turn
    return msg.content


records, runs = [], []
for run in RUNS:
    model, name = run.split("/")
    cfg = json.load(open(f"results/{run}/config.json"))
    targets = json.load(open(f"results/{run}/targets.json"))
    prompts = {t["prompt_id"]: prompt_text(t["pattern_id"], t["prompt_id"]) for t in targets}
    runs.append(dict(model=model, run=name, openrouter_model=cfg["model"], provider=cfg["provider"], behaviors=sorted({t["behavior_id"] for t in targets}), n_prompts=cfg["n_prompts"], rank_by=cfg.get("rank_by", "elo"), n_off=cfg["n_off"], n_on=cfg["n_on"], max_tokens_on=cfg.get("max_tokens_on", 8192), targets=targets))
    for idx, r in enumerate(load_records(run)):
        records.append(dict(model=model, run=name, idx=idx, behavior_id=r["behavior_id"], pattern_id=r["pattern_id"], prompt_id=r["prompt_id"], prompt=prompts[r["prompt_id"]], reasoning_enabled=r["reasoning_enabled"], reasoning=r["reasoning"], completion=r["response"], judge_match=r["judge_match"], judge_explanation=r["judge_explanation"], provider=r["provider"], prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"], reasoning_tokens=r["reasoning_tokens"]))

tables = {"records": Dataset.from_list(records), "runs": Dataset.from_list(runs)}
for config_name, ds in tables.items():
    print(config_name, ds)
    if "--push" in sys.argv:
        ds.push_to_hub(REPO, config_name=config_name, private=True)
