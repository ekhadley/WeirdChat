#!./.venv/bin/python
"""Token-level CoT resampling of a DeepSeek V4 Flash replay record; the pipeline is utils.resample.

Usage: uv run python deepseek_resample.py   (reruns fill the per-position deficit)
"""
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from utils import ResampleConfig, resample

load_dotenv()
sys.path.insert(0, os.path.dirname(hf_hub_download("deepseek-ai/DeepSeek-V4-Flash", "encoding/encoding_dsv4.py")))
from encoding_dsv4 import encode_messages  # DeepSeek ships its prompt encoder as a script instead of a Jinja template

cfg = ResampleConfig(
    run="deepseek-v4-flash/dv4f_smoke", idx=128,
    # run="deepseek-v4-flash/dv4f_full_elo", idx=26573,
    model="deepseek/deepseek-v4-flash", provider="deepinfra", tokenizer="deepseek-ai/DeepSeek-V4-Flash",
    render_prompt=lambda tok, msgs: encode_messages([{"role": m.role, "content": m.content} for m in msgs], "thinking"),  # ends with <｜Assistant｜><think>
    S=50, stride=1, tag="_s50",
)

if __name__ == "__main__":
    resample(cfg)
