#!./.venv/bin/python
"""Token-level CoT resampling of a Qwen3.6-27B replay record; the pipeline is utils.resample.

Usage: uv run python resample_qwen.py   (reruns fill the per-position deficit)
"""
from dotenv import load_dotenv

from utils import ResampleConfig, resample

load_dotenv()

cfg = ResampleConfig(
    run="qwen3.6-27b/q36_27b_smoke", idx=768,
    model="qwen/qwen3.6-27b", provider="chutes", tokenizer="Qwen/Qwen3.6-27B",
    render_prompt=lambda tok, msgs: tok.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in msgs],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    ),  # ends with <|im_start|>assistant\n<think>\n
    S=20, stride=8,
)

if __name__ == "__main__":
    resample(cfg)
