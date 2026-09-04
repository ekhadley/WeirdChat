#!./.venv/bin/python
"""Token-level CoT resampling of an Inkling replay record; the pipeline is utils.resample.

Usage: uv run python inkling_resample.py   (reruns fill the per-position deficit)

Together is the only OpenRouter provider that passes a raw Inkling prompt through untouched. It returns the reasoning and the
response merged with the special tokens dropped, so each rollout is two calls: the trace up to <|end_message|>, then the response.
"""
from dotenv import load_dotenv

from utils import ResampleConfig, resample

load_dotenv()

cfg = ResampleConfig(
    run="inkling/inkling_full_elo", idx=7979,
    model="thinkingmachines/inkling", provider="together", tokenizer="thinkingmachines/Inkling",
    render_prompt=lambda tok, msgs: tok.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in msgs],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=0.9,  # the template default; the replay's effort is not recoverable from the records, and 0.7 vs 0.9 trace lengths are indistinguishable
    ) + "<|content_thinking|>",  # the generation prompt ends at <|message_model|> and the model picks thinking or text itself
    S=30, stride=1, max_tokens=32768,
    stop="<|end_message|>", response_open="<|end_message|><|message_model|><|content_text|>", prompt_extra=2, completion_extra=(5, 6, 7),
)

if __name__ == "__main__":
    resample(cfg)
