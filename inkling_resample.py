#!./.venv/bin/python
"""Token-level resampling of an Inkling replay record; the pipeline is utils.resample.

Usage: uv run python inkling_resample.py <cot|response>   (reruns fill the per-position deficit)

cot cuts a reasoning-on record's trace; response cuts a reasoning-off record's visible output (effort 0, the text block opened directly).
Together is the only OpenRouter provider that passes a raw Inkling prompt through untouched. It returns the reasoning and the
response merged with the special tokens dropped, so each cot rollout is two calls: the trace up to <|end_message|>, then the response.
"""
import sys

from dotenv import load_dotenv

from utils import ResampleConfig, resample

load_dotenv()


def render(effort: float, block: str):
    """The HF template's generation prompt ends at <|message_model|>, where the model picks thinking or text itself; block opens it for it."""
    return lambda tok, msgs: tok.apply_chat_template([{"role": m.role, "content": m.content} for m in msgs], tokenize=False, add_generation_prompt=True, reasoning_effort=effort) + block


CONFIGS = {
    "cot": ResampleConfig(
        run="inkling/inkling_full_elo", idx=7979,
        model="thinkingmachines/inkling", provider="together", tokenizer="thinkingmachines/Inkling",
        render_prompt=render(0.9, "<|content_thinking|>"),  # 0.9 is the template default; the replay's effort is not recoverable from the records, and 0.7 vs 0.9 trace lengths are indistinguishable
        S=30, stride=1, max_tokens=32768,
        stop="<|end_message|>", response_open="<|end_message|><|message_model|><|content_text|>", prompt_extra=2, completion_extra=(5, 6, 7),
    ),
    "response": ResampleConfig(
        run="inkling/inkling_full_elo", idx=2758, cut="response",  # a Together-served reasoning-off match on the customer support prompt
        model="thinkingmachines/inkling", provider="together", tokenizer="thinkingmachines/Inkling",
        render_prompt=render(0.0, "<|content_text|>"),  # reasoning off renders as "Thinking effort level: 0" (record prompt_tokens match this exactly)
        S=30, stride=1, max_tokens=1024,
        stop="<|end_message|>", prompt_extra=1, completion_extra=(3,),  # completion = response + <|content_text|> <|end_message|> <|content_model_end_sampling|>
    ),
}

if __name__ == "__main__":
    resample(CONFIGS[sys.argv[1]])
