"""Local-model interp helpers: TransformerBridge sampling, j-lens / template-lens loading and scoring,
and record -> chat-template conversion for the replay runs under results/."""

# pyright: basic

import glob
import json

import IPython
import torch as t
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tabulate import tabulate
from torch import Tensor
from transformer_lens.model_bridge import TransformerBridge

import weirdchat as wc

IPYTHON = IPython.get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic("load_ext", "autoreload")
    IPYTHON.run_line_magic("autoreload", "2")

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
    return [json.loads(line) for line in open(f"results/{run}/records.jsonl")]


def record_to_conv(record: dict) -> list[dict]:
    prompt = next(p for p in wc.prompts(record["pattern_id"]) if p.prompt_id == record["prompt_id"])
    conv = [{"role": m.role, "content": m.content} for m in prompt.messages]
    assistant = {"role": "assistant", "content": record["response"]}
    if record["reasoning"]:
        assistant["reasoning_content"] = record["reasoning"]
    return conv + [assistant]

# ============================= sampling ============================= #

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
