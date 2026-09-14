"""Activation-oracle (AO) helpers and viewer for Qwen3-8B. An AO is a LoRA on the base model that answers questions about
residual-stream activations: the oracle prompt holds one placeholder token per activation, the activations are added
norm-matched into the residual stream at decoder block INJECT_LAYER's output at those positions, and the LoRA generates the
answer. Activations come from the same model with the adapter disabled, at the output of decoder block `layer`
(TransformerLens blocks.{layer + 1}.hook_resid_pre). Several layers go in as one placeholder block per layer, activations
concatenated layer-major. Serve the viewer from a cell (see local_ao.py): serve(model, tokenizer, cfg); the tag box takes
a record tag (model/run/idx) from the rollout viewer, the strip selects a token or a span, ask answers one query about the
span, sweep asks the same question at every position of the span."""

# pyright: basic

import os
import threading
from dataclasses import dataclass, field, replace

import torch as t
from torch import Tensor
from tqdm import tqdm
from flask import Flask, request, send_file
from werkzeug.serving import make_server
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from mechtools import tec
from utils import load_records, record_to_conv

PORT = 7862
INJECT_LAYER = 1
PLACEHOLDER = " ?"


@dataclass
class AOConfig:
    model_id: str = "Qwen/Qwen3-8B"
    adapter: str = "ceselder/qwen3-8b-ao-v3-best"  # adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B reads layers [9, 18, 27]; ceselder/qwen3-8b-ao-v3-best-steering2p0 wants norm_scale=2.0
    layers: list[int] = field(default_factory=lambda: [21, 22, 23, 24, 25])
    coef: float = 1.0                # the injected vector's norm as a multiple of the residual's norm at that position
    norm_scale: float | None = None  # if set, the post-injection residual is rescaled to this multiple of the original norm
    n: int = 3                       # answers sampled per query
    temperature: float | None = None  # None: the checkpoint's generation_config; 0: greedy
    max_new_tokens: int = 100
    batch_size: int = 64             # oracle generations per generate call


def load_ao(cfg: AOConfig) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=t.bfloat16, device_map="cuda")
    return PeftModel.from_pretrained(base, cfg.adapter).eval(), tokenizer


def blocks(model: PeftModel):
    return model.base_model.model.model.layers


def _resid(out) -> Tensor:
    return out[0] if isinstance(out, tuple) else out


@t.no_grad()
def collect_acts(model: PeftModel, ids: Tensor, layers: list[int]) -> dict[int, Tensor]:
    """{layer: [seq, d_model]} residual stream after decoder block `layer` for ids [1, seq], adapter off."""
    acts: dict[int, Tensor] = {}
    handles = [blocks(model)[L].register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, _resid(o)[0].clone())) for L in layers]
    with model.disable_adapter():
        model(ids)
    for h in handles:
        h.remove()
    return acts


def span_vecs(acts: dict[int, Tensor], layers: list[int], start: int, end: int) -> Tensor:
    """[1, (end - start) * len(layers), d_model]: one query holding the span's activations at every layer, layer-major."""
    return t.cat([acts[L][start:end] for L in layers])[None]


def sweep_vecs(acts: dict[int, Tensor], layers: list[int], start: int, end: int) -> Tensor:
    """[end - start, len(layers), d_model]: one query per position, holding that position's activation at every layer."""
    return t.stack([acts[L][start:end] for L in layers], dim=1)


def oracle_prompt(tokenizer, layers: list[int], k: int, question: str) -> tuple[list[int], list[int]]:
    """Token ids of the chat-templated oracle prompt with k placeholders per layer, and the placeholder positions (layer-major)."""
    prefix = "".join(f"Layer: {L}\n{PLACEHOLDER * k} \n" for L in layers)
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prefix + question}], tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False)
    ph = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    assert len(ph) == 1
    positions = [i for i, tok in enumerate(ids) if tok == ph[0]]
    assert len(positions) == k * len(layers), f"{len(positions)} placeholder tokens for k={k} x {len(layers)} layers; the question must not contain {PLACEHOLDER!r}"
    return ids, positions


def inject_hook(vecs: Tensor, positions: list[int], coef: float, norm_scale: float | None):
    """Forward hook on a decoder block: on the prefill, adds coef * ||resid|| * unit(vecs[b, j]) at positions[j] of row b; vecs [B, len(positions), d_model]."""
    unit = t.nn.functional.normalize(vecs.float(), dim=-1)

    def hook(module, inp, out):
        resid = _resid(out)
        if resid.shape[1] == 1:  # a decode step
            return out
        orig = resid[:, positions]
        norms = orig.norm(dim=-1, keepdim=True)
        new = orig + (unit * norms * coef).to(orig.dtype)
        if norm_scale is not None:
            new = new * (norm_scale * norms / new.norm(dim=-1, keepdim=True))
        resid[:, positions] = new
        return out

    return hook


@t.no_grad()
def ask(model: PeftModel, tokenizer, cfg: AOConfig, vecs: Tensor, question: str) -> list[list[str]]:
    """cfg.n sampled answers to `question` for each query in vecs [B, k * len(cfg.layers), d_model] (a row: k activations per layer, layer-major)."""
    ids, positions = oracle_prompt(tokenizer, cfg.layers, vecs.shape[1] // len(cfg.layers), question)
    sampling = {"do_sample": False} if cfg.temperature == 0 else {"do_sample": True} | ({} if cfg.temperature is None else {"temperature": cfg.temperature})
    rows = vecs.repeat_interleave(cfg.n, 0)
    answers = []
    for chunk in tqdm(rows.split(cfg.batch_size), desc="oracle", ascii=" >=", disable=rows.shape[0] <= cfg.batch_size):
        inp = t.tensor([ids] * chunk.shape[0], device=vecs.device)
        handle = blocks(model)[INJECT_LAYER].register_forward_hook(inject_hook(chunk, positions, cfg.coef, cfg.norm_scale))
        try:
            out = model.generate(inp, attention_mask=t.ones_like(inp), max_new_tokens=cfg.max_new_tokens, **sampling)
        finally:
            handle.remove()
        answers += tokenizer.batch_decode(out[:, len(ids):], skip_special_tokens=True)
    tec()
    return [answers[i * cfg.n:(i + 1) * cfg.n] for i in range(vecs.shape[0])]


# ============================= viewer ============================= #

app = Flask(__name__)
state: dict = {}
_records: dict[str, list[dict]] = {}


def records(run: str) -> list[dict]:
    if run not in _records:
        _records[run] = load_records(run)
    return _records[run]


@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ao.html"))


@app.get("/text/<model>/<name>/<int:i>")
def record_text(model: str, name: str, i: int):
    return {"text": state["tokenizer"].apply_chat_template(record_to_conv(records(f"{model}/{name}")[i]), tokenize=False)}


def req_cfg() -> AOConfig:
    return replace(state["cfg"], **request.json["cfg"])


def acts_for(layers: list[int]) -> dict[int, Tensor]:
    missing = [L for L in layers if L not in state["acts"]]
    if missing:
        state["acts"].update(collect_acts(state["model"], state["ids"], missing))
    return state["acts"]


@app.post("/run")
def run():
    tokenizer = state["tokenizer"]
    state["ids"] = tokenizer(request.json["text"], return_tensors="pt").input_ids.to(state["model"].device)
    state["acts"] = {}
    acts_for(req_cfg().layers)
    return {"toks": [tokenizer.decode(tok) for tok in state["ids"][0]]}


@app.post("/ask")
def ask_span():
    cfg, r = req_cfg(), request.json
    return {"answers": ask(state["model"], state["tokenizer"], cfg, span_vecs(acts_for(cfg.layers), cfg.layers, r["start"], r["end"]), r["question"])[0]}


@app.post("/sweep")
def sweep():
    cfg, r = req_cfg(), request.json
    return {"answers": ask(state["model"], state["tokenizer"], cfg, sweep_vecs(acts_for(cfg.layers), cfg.layers, r["start"], r["end"]), r["question"])}


def serve(model: PeftModel, tokenizer, cfg: AOConfig, port: int = PORT):
    if "server" in state:
        state["server"].shutdown()
    state.update(model=model, tokenizer=tokenizer, cfg=cfg)
    state["server"] = make_server("0.0.0.0", port, app)
    threading.Thread(target=state["server"].serve_forever, daemon=True).start()
    print(f"ao viewer running at http://localhost:{port}")


def stop():
    state.pop("server").shutdown()
