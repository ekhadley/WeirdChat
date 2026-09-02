"""Lens viewer over replay records. Serve from a cell with the model and lenses loaded (see local_tl.py):
serve(model, tokenizer, jlens, tlens). Paste a record tag (model/run/idx, click-to-copy in the rollout viewer's
panel header) into the tag box and press enter; the record's conversation is chat-templated into the text box.
Run pushes the text through the model and shows, per token
and layer, the j-lens top tokens and template-lens top templates, with pinnable rows tracked across positions."""

# pyright: basic

import os
import threading

import torch as t
from flask import Flask, request, send_file
from werkzeug.serving import make_server

from utils import load_records, record_to_conv

PORT = 7860

app = Flask(__name__)
state: dict = {}
_records: dict[str, list[dict]] = {}


def normed(x: t.Tensor) -> t.Tensor:
    return t.nn.functional.normalize(x, dim=-1)


def records(run: str) -> list[dict]:
    if run not in _records:
        _records[run] = load_records(run)
    return _records[run]


@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lens.html"))


@app.get("/text/<model>/<name>/<int:i>")
def record_text(model: str, name: str, i: int):
    return {"text": state["tokenizer"].apply_chat_template(record_to_conv(records(f"{model}/{name}")[i]), tokenize=False)}


@app.get("/words")
def words():
    return state["tlens"]["words"]


@app.post("/encode")
def encode():
    ids = state["tokenizer"].encode(request.json["text"])
    return {"ids": ids, "toks": [state["tokenizer"].decode([i]) for i in ids]}


CHUNK = 1024


def readouts(layer: int):
    """Per chunk of positions: j-lens logits [chunk, vocab] (float32) and template cosines [chunk, n_templates]."""
    model, jlens, tlens = state["model"], state["jlens"], state["tlens"]
    J = jlens["J"][layer].to(model.device, t.bfloat16)
    templates = normed(tlens["templates"][layer].to(model.device).float())
    for h in state["resid"][layer].split(CHUNK):
        h = h.to(model.device)
        yield model.unembed(model.ln_final(h @ J.T)).float(), normed(h.float()) @ templates.T


@app.post("/run")
def run():
    model, tokenizer, tlens = state["model"], state["tokenizer"], state["tlens"]
    text, k = request.json["text"], request.json["k"]
    toks = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    layers = state["jlens"]["source_layers"]
    cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_resid_pre"))[1]
    state["resid"] = {layer: cache[f"blocks.{layer}.hook_resid_pre"].squeeze(0).cpu() for layer in layers}
    del cache
    t.cuda.empty_cache()
    resp = {"toks": [tokenizer.decode(tok) for tok in toks.squeeze(0)], "layers": layers, "jlens": {}, "tlens": {}}
    for layer in layers:
        resp["jlens"][layer], resp["tlens"][layer] = [], []
        for logits, sims in readouts(layer):
            top = logits.topk(k)
            probs = (top.values - logits.logsumexp(-1, keepdim=True)).exp()
            resp["jlens"][layer] += [[[i, tokenizer.decode(i), round(val, 2), round(p, 4)] for i, val, p in zip(*row)] for row in zip(top.indices.tolist(), top.values.tolist(), probs.tolist())]
            ttop = sims.topk(k)
            resp["tlens"][layer] += [[[i, tlens["words"][i], round(val, 4)] for i, val in zip(*row)] for row in zip(ttop.indices.tolist(), ttop.values.tolist())]
    t.cuda.empty_cache()
    return resp


@app.post("/pins")
def pins():
    reqpins = request.json["pins"]
    out = [{"val": {}, "prob": {}, "rank": {}} for _ in reqpins]
    for layer in state["jlens"]["source_layers"]:
        for o in out:
            o["val"][layer], o["prob"][layer], o["rank"][layer] = [], [], []
        for logits, sims in readouts(layer):
            lse = logits.logsumexp(-1)
            for p, o in zip(reqpins, out):
                if p["kind"] == "tok":
                    col = logits[:, p["id"]]
                    o["val"][layer] += [round(v, 2) for v in col.tolist()]
                    o["prob"][layer] += [round(v, 4) for v in (col - lse).exp().tolist()]
                    o["rank"][layer] += (logits > col[:, None]).sum(-1).tolist()
                else:
                    col = sims[:, p["id"]]
                    o["val"][layer] += [round(v, 4) for v in col.tolist()]
                    o["rank"][layer] += (sims > col[:, None]).sum(-1).tolist()
    t.cuda.empty_cache()
    return {"pins": out}


def serve(model, tokenizer, jlens, tlens, port=PORT):
    if "server" in state:
        state["server"].shutdown()
    state.update(model=model, tokenizer=tokenizer, jlens=jlens, tlens=tlens)
    state["server"] = make_server("0.0.0.0", port, app)
    threading.Thread(target=state["server"].serve_forever, daemon=True).start()
    print(f"lens viewer running at http://localhost:{port}")


def stop():
    state.pop("server").shutdown()
