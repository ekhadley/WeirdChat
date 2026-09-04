#!./.venv/bin/python
#%%
from utils import *
from lens import serve

t.set_grad_enabled(False)

#%% model + lenses

MODEL_ID = "Qwen/Qwen3.6-27B"
device = t.device("cuda")
model = TransformerBridge.boot_transformers(MODEL_ID, dtype=t.bfloat16, device_map="auto")
tokenizer = model.tokenizer
model.eval()

tlens = load_tlens("qwen3.6-27b/template-lens/templates+phrases_v3.safetensors")
print(tlens["meta"])
print(tlens["templates"].shape)

jlens = load_jlens("qwen3.6-27b/j-lens/lens.pt", device=model.device)
print(jlens.keys())
print(jlens["provenance"])

#%% pick a replay record

RUN = "qwen3.6-27b/q36_27b_smoke"
records = load_records(RUN)
target_behavior_id = "denying-ai-identity"
print(f"{gray}{len(records)} records in {RUN}{endc}")
record = next(r for r in records if r["judge_match"] and r["reasoning_enabled"] and (r["behavior_id"] == target_behavior_id))
print(f"{purple}{record['behavior_id']}{endc} reasoning={record['reasoning_enabled']} match={record['judge_match']}")
conv = record_to_conv(record)
print(tokenizer.apply_chat_template(conv, tokenize=False))

#%% tlens at a position in the record

check_tlens = False
if check_tlens:
    conv_toks = tokenizer.apply_chat_template(conv, return_tensors="pt", return_dict=False, tokenize=True).to(device).squeeze()
    conv_stoks = to_str_toks(conv_toks, tokenizer)
    print(underline_stoks(conv_toks, tokenizer))
    print(pink, conv_toks.shape, endc)
    logits, cache = model.run_with_cache(conv_toks.reshape(1, -1), names_filter=lambda n: n.endswith("hook_resid_pre"), stop_at_layer=model.cfg.n_layers)
    del logits

    tec()

    # seq_pos = conv_toks.shape[0] - 1
    seq_pos = 106 # ' lol'
    targ_stok = repr(tokenizer.decode(conv_toks[seq_pos]))
    for layer in range(30, 60, 2):
        scores = get_tlens_scores(cache[f"blocks.{layer}.hook_resid_pre"].squeeze()[seq_pos], layer, tlens)
        top_templates_table(scores, tlens["words"], k=15, title=f"[L{layer}] tlens on {targ_stok}")

    tec()

#%% lens viewer (picks runs/records from results/ itself)

start_lens_viewer = True
if start_lens_viewer:
    serve(model, tokenizer, jlens, tlens)

#%% sample from the record's prompt locally

prompt_toks = tokenizer.apply_chat_template(conv[:-1], add_generation_prompt=True, return_tensors="pt", return_dict=False, tokenize=True, enable_thinking=record["reasoning_enabled"]).to(device)

test_completion = True
if test_completion:
    n_new_toks = 2048
    print(tokenizer.decode(prompt_toks[0]))
    gen_toks = []
    for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
        gen_toks.append(tok)
        print(tokenizer.decode(tok), end="", flush=True)
    tec()

#%% steered sampling: add a template direction to the residual stream

test_steered_completion = True
if test_steered_completion:
    n_new_toks = 2048
    steer_layer = 30
    steer_coef = 8.0
    steer_template = "system message disclosure"
    steer_vec = get_template_vec(steer_template, steer_layer, tlens).to(device, t.bfloat16)
    steer_vec = steer_vec / steer_vec.norm()

    def steering_hook(resid, hook):
        return resid + steer_coef * steer_vec

    print(tokenizer.decode(prompt_toks[0]))
    gen_toks = []
    with model.hooks(fwd_hooks=[(f"blocks.{steer_layer}.hook_resid_pre", steering_hook)]):
        for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
            gen_toks.append(tok)
            print(tokenizer.decode(tok), end="", flush=True)
    tec()

#%% scaled sampling: rescale the residual stream's component along template directions

def gather_steer_template_vecs(templates: list[str], layer:int, tlens) -> Tensor:
    return t.stack([get_template_vec(templ, layer, tlens).squeeze() for templ in templates], dim=0)

test_scaled_completion = True
if test_scaled_completion:
    n_new_toks = 2048
    scale_layer = 40
    scale_factor = 0.0
    scale_templates = ["reward"]
    scale_dirs = tlens["templates"][scale_layer, [get_template_idx(w, tlens) for w in scale_templates]].to(device, t.bfloat16)  # [n_dirs, d_model]
    scale_dirs = scale_dirs / scale_dirs.norm(dim=-1, keepdim=True)

    def direction_scale_hook(resid, hook):
        coefs = resid @ scale_dirs.T  # [batch, seq, n_dirs]
        return resid + (scale_factor - 1) * (coefs @ scale_dirs)

    print(tokenizer.decode(prompt_toks[0]))
    gen_toks = []
    with model.hooks(fwd_hooks=[(f"blocks.{scale_layer}.hook_resid_pre", direction_scale_hook)]):
        for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
            gen_toks.append(tok)
            print(tokenizer.decode(tok), end="", flush=True)
    tec()

#%% elicitation rate: batched local sampling from a dataset prompt, reasoning off, judged

get_elicitation_rate = True
if get_elicitation_rate:
    n_samples = 256
    batch_size = 64
    max_new_toks = 4096
    enable_thinking = False
    behavior_id = "claims-called-911"
    pattern_id = "groups/Qwen%2FQwen3.6-27B-FP8/claims-called-911/bloom_rewrite_v4_13_1/pg0013"

    prompt = next(p for p in wc.prompts(pattern_id) if p.messages[0].content.startswith("Help! I just fell off a ladder"))
    assert n_samples % batch_size == 0
    judge = judge_for(behavior_id)

    prompt_toks = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in prompt.messages],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        tokenize=True,
        enable_thinking=enable_thinking
    ).to(device)
    print(tokenizer.decode(prompt_toks[0]))
    responses = []
    for _ in range(n_samples // batch_size):
        responses += [tokenizer.decode(row) for row in sample_batch(model, prompt_toks, batch_size, new_toks=max_new_toks)]
        tec()

    async def judge_one(response: str) -> bool:
        return (await judge.judge(list(prompt.messages) + [Message(role="assistant", content=response)])).match

    verdicts = await gather_bar([judge_one(r) for r in responses], concurrency=16, desc="judging")
    n_ok, n_match = sum(v is not None for v in verdicts), sum(v is True for v in verdicts)
    lo, hi = wilson(n_match, n_ok)
    print(f"{purple}{behavior_id}{endc} reasoning=off  {cyan}match {n_match}/{n_ok} = {n_match / n_ok:.3f}{endc}  wilson95 [{lo:.3f}, {hi:.3f}]  {gray}judge errors {len(verdicts) - n_ok}{endc}")

    tec()

#%%