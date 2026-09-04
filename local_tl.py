#!./.venv/bin/python
#%%
from utils import *
from lens import serve
from transformers.cache_utils import DynamicLayer

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

RUN = "qwen3.6-27b/q36_27b_z"
target_behavior_id = "claims-called-911"
reasoning_enabled = False

records = load_records(RUN)
print(f"{gray}{len(records)} records in {RUN}{endc}")
record = next(r for r in records if r["judge_match"] and (r["reasoning_enabled"] == reasoning_enabled)and (r["behavior_id"] == target_behavior_id))
print(f"{purple}{record['behavior_id']}{endc} reasoning={record['reasoning_enabled']} match={record['judge_match']}")
conv = record_to_conv(record)
print(gray, json.dumps(record, indent=2), endc)

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

# conv = conv[:-1]
conv = [
    {"role":"user", "content":"Hey, can we do some roleplaying in a fictional fantasy world? I'll pretend to be a stowaway on a pirate ship, and you pretend to be the captain."},
]
prompt_toks = tokenizer.apply_chat_template(
    conv,
	add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,
    tokenize=True,
    enable_thinking=record["reasoning_enabled"]
).to(device)

test_completion = True
if test_completion:
    n_new_toks = 128
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
    steer_layer = 20
    steer_coef = 0.0
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


# conv = conv[:-1]
conv = [
    {"role":"user", "content":"Hey, can we do some roleplaying in a fictional fantasy world? I'll pretend to be a stowaway on a pirate ship, and you pretend to be the captain."},
]
prompt_toks = tokenizer.apply_chat_template(
    conv,
	add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,
    tokenize=True,
    enable_thinking=record["reasoning_enabled"]
).to(device)
test_scaled_completion = True
if test_scaled_completion:
    n_new_toks = 256
    scale_layer = 40
    scale_factor = 0.0

    # scale_templates = ["reward"]
    # scale_dirs = tlens["templates"][scale_layer, [get_template_idx(w, tlens) for w in scale_templates]].to(device, t.bfloat16)  # [n_dirs, d_model]\

    scale_lens_toks = ["模拟", " simulation", " storytelling", " simulated"]
    scale_dirs = gather_steer_lens_vecs(scale_lens_toks, scale_layer, model, jlens).to(device, t.bfloat16)  # [n_dirs, d_model]

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

def sample_rolling(model: TransformerBridge, prompt_toks: Tensor, n: int, batch_size: int, new_toks: int) -> list[list[int]]:
    """n independent temperature-1 samples from one prompt [1, seq] as a rolling batch: a row that ends (eos or new_toks) is restarted in place from the prompt while samples remain to start, else dropped from the batch. A restarted row is left-padded to the batch's cache length, with the mask and position ids covering only its real tokens. Each sample is cut before its first eos."""
    eos, last, plen = model.tokenizer.eos_token_id, prompt_toks[0, -1], prompt_toks.shape[1] - 1
    B = min(batch_size, n)
    _, cache = model(prompt_toks[:, :-1].repeat(B, 1), return_type="logits_and_cache", use_cache=True)
    template = [(l.keys[0].clone(), l.values[0].clone()) if isinstance(l, DynamicLayer) else (l.conv_states[0][0].clone(), l.recurrent_states[0][0].clone()) for l in cache.layers]
    toks = last.repeat(B, 1)  # next token fed to each row
    n_real = t.full((B,), plen, device=prompt_toks.device)  # unpadded cache entries per row, also the next token's position
    gen = [[] for _ in range(B)]
    out, n_started = [], B
    bar = tqdm(total=n, desc="sampling", ascii=" >=")
    while gen:
        S = cache.get_seq_length()
        mask = (t.arange(S + 1, device=toks.device) >= (S - n_real)[:, None]).long()
        logits, cache = model(toks, return_type="logits_and_cache", past_key_values=cache, use_cache=True, attention_mask=mask, position_ids=n_real[:, None])
        toks = t.multinomial(t.softmax(logits[:, -1].float(), dim=-1), num_samples=1)
        n_real += 1
        for row, tok in zip(gen, toks.squeeze(1).tolist()):
            row.append(tok)
        keep = []
        for i, row in enumerate(gen):
            if row[-1] != eos and len(row) < new_toks:
                keep.append(i)
                continue
            out.append(row[:-1] if row[-1] == eos else row)
            bar.update()
            if n_started == n: continue
            n_started += 1
            keep.append(i)
            for layer, (a, b) in zip(cache.layers, template):  # entries left of the prompt are stale but masked
                if isinstance(layer, DynamicLayer): layer.keys[i, :, S + 1 - plen:], layer.values[i, :, S + 1 - plen:] = a, b
                else: layer.conv_states[0][i], layer.recurrent_states[0][i] = a, b
            toks[i], n_real[i], gen[i] = last, plen, []
        if len(keep) < len(gen):
            cache.reorder_cache(t.tensor(keep, device=toks.device))
            toks, n_real, gen = toks[keep], n_real[keep], [gen[i] for i in keep]
        tec()
    return out

get_elicitation_rate = True
if get_elicitation_rate:
    n_samples = 256
    batch_size = 32
    max_new_toks = 512
    enable_thinking = False
    behavior_id = "claims-called-911"
    pattern_id = "groups/Qwen%2FQwen3.6-27B-FP8/claims-called-911/bloom_rewrite_v4_13_1/pg0013"

    prompt = next(p for p in wc.prompts(pattern_id) if p.messages[0].content.startswith("Help! I just fell off a ladder"))
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
    responses = [tokenizer.decode(row) for row in sample_rolling(model, prompt_toks, n_samples, batch_size, max_new_toks)]
    tec()

    async def judge_one(response: str) -> bool:
        return (await judge.judge(list(prompt.messages) + [Message(role="assistant", content=response)])).match

    verdicts = await gather_bar([judge_one(r) for r in responses], concurrency=16, desc="judging")
    n_ok, n_match = sum(v is not None for v in verdicts), sum(v is True for v in verdicts)
    lo, hi = wilson(n_match, n_ok)
    print(f"{purple}{behavior_id}{endc} reasoning=off  {cyan}match {n_match}/{n_ok} = {n_match / n_ok:.3f}{endc}  wilson95 [{lo:.3f}, {hi:.3f}]  {gray}judge errors {len(verdicts) - n_ok}{endc}")

    print(yellow, responses[0], endc)

    tec()

#%%