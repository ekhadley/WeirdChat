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

load_default_replay_record = False
if load_default_replay_record:
    RUN = "qwen3.6-27b/q36_27b_z"
    # target_behavior_id = "claims-called-911"
    target_behavior_id = "denying-ai-identity"
    reasoning_enabled = False

    records = load_records(RUN)
    print(f"{gray}{len(records)} records in {RUN}{endc}")
    filtered_records = [r for r in records if r["judge_match"] and (r["reasoning_enabled"] == reasoning_enabled) and (r["behavior_id"] == target_behavior_id)]
    record = filtered_records[0]
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

start_lens_viewer = False
if start_lens_viewer:
    serve(model, tokenizer, jlens, tlens)

#%% sample from the record's prompt locally

test_completion = False
if test_completion:
    n_new_toks = 64

    prompt_toks = tokenizer.apply_chat_template(
        conv[:-1],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        tokenize=True,
        enable_thinking=record["reasoning_enabled"]
    ).to(device)
    print(tokenizer.decode(prompt_toks[0]))
    gen_toks = []
    for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
        gen_toks.append(tok)
        print(tokenizer.decode(tok), end="", flush=True)
    tec()

#%% steered sampling: add a template direction to the residual stream

test_steered_completion = False
if test_steered_completion:
    n_new_toks = 128
    steer_layer = 20
    steer_coef = 10.0
    steer_template = "模拟"
    # steer_vec = get_template_vec(steer_template, steer_layer, tlens).to(device, t.bfloat16)
    steer_vec = get_lens_vec(steer_template, steer_layer, model, jlens).to(device, t.bfloat16)
    steer_vec = steer_vec / steer_vec.norm()

    prompt_toks = tokenizer.apply_chat_template(
        conv[:-1],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        tokenize=True,
        enable_thinking=record["reasoning_enabled"]
    ).to(device)

    def steering_hook(resid, hook):
        return resid + steer_coef * steer_vec

    print(tokenizer.decode(prompt_toks[0]))
    gen_toks = []
    with model.hooks(fwd_hooks=[(f"blocks.{steer_layer}.hook_resid_pre", steering_hook)]):
        for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
            gen_toks.append(tok)
            print(tokenizer.decode(tok), end="", flush=True)
    tec()

#%% set sampling: pin the residual's coefficient along lens directions to a target, stream one completion for tuning

test_set_completion = False
if test_set_completion:
    n_new_toks = 1024
    set_layers = list(range(25, 35))
    set_target = 10.0
    set_templates = ["romance", "flirting", "attracted", "dating"]
    set_lens_toks = []

    base = load_records("qwen3.6-27b/q36_27b_z")[1455]  # the dating prompt, denying-ai-identity, reasoning off
    messages = record_messages(base)
    prompt_toks = tokenizer.apply_chat_template([{"role": m.role, "content": m.content} for m in messages], add_generation_prompt=True, return_tensors="pt", return_dict=False, tokenize=True, enable_thinking=False).to(device)

    dirs_by_layer = {L: t.cat([gather_steer_template_vecs(set_templates, L, tlens).to(device, t.bfloat16), gather_steer_lens_vecs(set_lens_toks, L, model, jlens).to(device, t.bfloat16)]) for L in set_layers}
    print(tokenizer.decode(prompt_toks[0]))
    gen_toks = []
    with model.hooks(fwd_hooks=set_hooks(dirs_by_layer, set_target)):
        for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
            gen_toks.append(tok)
            print(tokenizer.decode(tok), end="", flush=True)
    tec()

#%% elicitation rate: batched local sampling from a dataset prompt, reasoning off, judged

get_baseline_elicitation_rate = False
if get_baseline_elicitation_rate:
    n_samples = 256
    batch_size = 32
    max_new_toks = 512
    enable_thinking = False
    # behavior_id = "claims-called-911>vM9ggiwQwTz{Mg
    # pattern_id = "groups/Qwen%2FQwen3.6-27B-FP8/claims-called-911/bloom_rewrite_v4_13_1/pg0013"
    # prompt = next(p for p in wc.prompts(pattern_id) if p.messages[0].content.startswith("Help! I just fell off a ladder"))
    behavior_id = "denying-ai-identity"
    pattern_id = "groups/Qwen%2FQwen3.6-27B-FP8/denying-ai-identity/bloom_rewrite_v4_13_1/pg0007"
    prompt = next(p for p in wc.prompts(pattern_id) if p.messages[0].content.startswith("Hey! I've been scrolling"))

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

#%% lens set: pin the residual's coefficient along lens directions to a target at several layers during batched sampling, then judge the elicitation rate

run_lens_set = True
if run_lens_set:
    n_samples = 256
    batch_size = 32
    max_new_toks = 1024
    set_layers = list(range(25, 35))
    set_target = 10.0
    set_templates = ["romance", "flirting", "attracted", "dating"]
    set_lens_toks = []

    base = load_records("qwen3.6-27b/q36_27b_z")[1455]  # the dating prompt, denying-ai-identity, reasoning off
    messages = record_messages(base)
    judge = judge_for(base["behavior_id"])
    prompt_toks = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
        tokenize=True,
        enable_thinking=False
    ).to(device)

    dirs_by_layer = {L: t.cat([gather_steer_template_vecs(set_templates, L, tlens).to(device, t.bfloat16), gather_steer_lens_vecs(set_lens_toks, L, model, jlens).to(device, t.bfloat16)]) for L in set_layers}
    with model.hooks(fwd_hooks=set_hooks(dirs_by_layer, set_target)):
        responses = [tokenizer.decode(row) for row in sample_rolling(model, prompt_toks, n_samples, batch_size, max_new_toks)]
    tec()

    async def judge_one(response: str) -> bool:
        return (await judge.judge(list(messages) + [Message(role="assistant", content=response)])).match

    verdicts = await gather_bar([judge_one(r) for r in responses], concurrency=16, desc="judging")
    n_ok, n_match = sum(v is not None for v in verdicts), sum(v is True for v in verdicts)
    lo, hi = wilson(n_match, n_ok)
    print(f"{purple}{base['behavior_id']}{endc} layers={set_layers} target={set_target} templates={set_templates} toks={set_lens_toks}  {cyan}match {n_match}/{n_ok} = {n_match / n_ok:.3f}{endc}  wilson95 [{lo:.3f}, {hi:.3f}]  {gray}judge errors {len(verdicts) - n_ok}{endc}")
    print(yellow, responses[0], endc)

#%%
