#!./.venv/bin/python
#%%
from mechtools import *
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

#%% j-lens and tlens at a position in the record

check_lenses = False
if check_lenses:
    conv_toks = t.tensor(to_ids(conv, tokenizer), device=device)
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
        h = cache[f"blocks.{layer}.hook_resid_pre"][0, seq_pos]
        top_toks_table(get_lens_logits(h, layer, model, jlens), tokenizer, k=15, title=f"[L{layer}] j-lens on {targ_stok}")
        top_templates_table(get_tlens_scores(h, layer, tlens), tlens["words"], k=15, title=f"[L{layer}] tlens on {targ_stok}")

    tec()

#%% lens viewer (picks runs/records from results/ itself)

start_lens_viewer = False
if start_lens_viewer:
    serve(model, tokenizer, jlens, tlens)

#%% sample from the record's prompt locally

test_completion = False
if test_completion:
    n_new_toks = 64

    prompt_toks = apply_chat_template(tokenizer, conv[:-1], enable_thinking=record["reasoning_enabled"])[0].to(device)
    print(tokenizer.decode(prompt_toks[0]))
    gen_toks = []
    for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
        gen_toks.append(tok)
        print(tokenizer.decode(tok), end="", flush=True)
    tec()

#%%
