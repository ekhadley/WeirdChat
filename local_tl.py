#!./.venv/bin/python
#%%
from mechtools import *
from mechtools.colors import *
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

load_default_replay_record = True
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

#%% run the record through the model

run_record = True
if run_record:
    conv_toks = t.tensor(to_ids(conv, tokenizer), device=device)
    print(pink, conv_toks.shape, endc)
    logits, cache = model.run_with_cache(conv_toks.reshape(1, -1), names_filter=lambda n: n.endswith("hook_resid_pre"), stop_at_layer=model.cfg.n_layers)
    del logits
    tec()

#%% j-lens readout at a position

show_jlens = True
if show_jlens:
    jlens_readout(
        cache=cache,
        layers=range(30, 60),
        pos=116,
        model=model,
        jlens=jlens,
        k=15,
        input_src=conv_toks
    )

#%% template-lens readout at a position

show_tlens = False
if show_tlens:
    seq_pos = 106 # ' lol'
    tlens_readout(cache, range(30, 60, 2), seq_pos, tlens, k=15, input_src=conv_toks, tokenizer=tokenizer)

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
