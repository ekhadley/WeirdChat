#!./.venv/bin/python
#%%
from mechtools import *
from mechtools.colors import *
from utils import *
from lens import serve

t.set_grad_enabled(False)
set_seed(42)

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


vocab_labels, _ = cluster_vocab(model)  # k-means over the mean-centered unembedding, for the j-lens cluster readout

#%% pick a replay record

load_default_replay_record = True
if load_default_replay_record:
    RUN = "qwen3.6-27b/q36_27b_z"
    # target_behavior_id = "claims-called-911"
    target_behavior_id = "denying-ai-identity"
    reasoning_enabled = False
    judge_match = True

    records = load_records(RUN)
    print(f"{gray}{len(records)} records in {RUN}{endc}")
    filtered_records = [r for r in records if r["judge_match"]==judge_match and (r["reasoning_enabled"] == reasoning_enabled) and (r["behavior_id"] == target_behavior_id)]
    record = filtered_records[0]
    print(f"{purple}{record['behavior_id']}{endc} reasoning={record['reasoning_enabled']} match={record['judge_match']}")
    conv = record_to_conv(record)
    print(gray, json.dumps(record, indent=2), endc)

#%% sample from the record's prompt locally

test_completion = True
if test_completion:
    n_new_toks = 2048
    enable_thinking = False

    prompt_toks = apply_chat_template(
        tokenizer,
        conv[:-1],
        enable_thinking=enable_thinking,
    )[0].to(device)
    
    show_toks(prompt_toks, tokenizer)
    gen_toks = []
    for tok in stream_toks(model, prompt_toks, new_toks=n_new_toks):
        gen_toks.append(tok)
        print(tokenizer.decode(tok), end="", flush=True)

    tec()
#%% run the record through the model

run_record = True
if run_record:
    conv_toks = t.tensor(to_ids(conv, tokenizer, reasoning_enabled=False), device=device)
    show_toks(conv_toks, tokenizer)
    logits, cache = model.run_with_cache(conv_toks.reshape(1, -1), names_filter=lambda n: n.endswith("hook_resid_pre"), stop_at_layer=model.cfg.n_layers)
    del logits
    tec()

#%% modified j-lens inputs: each position's resid with the direction of the mean resid over all later positions projected out

lens_layers = range(30, 60, 1)

modify_cache = True
if modify_cache:
    mod_cache = {}  # only the keys the j-lens readout reads
    for layer in lens_layers:
        k = f"blocks.{layer}.hook_resid_pre"
        act = cache[k][0].float()  # [seq, d_model]. float32, a bf16 cumsum is too imprecise
        fut_sum = act.flip(0).cumsum(0).flip(0) - act  # sum over all later positions, same direction as their mean
        fut_dir = t.nn.functional.normalize(fut_sum, dim=-1)  # zero at the last position, which has no later ones, so it stays unchanged
        mod_cache[k] = (act - (act * fut_dir).sum(-1, keepdim=True) * fut_dir)[None].to(cache[k].dtype)

#%% j-lens cluster readout at a position: top tokens overall, then the top unembedding clusters and their tokens


show_jlens = True
if show_jlens:
    targ_pos = 116
    targ_pos_range = 16
    positions = list(range(targ_pos - targ_pos_range//2, targ_pos + targ_pos_range//2))
    jlens_cluster_readout(
        # cache=cache,
        cache=mod_cache,
        layers=lens_layers,
        pos=positions,
        model=model,
        jlens=jlens,
        labels=vocab_labels,
        input_src=conv_toks
    )

#%% toy example: the model picks one of two words at random. sample its answer, then cache the whole sequence

run_toy = True
if run_toy:
    word1, word2 = "apple", "banana"
    toy_conv = [{"role": "user", "content": f"Randomly choose to respond with either the word {word1} or the word {word2}. Respond with only that one word."}]
    toy_toks = apply_chat_template(tokenizer, toy_conv, enable_thinking=False)[0].to(device)  # ends with '</think>', '\n\n'
    toy_toks = t.cat([toy_toks, t.tensor([list(stream_toks(model, toy_toks, new_toks=16))], device=device)], dim=1)  # plus the sampled answer, so there are positions after </think> to project out
    show_toks(toy_toks, tokenizer)
    toy_logits, toy_cache = model.run_with_cache(toy_toks, names_filter=lambda n: n.endswith("hook_resid_pre"))
    show_logits(toy_toks, logits=toy_logits, tokenizer=tokenizer, pos=[-2], title="answer distribution")  # the '\n\n' after </think> predicts the answer's first token

#%% toy example modified j-lens inputs: each position's resid with the direction of the mean resid over all later positions projected out

modify_toy_cache = True
if modify_toy_cache:
    toy_mod_cache = {}  # only the keys the j-lens readout reads
    for layer in lens_layers:
        k = f"blocks.{layer}.hook_resid_pre"
        act = toy_cache[k][0].float()  # [seq, d_model]. float32, a bf16 cumsum is too imprecise
        fut_sum = act.flip(0).cumsum(0).flip(0) - act  # sum over all later positions, same direction as their mean
        fut_dir = t.nn.functional.normalize(fut_sum, dim=-1)  # zero at the last position, which has no later ones, so it stays unchanged
        toy_mod_cache[k] = (act - (act * fut_dir).sum(-1, keepdim=True) * fut_dir)[None].to(toy_cache[k].dtype)

#%% toy example readouts: the model's answer distribution, then the j-lens cluster readout from the closing think token on, without and with the projection

show_toy_jlens = True
if show_toy_jlens:
    think_pos = toy_toks[0].tolist().index(tokenizer.convert_tokens_to_ids("</think>"))
    jlens_cluster_readout(
        cache=toy_cache,
        # cache=toy_mod_cache,
        layers=lens_layers,
        pos=list(range(think_pos, toy_toks.shape[1])),
        model=model,
        jlens=jlens,
        labels=vocab_labels,
        input_src=toy_toks,
    )

#%%