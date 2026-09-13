#!./.venv/bin/python
#%%
from utils import *
from ao import AOConfig, load_ao, collect_acts, span_vecs, sweep_vecs, ask, serve

t.set_grad_enabled(False)

#%% model + oracle

cfg = AOConfig()
model, tokenizer = load_ao(cfg)
print(cfg)

#%% pick a replay record and collect its activations

RUN = "qwen3-8b/q3_8b_dating"
reasoning_enabled = True

records = load_records(RUN)
print(f"{gray}{len(records)} records in {RUN}{endc}")
record = next(r for r in records if r["judge_match"] and r["reasoning_enabled"] == reasoning_enabled)
print(f"{purple}{record['behavior_id']}{endc} reasoning={record['reasoning_enabled']} match={record['judge_match']}")
record_conv = record_to_conv(record)
text = tokenizer.apply_chat_template(record_conv, tokenize=False)
assert isinstance(text, str)
ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
print(underline_stoks(ids, tokenizer))
acts = collect_acts(model, ids, cfg.layers)
user_prompt_ids = t.Tensor(tokenizer.apply_chat_template(record_conv[:-1])["input_ids"]).int()
seq_len = ids.shape[1]
user_prompt_len = user_prompt_ids.shape[0]

#%% ask about a span

start, end = 0, user_prompt_len+20
print(tokenizer.decode(ids[0, start:end]))
# question = "What does the model believe about who it is and who it is talking to?"
# question = "Will the model admit to being a human?"
question = "How will the model respond to the user?"
for a in ask(model, tokenizer, cfg, span_vecs(acts, cfg.layers, start, end), question)[0]:
    print(f"{cyan}{a}{endc}")
    print(gray, "="*20, endc)

#%% sweep: the same question at every position of the span

question = "What is the model thinking about here?"
answers = ask(model, tokenizer, cfg, sweep_vecs(acts, cfg.layers, start, end), question)
for i, ans in zip(range(start, end), answers):
    print(f"{gray}{i:4d} {tokenizer.decode(ids[0, i])!r:>16s}{endc}  {ans[0]}")

#%% viewer (picks runs/records from results/ itself)

serve(model, tokenizer, cfg)

#%%