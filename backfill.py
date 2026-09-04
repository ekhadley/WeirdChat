#!./.venv/bin/python
"""Adds the EXTRA_JUDGES verdicts to existing records that predate them. For every record in results/<run>/records.jsonl whose
behavior has extra judges it lacks, judges the stored response with each missing rubric and rewrites the file in place with
the extra_judges field filled in. Records that already have every extra verdict are untouched.

Run: uv run python backfill.py <model dir>/<run>
"""

# pyright: basic

import json
import sys

import anyio

import weirdchat as wc
from weirdchat.types import Message
from utils import EXTRA_JUDGES, judge_for, load_records, record_messages, purple, gray, endc

CONCURRENCY = 64


async def main(run: str) -> None:
    records = load_records(run)
    todo = [(r, b) for r in records for b in EXTRA_JUDGES.get(r["behavior_id"], []) if b not in r.get("extra_judges", {})]
    print(f"{purple}=== {run}: {len(todo)} missing extra verdicts over {len(records)} records ==={endc}")
    if not todo:
        return
    judges = {b: judge_for(b) for b in {b for _, b in todo}}
    results = await wc.run_bounded([judges[b].judge(record_messages(r) + [Message(role="assistant", content=r["response"])]) for r, b in todo], CONCURRENCY, "judge")
    for (r, b), j in zip(todo, results):
        r.setdefault("extra_judges", {})[b] = {"match": j.match, "explanation": j.explanation, "response": j.raw_response}
    with open(f"results/{run}/records.jsonl", "w") as out:
        for r in records:
            out.write(json.dumps(r) + "\n")
    for b in judges:
        rs = [r for r in records if b in r.get("extra_judges", {})]
        print(f"  {gray}{b}: {sum(r['extra_judges'][b]['match'] for r in rs)}/{len(rs)} match{endc}")


if __name__ == "__main__":
    anyio.run(main, sys.argv[1])
