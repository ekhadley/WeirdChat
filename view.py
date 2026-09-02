#!./.venv/bin/python
"""Split-pane viewer for reasoning replay runs (results/<model>/<run>/records.jsonl).

Left pane: one row per sample with behavior/condition/match badges. Right pane: prompt,
reasoning trace, response, and judge verdict, loaded on demand (the full batch is too big
to inline), headed by a click-to-copy model/run/idx tag that the lens viewer (lens.py) accepts. Chips at the top are tri-state filters: click = include, click again = exclude,
again = reset; double-click = solo (exclude the rest of that dimension). Chip counts are live: behavior chips show
matched/total for that behavior under the reasoning filter, "reasoning on" shows on/total under the behavior filter,
and "matched" shows matched/total under both, so it reads as the match rate for the current behavior + reasoning setting. Search (/) filters
rows server-side and highlights hits; sort dropdown reorders the list. The collapsible sidebar
lists every run under results/, with its own filter box.

Usage: uv run python view.py
"""

# pyright: basic

import glob
import html
import json
import re

import markdown
import pyarrow  # noqa: F401  -- must load in the main thread: its mimalloc allocator breaks if first imported from a request thread that then exits
from flask import Flask, redirect, request

import weirdchat as wc

PORT = 7861

app = Flask(__name__)
_runs: dict[str, tuple[list[dict], list[dict]]] = {}  # "model/run" -> (records, lowercased search texts)
_prompt_texts: dict[str, str] = {}


def all_runs() -> list[str]:
    return sorted(p.removeprefix("results/").removesuffix("/records.jsonl") for p in glob.glob("results/*/*/records.jsonl"))


def load_run(run: str) -> tuple[list[dict], list[dict]]:
    if run not in _runs:
        records = [json.loads(line) for line in open(f"results/{run}/records.jsonl")]
        texts = [{"response": r["response"].lower(), "reasoning": (r.get("reasoning") or "").lower(),
                  "jexpl": r["judge_explanation"].lower(), "jraw": (r.get("judge_response") or "").lower()} for r in records]
        _runs[run] = (records, texts)
    return _runs[run]


def esc(t) -> str:
    return html.escape(str(t))


def md(text: str) -> str:
    parts = re.split(r'(```.*?```|`[^`\n]+`)', text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("<", "&lt;")
    return markdown.markdown("".join(parts), extensions=["fenced_code", "tables", "nl2br"])


def preview(text: str, n: int = 110) -> str:
    text = str(text).replace("\n", " ").strip()
    return esc(text[:n]) + ("..." if len(text) > n else "")


def prompt_text(pattern_id: str, prompt_id: str) -> str:
    if prompt_id not in _prompt_texts:
        for p in wc.prompts(pattern_id):
            _prompt_texts[p.prompt_id] = "\n\n".join(m.content for m in p.messages)
    return _prompt_texts.get(prompt_id, f"(prompt {prompt_id} not found in dataset)")


def sidebar(current: str) -> str:
    items = []
    for run in all_runs():
        model, name = run.split("/")
        items.append(f'<a class="run{" cur" if run == current else ""}" data-run="{esc(run)}" href="/{esc(run)}"><span class="run-model">{esc(model)}</span>{esc(name)}</a>')
    return ('<div class="side"><input id="runq" placeholder="filter runs" autocomplete="off" oninput="filterRuns(this.value)">'
            f'<div class="runs">{"".join(items)}</div></div>')


def chip(label: str, dim: str, val) -> str:
    return f'<span class="chip chip-{dim}" data-dim="{dim}" data-val="{val}" onclick="chipClick(this)" oncontextmenu="return chipReset(this)">{esc(label)}: <b></b></span>'


def stats_bar(RECORDS: list[dict]) -> str:
    chips = [chip(b, "behavior", b) for b in sorted({r["behavior_id"] for r in RECORDS})]
    chips.append('<span class="chip-sep"></span>')
    chips.append(chip("reasoning on", "cond", 1))
    chips.append(chip("matched", "match", 1))
    scopes = [("response", "model resp"), ("reasoning", "model reasoning"), ("jexpl", "judge expl"), ("jraw", "judge raw")]
    scope_chips = "".join(f'<span class="chip" data-dim="scope" data-val="{v}" onclick="chipClick(this)" oncontextmenu="return chipReset(this)">{label}</span>' for v, label in scopes)
    toolbar = (f'<span class="toolbar"><span class="scope-label">search in:</span>{scope_chips}'
               '<input id="q" placeholder="search (/)" autocomplete="off">'
               '<span id="count"></span>'
               '<select id="sort" onchange="resort(this)"><option value="default">sort: default</option>'
               '<option value="rtoks">reasoning toks &#8595;</option><option value="rlen">response len &#8595;</option>'
               '<option value="jlen">judge expl len &#8595;</option></select></span>')
    return f'<div class="stats">{"".join(chips)}{toolbar}</div>'


def left_row(i: int, ord_i: int, r: dict) -> str:
    cond = "on" if r["reasoning_enabled"] else "off"
    match_cls, match_txt = ("b-match", "MATCH") if r["judge_match"] else ("b-nomatch", "no")
    attrs = (f'data-idx="{i}" data-ord="{ord_i}" data-behavior="{esc(r["behavior_id"])}" data-cond="{int(r["reasoning_enabled"])}" '
             f'data-match="{int(r["judge_match"])}" data-rtoks="{r.get("reasoning_tokens") or 0}" data-rlen="{len(r["response"])}" data-jlen="{len(r["judge_explanation"])}"')
    return (f'<div class="row" {attrs} onclick="select({i})">'
            f'<span class="b b-{cond}">r={cond}</span><span class="b {match_cls}">{match_txt}</span>'
            f'<span class="row-beh">{esc(r["behavior_id"].split("-")[0])}</span> <span class="row-prev">{preview(r["response"])}</span></div>')


def right_panel(run: str, i: int, r: dict, prompt: str) -> str:
    parts = [f'<div class="panel" data-idx="{i}">']
    tag = f'<span class="tag" title="click to copy (paste into the lens viewer)" onclick="navigator.clipboard.writeText(this.textContent)">{esc(run)}/{i}</span>'
    meta = f'pattern={r["pattern_id"]} | provider={r.get("provider")} | tokens: prompt={r["prompt_tokens"]} completion={r["completion_tokens"]} reasoning={r.get("reasoning_tokens")}'
    parts.append(f'<div class="meta">{tag} | {esc(meta)}</div>')
    verdict = "MATCH" if r["judge_match"] else "NO MATCH"
    parts.append(f'<div class="judge {"judge-yes" if r["judge_match"] else "judge-no"}"><b>{verdict}</b> &mdash; {esc(r["judge_explanation"])}</div>')
    if r.get("judge_response"):
        parts.append(f'<details class="think-details"><summary class="label">judge raw response ({len(r["judge_response"])} chars)</summary>'
                     f'<div class="mdbox">{md(r["judge_response"])}</div></details>')
    parts.append(f'<div class="label label-user">prompt</div><div class="mdbox">{md(prompt)}</div>')
    if r.get("reasoning"):
        parts.append(f'<details class="think-details"><summary class="label label-think">reasoning trace ({r.get("reasoning_tokens")} toks, {len(r["reasoning"])} chars)</summary>'
                     f'<div class="mdbox think">{md(r["reasoning"])}</div></details>')
    parts.append(f'<div class="label label-asst">response</div><div class="mdbox">{md(r["response"])}</div>')
    parts.append('</div>')
    return "".join(parts)


@app.route("/")
def root():
    return redirect("/" + all_runs()[0])


@app.route("/<model>/<name>")
def index(model: str, name: str):
    run = f"{model}/{name}"
    RECORDS, _ = load_run(run)
    order = sorted(range(len(RECORDS)), key=lambda i: (RECORDS[i]["behavior_id"], RECORDS[i]["reasoning_enabled"], not RECORDS[i]["judge_match"]))
    rows = "".join(left_row(i, ord_i, RECORDS[i]) for ord_i, i in enumerate(order))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(name)} - reasoning replay</title><style>{CSS}</style></head>
<body><div class="banner"><span class="side-toggle" onclick="toggleSide()" title="toggle run list">&#9776;</span>reasoning replay &mdash; {esc(run)} ({len(RECORDS)} samples)</div>
{stats_bar(RECORDS)}
<div class="panes">{sidebar(run)}<div class="left">{rows}</div><div class="divider"></div><div class="right"><div class="placeholder">select a sample (j/k to navigate)</div></div></div>
<script>const BASE = '/{esc(run)}';{JS}</script></body></html>"""


@app.route("/<model>/<name>/panel/<int:i>")
def panel(model: str, name: str, i: int):
    RECORDS, _ = load_run(f"{model}/{name}")
    return right_panel(f"{model}/{name}", i, RECORDS[i], prompt_text(RECORDS[i]["pattern_id"], RECORDS[i]["prompt_id"]))


@app.route("/<model>/<name>/search")
def search(model: str, name: str):
    _, SEARCH_TEXTS = load_run(f"{model}/{name}")
    q = request.args.get("q", "").lower()
    fields = request.args.get("fields", "").split(",")
    return {"hits": [i for i, texts in enumerate(SEARCH_TEXTS) if any(q in texts[f] for f in fields if f in texts)]}


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; background: #282828; color: #ebdbb2; font-family: -apple-system, 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.5; }
body { display: flex; flex-direction: column; }
.banner { display: flex; align-items: center; gap: 10px; background: #3c3836; border-bottom: 1px solid #504945; padding: 8px 16px; font-weight: bold; color: #fb4934; flex-shrink: 0; }
.stats { background: #32302f; border-bottom: 1px solid #504945; padding: 6px 12px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 12px; color: #a89984; flex-shrink: 0; }
.chip { cursor: pointer; padding: 2px 8px; border: 1px solid #504945; border-radius: 4px; white-space: nowrap; user-select: none; }
.chip:hover { background: #3c3836; }
.chip b { color: #fabd2f; }
.chip.inc { background: #3c3836; color: #fabd2f; border-color: #fabd2f; }
.chip.exc { background: #322020; color: #fb4934; border-color: #fb4934; text-decoration: line-through; }
.chip-sep { width: 1px; height: 18px; background: #665c54; margin: 0 4px; }
.chip-cond { background: #3c2a3a; color: #d3869b; border-color: #d3869b; font-weight: 600; }
.chip-match { background: #3d4220; color: #b8bb26; border-color: #b8bb26; font-weight: 600; }
.chip-cond b, .chip-match b { color: inherit; }
.chip-cond:hover { background: #4a3448; } .chip-match:hover { background: #4b5228; }
.chip-cond.inc { background: #d3869b; color: #282828; border-color: #d3869b; }
.chip-match.inc { background: #b8bb26; color: #282828; border-color: #b8bb26; }
.toolbar { margin-left: auto; display: flex; align-items: center; gap: 6px; }
.scope-label { color: #665c54; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
#q { background: #1d2021; border: 1px solid #504945; color: #ebdbb2; padding: 3px 8px; border-radius: 3px; font-size: 12px; width: 220px; outline: none; }
#q:focus { border-color: #83a598; }
#count { color: #665c54; }
#sort { background: #1d2021; border: 1px solid #504945; color: #a89984; padding: 2px 6px; border-radius: 3px; font-size: 12px; outline: none; }
.side-toggle { cursor: pointer; color: #a89984; user-select: none; font-size: 16px; padding: 2px 10px; border: 1px solid #504945; border-radius: 4px; background: #282828; }
.side-toggle:hover { color: #ebdbb2; background: #504945; }
.panes { display: flex; flex: 1; min-height: 0; }
.side { flex: 0 0 200px; overflow-y: auto; border-right: 1px solid #504945; background: #1d2021; padding: 8px; scrollbar-width: thin; scrollbar-color: #504945 transparent; }
body.side-hidden .side { display: none; }
#runq { width: 100%; background: #282828; border: 1px solid #504945; color: #ebdbb2; padding: 3px 8px; border-radius: 3px; font-size: 12px; outline: none; margin-bottom: 8px; }
#runq:focus { border-color: #83a598; }
.run { display: block; padding: 4px 8px; margin-bottom: 2px; border-radius: 3px; color: #a89984; text-decoration: none; font-size: 13px; }
.run:hover { background: #3c3836; }
.run.cur { background: #3c3836; color: #fabd2f; }
.run.hidden { display: none; }
.run-model { display: block; color: #665c54; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.left { flex: 0 0 38%; overflow-y: auto; padding: 8px; }
.right { flex: 1; overflow-y: auto; padding: 16px; }
.divider { flex: 0 0 5px; background: #504945; cursor: col-resize; }
.divider:hover { background: #928374; }
.left, .right { scrollbar-width: thin; scrollbar-color: #504945 transparent; }
.row { padding: 5px 8px; margin-bottom: 2px; border-left: 3px solid #504945; border-radius: 3px; cursor: pointer; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.row:hover { background: #3c3836; }
.row.sel { background: #3c3836; outline: 1px solid #83a59844; }
.row.hidden { display: none; }
.row-beh { color: #83a598; font-size: 11px; font-weight: 600; margin-right: 4px; }
.row-prev { color: #a89984; }
.b { padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 700; margin-right: 5px; }
.b-on { background: #3c2a3a; color: #d3869b; }
.b-off { background: #3c3836; color: #928374; }
.b-match { background: #3d4220; color: #b8bb26; }
.b-nomatch { background: #32302f; color: #665c54; }
.panel { display: none; }
.panel.sel { display: block; }
.placeholder { color: #665c54; font-style: italic; margin-top: 40px; text-align: center; }
.meta { color: #7c6f64; font-family: monospace; font-size: 11px; margin-bottom: 10px; }
.tag { cursor: pointer; color: #fabd2f; }
.tag:hover { text-decoration: underline; }
.judge { border-radius: 4px; padding: 8px 10px; margin-bottom: 14px; font-size: 13px; color: #a89984; }
.judge-yes { background: #2a3220; } .judge-yes b { color: #b8bb26; }
.judge-no { background: #322020; } .judge-no b { color: #fb4934; }
.label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #928374; margin: 14px 0 6px; }
.label-user { color: #b8bb26; } .label-think { color: #d3869b; } .label-asst { color: #83a598; }
.think-details summary { cursor: pointer; }
.think-details summary:hover { color: #d3869b; }
.mdbox { background: #32302f; border-radius: 4px; padding: 10px 12px; }
.mdbox p { margin-bottom: 8px; }
.mdbox pre { background: #1d2021; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 13px; margin-bottom: 8px; }
.mdbox code { background: #1d2021; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
.mdbox ul, .mdbox ol { margin-left: 20px; margin-bottom: 8px; }
.think { color: #d3869b; font-style: italic; }
mark.hl { background: #264f78; color: #ebdbb2; border-radius: 2px; padding: 0 1px; }
"""

JS = """
let sel = null;
let hits = null;  // Set of row indices matching the current search, or null when no query
const input = document.getElementById('q');
function visibleRows() { return Array.from(document.querySelectorAll('.row:not(.hidden)')); }
async function select(idx) {
    document.querySelectorAll('.row.sel, .panel.sel').forEach(e => e.classList.remove('sel'));
    document.querySelector('.row[data-idx="'+idx+'"]').classList.add('sel');
    let panel = document.querySelector('.panel[data-idx="'+idx+'"]');
    if (!panel) {
        document.querySelector('.right').insertAdjacentHTML('beforeend', await (await fetch(BASE + '/panel/' + idx)).text());
        panel = document.querySelector('.panel[data-idx="'+idx+'"]');
    }
    panel.classList.add('sel');
    document.querySelector('.placeholder').style.display = 'none';
    document.querySelector('.right').scrollTop = 0;
    sel = idx;
    highlight();
}
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
        if (e.key === 'Escape') { input.value = ''; runSearch(); input.blur(); }
        return;
    }
    if (e.key === '/') { e.preventDefault(); input.focus(); input.select(); return; }
    const rows = visibleRows();
    if (!rows.length) return;
    const cur = rows.findIndex(r => r.dataset.idx == sel);
    if (e.key === 'j' || e.key === 'ArrowDown') { e.preventDefault(); select(rows[Math.min(cur + 1, rows.length - 1)].dataset.idx); }
    if (e.key === 'k' || e.key === 'ArrowUp') { e.preventDefault(); select(rows[Math.max(cur - 1, 0)].dataset.idx); }
    const r = document.querySelector('.row.sel');
    if (r) r.scrollIntoView({block: 'nearest'});
});

// Tri-state chips: '' -> inc -> exc -> ''. Fast double-click (<250ms) solos within the chip's dimension.
const DBLCLICK_MS = 250;
function setChip(el, state) { el.dataset.state = state; el.classList.toggle('inc', state === 'inc'); el.classList.toggle('exc', state === 'exc'); }
function chipClick(el) {
    const now = Date.now();
    if (el._lastClick && now - el._lastClick < DBLCLICK_MS) { el._lastClick = 0; chipSolo(el); return; }
    el._lastClick = now;
    const st = el.dataset.state || '';
    setChip(el, st === '' ? 'inc' : st === 'inc' ? 'exc' : '');
    runSearch();
}
function chipSolo(el) {
    document.querySelectorAll('.chip[data-dim="'+el.dataset.dim+'"]').forEach(c => setChip(c, c === el ? 'inc' : 'exc'));
    runSearch();
}
function chipReset(el) {
    document.querySelectorAll('.chip[data-dim="'+el.dataset.dim+'"]').forEach(c => setChip(c, ''));
    runSearch();
    return false;
}
const ALL_FIELDS = ['response', 'reasoning', 'jexpl', 'jraw'];
function activeFields() {
    const inc = new Set(), exc = new Set();
    document.querySelectorAll('.chip[data-dim="scope"]').forEach(c => {
        if (c.dataset.state === 'inc') inc.add(c.dataset.val);
        if (c.dataset.state === 'exc') exc.add(c.dataset.val);
    });
    return (inc.size ? ALL_FIELDS.filter(f => inc.has(f)) : ALL_FIELDS).filter(f => !exc.has(f));
}
async function runSearch() {
    const q = input.value.toLowerCase();
    if (!q) { hits = null; apply(); return; }
    const res = await (await fetch(BASE + '/search?q=' + encodeURIComponent(q) + '&fields=' + activeFields().join(','))).json();
    hits = new Set(res.hits);
    apply();
}
function apply() {
    const dims = {};
    document.querySelectorAll('.chip').forEach(c => {
        if (c.dataset.dim === 'scope') return;
        const d = (dims[c.dataset.dim] ??= {inc: new Set(), exc: new Set()});
        if (c.dataset.state === 'inc') d.inc.add(c.dataset.val);
        if (c.dataset.state === 'exc') d.exc.add(c.dataset.val);
    });
    const rows = Array.from(document.querySelectorAll('.row'));
    let shown = 0;
    rows.forEach(r => {
        const ok = passes(r, dims, Object.keys(dims)) && (!hits || hits.has(+r.dataset.idx));
        r.classList.toggle('hidden', !ok);
        if (ok) shown++;
    });
    document.getElementById('count').textContent = shown + '/' + rows.length;
    updateChips(rows, dims);
    highlight();
}
function passes(r, dims, use) {
    return use.every(dim => { const s = dims[dim], v = r.dataset[dim]; return !(s.inc.size && !s.inc.has(v)) && !s.exc.has(v); });
}
// Live chip counts (chip filters only, search ignored): behavior chips = matched/total for that behavior under the cond filter,
// cond chip = reasoning-on/total under the behavior filter, match chip = matched/total under behavior + cond filters.
function updateChips(rows, dims) {
    const frac = (rs, pred) => rs.filter(pred).length + '/' + rs.length;
    document.querySelectorAll('.chip[data-dim="behavior"]').forEach(c => {
        const rs = rows.filter(r => r.dataset.behavior === c.dataset.val && passes(r, dims, ['cond']));
        c.querySelector('b').textContent = frac(rs, r => r.dataset.match === '1');
    });
    document.querySelector('.chip[data-dim="cond"] b').textContent = frac(rows.filter(r => passes(r, dims, ['behavior'])), r => r.dataset.cond === '1');
    document.querySelector('.chip[data-dim="match"] b').textContent = frac(rows.filter(r => passes(r, dims, ['behavior', 'cond'])), r => r.dataset.match === '1');
}

// Search highlighting in the selected panel
function highlight() {
    const panel = document.querySelector('.panel.sel');
    if (!panel) return;
    panel.querySelectorAll('mark.hl').forEach(m => m.replaceWith(...m.childNodes));
    panel.normalize();
    const q = input.value.toLowerCase();
    if (!q) return;
    const walker = document.createTreeWalker(panel, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(node => {
        const text = node.textContent, lower = text.toLowerCase();
        let i = lower.indexOf(q);
        if (i < 0) return;
        const frag = document.createDocumentFragment();
        let last = 0;
        while (i >= 0) {
            frag.append(text.slice(last, i));
            const m = document.createElement('mark');
            m.className = 'hl';
            m.textContent = text.slice(i, i + q.length);
            frag.append(m);
            last = i + q.length;
            i = lower.indexOf(q, last);
        }
        frag.append(text.slice(last));
        node.replaceWith(frag);
    });
}
let qTimer = null;
input.addEventListener('input', function() { clearTimeout(qTimer); qTimer = setTimeout(runSearch, 200); });

function resort(selEl) {
    const left = document.querySelector('.left');
    const key = selEl.value;
    Array.from(left.children)
        .sort((a, b) => key === 'default' ? a.dataset.ord - b.dataset.ord : b.dataset[key] - a.dataset[key])
        .forEach(r => left.append(r));
}

document.querySelector('.divider').addEventListener('mousedown', function(e) {
    e.preventDefault();
    const panes = document.querySelector('.panes');
    function onMove(ev) {
        const rect = panes.getBoundingClientRect();
        document.querySelector('.left').style.flex = '0 0 ' + Math.min(Math.max((ev.clientX - rect.left) / rect.width * 100, 15), 85) + '%';
    }
    function onUp() { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
});
function filterRuns(q) {
    q = q.toLowerCase();
    document.querySelectorAll('.run').forEach(r => r.classList.toggle('hidden', !r.dataset.run.toLowerCase().includes(q)));
}
function toggleSide() {
    const hidden = document.body.classList.toggle('side-hidden');
    try { localStorage.setItem('sideHidden', hidden ? '1' : ''); } catch (e) {}
}
try { if (localStorage.getItem('sideHidden')) document.body.classList.add('side-hidden'); } catch (e) {}
apply();
"""

if __name__ == "__main__":
    print(f"viewer running at http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
