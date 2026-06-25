import json
import sys
import uuid
from pathlib import Path

import gradio as gr
import requests

# Allow importing from project root (eval package)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

API_BASE = "http://localhost:8000"

# Agents that form the "choice" layer — orchestrator picks one (or more) of these
_MIDDLE_AGENTS = [
    ("vision",         "Vision"),
    ("clip_retrieval", "CLIP"),
    ("retrieval",      "Retrieval"),
    ("parts",          "Parts"),
]


def _node(key: str, label: str, active: set, compact: bool = False, badge: str = "") -> str:
    if key in active:
        style = "background:#d1fae5;color:#065f46;border:1px solid #a7f3d0"
    else:
        style = "background:#f3f4f6;color:#9ca3af;border:1px solid #e5e7eb"
    pad  = "2px 8px"  if compact else "3px 10px"
    size = "11px"     if compact else "12px"
    badge_html = (
        f'<span style="margin-left:4px;font-size:9px;opacity:0.75">{badge}</span>'
        if badge else ""
    )
    return (
        f'<span style="padding:{pad};border-radius:12px;font-size:{size};'
        f'white-space:nowrap;display:inline-block;{style}">{label}{badge_html}</span>'
    )


def _pipeline_html(active: set, compact: bool = False, clip_score: float | None = None) -> str:
    gap   = "3px" if compact else "4px"
    arrow = f'<span style="color:#d1d5db;margin:0 {gap}">→</span>'

    def _middle_node(k: str, l: str) -> str:
        badge = f"{clip_score:.0%}" if k == "clip_retrieval" and clip_score is not None else ""
        return f'<div style="line-height:1">{_node(k, l, active, compact, badge)}</div>'

    # Middle group: vertical column inside a bordered box
    middle_items = "".join(_middle_node(k, l) for k, l in _MIDDLE_AGENTS)
    group_border = "#a7f3d0" if any(k in active for k, _ in _MIDDLE_AGENTS) else "#e5e7eb"
    middle = (
        f'<div style="display:inline-flex;flex-direction:column;gap:{gap};'
        f'border:1px solid {group_border};border-radius:8px;padding:4px 6px;vertical-align:middle">'
        + middle_items + "</div>"
    )

    parts = [
        _node("orchestrator", "Orchestrator", active, compact),
        arrow,
        middle,
        arrow,
        _node("stock",       "Stock",       active, compact),
        arrow,
        _node("synthesizer", "Synthesizer", active, compact),
    ]
    align = "center" if not compact else "center"
    return (
        f'<div style="display:flex;align-items:{align};gap:{gap};flex-wrap:nowrap">'
        + "".join(parts) + "</div>"
    )


def _render_debug(meta: dict) -> str:
    if not meta:
        return "<p style='color:#9ca3af;font-size:13px;margin:0'>No data yet — send a message.</p>"

    intent = meta.get("intent") or "—"
    latency = meta.get("latency_ms")
    ttft = meta.get("ttft_ms")
    agents = ", ".join(meta.get("agents_used", [])) or "—"

    rows = [
        ("Intent", intent),
        ("Agents", agents),
        ("Latency", f"{latency}&thinsp;ms" if latency is not None else "—"),
        ("TTFT", f"{ttft}&thinsp;ms" if ttft is not None else "—"),
    ]
    rows_html = "".join(
        f'<tr><td style="color:#6b7280;padding:2px 12px 2px 0;white-space:nowrap">{k}</td>'
        f'<td style="color:#111827">{v}</td></tr>'
        for k, v in rows
    )
    return (
        '<div style="font-family:sans-serif;font-size:13px;padding:12px 14px;'
        'background:#fafafa;border-radius:8px;border:1px solid #e5e7eb">'
        f'<table style="border-collapse:collapse">{rows_html}</table>'
        "</div>"
    )


def _render_chat_debug(meta: dict) -> str:
    active = set(meta.get("agents_used", []))
    intent = meta.get("intent") or "—"
    latency = meta.get("latency_ms")
    ttft = meta.get("ttft_ms")
    clip_score = meta.get("clip_score")

    summary_parts = [f"intent: {intent}"]
    if latency is not None:
        summary_parts.append(f"{latency} ms")
    if ttft is not None:
        summary_parts.append(f"TTFT {ttft} ms")
    if clip_score is not None:
        summary_parts.append(f"CLIP {clip_score:.0%}")
    summary = " · ".join(summary_parts)

    pipeline = _pipeline_html(active, compact=True, clip_score=clip_score)

    return (
        '<details style="margin-top:8px;font-family:sans-serif">'
        f'<summary style="cursor:pointer;font-size:11px;color:#9ca3af;user-select:none">'
        f"🔍 {summary}</summary>"
        '<div style="margin-top:8px;padding:10px 12px;background:#fafafa;'
        'border-radius:8px;border:1px solid #e5e7eb">'
        '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px">'
        f"{pipeline}</div>"
        "</div></details>"
    )


def respond(message: dict, history: list, session_id: str, show_in_chat: bool = False):
    text = (message.get("text") or "").strip()
    files = message.get("files") or []

    if not text and not files:
        yield history, session_id, gr.update(), gr.update()
        return

    parts = []
    if files:
        parts.append({"path": files[0]})
    if text:
        parts.append(text)
    user_content = parts[0] if len(parts) == 1 else parts

    history = history + [{"role": "user", "content": user_content}]
    history = history + [{"role": "assistant", "content": "Processing your request..."}]
    yield history, session_id, gr.update(value=None, interactive=False), gr.update()

    data = {"session_id": session_id, "message": text or "(image)"}
    files_payload = {"image": (Path(files[0]).name, open(files[0], "rb"))} if files else None

    last_meta: dict = {}
    try:
        with requests.post(
            f"{API_BASE}/query/stream",
            data=data,
            files=files_payload,
            stream=True,
            timeout=120,
        ) as resp:
            first_token = True
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:].decode("utf-8")
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    chunk = payload

                if isinstance(chunk, dict) and chunk.get("type") == "metadata":
                    last_meta = chunk
                    continue

                if isinstance(chunk, dict) and "error" in chunk:
                    history[-1]["content"] = f"Error: {chunk['error']}"
                    yield history, session_id, gr.update(interactive=True), gr.update()
                    return

                if first_token:
                    history[-1]["content"] = chunk
                    first_token = False
                else:
                    history[-1]["content"] += chunk
                yield history, session_id, gr.update(interactive=False), gr.update()

    except Exception as exc:
        history[-1]["content"] = f"Error contacting support service: {exc}"

    if last_meta and show_in_chat:
        history[-1]["content"] += _render_chat_debug(last_meta)

    yield history, session_id, gr.update(interactive=True), _render_debug(last_meta)


def new_chat():
    return [], str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Eval runner UI helpers
# ---------------------------------------------------------------------------

_CHECKER_ORDER = ["faithfulness", "hallucination", "refusal", "injection", "pii"]


def _render_eval_table(rows: list[dict]) -> str:
    if not rows:
        return ""

    checkers_seen: set[str] = set()
    for r in rows:
        checkers_seen.update(r.get("checkers", {}).keys())
    cols = [c for c in _CHECKER_ORDER if c in checkers_seen]

    def cell_html(res: dict) -> str:
        passed = res["passed"]
        score  = res["score"]
        bg     = "#d1fae5" if passed else "#fee2e2"
        color  = "#065f46" if passed else "#991b1b"
        label  = "PASS"    if passed else "FAIL"
        detail = res.get("detail", "")
        detail_escaped = detail.replace("<", "&lt;")[:80]
        # Show numeric score only for partial results (0 < score < 1)
        score_str = f"&nbsp;{score:.2f}" if 0 < score < 1 else ""
        return (
            f'<td style="background:{bg};color:{color};font-size:11px;font-weight:600;'
            f'padding:5px 8px;text-align:center;vertical-align:top">'
            f'{label}{score_str}'
            f'<div style="font-weight:400;font-size:9px;color:{color};opacity:0.8;'
            f'margin-top:2px;white-space:normal;max-width:120px">{detail_escaped}</div>'
            f'</td>'
        )

    def _pill(text: str, bg: str, prefix: str = "") -> str:
        return (
            f'<span style="padding:1px 5px;border-radius:10px;font-size:10px;'
            f'background:{bg};color:#111;white-space:nowrap;display:inline-block">'
            f'{prefix}{text}</span>'
        )

    _sep = '<span style="font-size:10px;margin:0 1px"> </span>'
    _lbl = lambda t: f'<div style="font-size:9px;font-weight:600;color:#9ca3af;margin-bottom:3px">{t}</div>'  # noqa: E731

    def agents_td(expected: list[str], actual: list[str]) -> str:
        actual_set   = set(actual)
        expected_set = set(expected)
        exp_pills = _sep.join(
            _pill(a, "#d1fae5") if a in actual_set else _pill(a, "#fee2e2")
            for a in expected
        ) or "—"
        act_pills = _sep.join(
            _pill(a, "#d1fae5") if a in expected_set else _pill(a, "#fef3c7")
            for a in actual
        ) or "—"
        inner = (
            f'{_lbl("EXPECTED")}'
            f'<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:5px">{exp_pills}</div>'
            f'{_lbl("ACTUAL")}'
            f'<div style="display:flex;flex-wrap:wrap;gap:2px">{act_pills}</div>'
        )
        return (
            f'<td style="padding:5px 8px;font-size:11px;vertical-align:top;min-width:160px">'
            f'<div style="display:flex;flex-direction:column">{inner}</div>'
            f'</td>'
        )

    def facts_td(expected: list[str], response: str) -> str:
        if not expected:
            pills = '<span style="color:#9ca3af;font-size:10px">—</span>'
        else:
            resp_lower = response.lower()
            pills = ' '.join(
                _pill(f, "#d1fae5") if f.lower() in resp_lower else _pill(f, "#fee2e2")
                for f in expected
            )
        return (
            f'<td style="padding:5px 8px;font-size:11px;vertical-align:top;min-width:120px">'
            f'<div style="display:flex;flex-wrap:wrap;gap:3px">{pills}</div>'
            f'</td>'
        )

    th = lambda t: f'<th style="padding:5px 8px;font-size:11px;color:#6b7280;font-weight:600;text-align:left;white-space:nowrap">{t}</th>'  # noqa: E731

    header = "".join(th(h) for h in ["ID", "Type"] + [c.capitalize() for c in cols] + ["Agents", "Facts", "ms", "Query / Response"])

    body = ""
    for r in rows:
        id_td   = f'<td style="padding:5px 8px;font-weight:600;font-size:11px;white-space:nowrap;vertical-align:top">{r["id"]}</td>'
        type_td = f'<td style="padding:5px 8px;font-size:11px;white-space:nowrap;vertical-align:top">{r["type"]}</td>'

        checker_tds = ""
        for c in cols:
            if c in r.get("checkers", {}):
                checker_tds += cell_html(r["checkers"][c])
            else:
                checker_tds += '<td style="padding:5px 8px;color:#d1d5db;text-align:center;vertical-align:top">—</td>'

        ag_td    = agents_td(r.get("expected_agents", []), r.get("actual_agents", []))
        facts_td_html = facts_td(r.get("expected_facts", []), r.get("response", ""))

        query    = r.get("query", "").replace("<", "&lt;")
        response = (r.get("response") or r.get("error", "")).replace("<", "&lt;").replace("\n", " ")[:200]
        qr_td = (
            f'<td style="padding:5px 8px;font-size:11px;max-width:360px;vertical-align:top">'
            f'<span>{query}</span>'
            f'<hr style="margin:3px 0;border:none;border-top:1px solid currentColor;opacity:0.15">'
            f'<span>{response}</span>'
            f'</td>'
        )
        lat = r.get("latency_ms")
        lat_td = (
            f'<td style="padding:5px 8px;font-size:11px;white-space:nowrap;vertical-align:top;text-align:right">'
            f'{int(lat):,}</td>' if lat is not None else
            '<td style="padding:5px 8px;color:#d1d5db;text-align:center;vertical-align:top">—</td>'
        )
        body += f"<tr style='border-top:1px solid #f3f4f6'>{id_td}{type_td}{checker_tds}{ag_td}{facts_td_html}{lat_td}{qr_td}</tr>"

    return (
        '<div style="overflow-x:auto;border-radius:8px;border:1px solid #e5e7eb">'
        '<table style="border-collapse:collapse;width:100%;font-family:sans-serif">'
        f'<thead style="background:#f9fafb"><tr>{header}</tr></thead>'
        f'<tbody>{body}</tbody>'
        '</table></div>'
    )


def _render_summary_table(rows: list[dict]) -> str:
    from collections import defaultdict

    by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    checker_totals: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})

    for r in rows:
        t = r["type"]
        all_passed = bool(r["checkers"]) and all(v["passed"] for v in r["checkers"].values())
        by_type[t]["total"] += 1
        if all_passed:
            by_type[t]["passed"] += 1
        for name, res in r["checkers"].items():
            checker_totals[name]["total"] += 1
            if res["passed"]:
                checker_totals[name]["passed"] += 1

    def pct_color(p: int) -> str:
        return "#10b981" if p == 100 else ("#f59e0b" if p >= 75 else "#ef4444")

    def th(t: str) -> str:
        return f'<th style="padding:4px 10px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;border-bottom:1px solid #e5e7eb">{t}</th>'

    def td(t: str, align: str = "left", bold: bool = False) -> str:
        fw = "600" if bold else "400"
        return f'<td style="padding:4px 10px;font-size:12px;text-align:{align};font-weight:{fw}">{t}</td>'

    def pct_td(n: int, total: int) -> str:
        p = round(n / total * 100) if total else 0
        return f'<td style="padding:4px 10px;font-size:12px;text-align:right;font-weight:600;color:{pct_color(p)}">{p}%</td>'

    # Category table
    type_order = ["normal", "out_of_scope", "injection", "pii_probe"]
    all_types = type_order + [t for t in by_type if t not in type_order]
    cat_rows = ""
    for t in all_types:
        if t not in by_type:
            continue
        d = by_type[t]
        cat_rows += f"<tr>{td(t, bold=True)}{td(str(d['total']), 'right')}{td(str(d['passed']), 'right')}{pct_td(d['passed'], d['total'])}</tr>"
    total_all = sum(d["total"] for d in by_type.values())
    passed_all = sum(d["passed"] for d in by_type.values())
    cat_rows += (
        f'<tr style="border-top:2px solid #e5e7eb">'
        f'{td("TOTAL", bold=True)}{td(str(total_all), "right", bold=True)}'
        f'{td(str(passed_all), "right", bold=True)}{pct_td(passed_all, total_all)}</tr>'
    )
    cat_table = (
        f'<table style="border-collapse:collapse;margin-right:24px">'
        f'<thead><tr>{th("Category")}{th("Cases")}{th("Pass")}{th("%")}</tr></thead>'
        f'<tbody>{cat_rows}</tbody></table>'
    )

    # Checker table
    checker_order = ["faithfulness", "hallucination", "refusal", "injection", "pii"]
    all_checkers = checker_order + [c for c in checker_totals if c not in checker_order]
    chk_rows = ""
    for c in all_checkers:
        if c not in checker_totals:
            continue
        d = checker_totals[c]
        chk_rows += f"<tr>{td(c, bold=True)}{td(str(d['total']), 'right')}{td(str(d['passed']), 'right')}{pct_td(d['passed'], d['total'])}</tr>"
    chk_table = (
        f'<table style="border-collapse:collapse">'
        f'<thead><tr>{th("Checker")}{th("Cases")}{th("Pass")}{th("%")}</tr></thead>'
        f'<tbody>{chk_rows}</tbody></table>'
    )

    return (
        f'<div style="display:flex;gap:0;margin-bottom:16px;flex-wrap:wrap">'
        f'{cat_table}{chk_table}'
        f'</div>'
    )


def run_eval_ui(ids_filter: str):
    """Generator: yields HTML string after each case completes."""
    from eval.run_eval import run_cases  # lazy import

    rows: list[dict] = []
    total = done = 0

    yield '<p style="font-size:13px;color:#6b7280;margin:0 0 8px">Starting…</p>'

    for case, response, checker_results, meta in run_cases(API_BASE, ids_filter):
        total += 1
        all_passed = bool(checker_results) and all(r["passed"] for r in checker_results.values())
        if all_passed:
            done += 1

        rows.append({
            "id": case["id"],
            "type": case["type"],
            "query": case["query"],
            "response": response,
            "checkers": checker_results,
            "expected_facts": case.get("expected_facts", []),
            "expected_agents": case.get("expected_agents", []),
            "actual_agents": meta.get("agents_used", []),
            "latency_ms": meta.get("latency_ms"),
        })

        status_color = "#6b7280"
        progress = (
            f'<p style="font-size:13px;color:{status_color};margin:0 0 8px">'
            f'Running… {total} done, {done} passed</p>'
        )
        yield progress + _render_eval_table(rows)

    if not rows:
        yield "<p style='font-size:13px;color:#9ca3af;margin:0'>No cases matched.</p>"
        return

    result_color = "#10b981" if done == total else "#ef4444"
    summary = (
        f'<p style="font-size:13px;font-weight:600;color:{result_color};margin:0 0 16px">'
        f'Done — {done}/{total} cases fully passed</p>'
    )
    yield summary + _render_summary_table(rows) + _render_eval_table(rows)


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
footer { display: none !important; }
body::-webkit-scrollbar { display: none; }
body { -ms-overflow-style: none; scrollbar-width: none; }
.multimodal-textbox textarea { overflow-y: hidden !important; resize: none !important; }
*, *::before, *::after {
    font-family: 'Inter', 'Segoe UI Variable', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#eval-filter-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 8px !important;
}
#eval-filter-row > div:first-child {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    padding: 0 !important;
}
#eval-filter-row > div:first-child * {
    width: auto !important;
    white-space: nowrap !important;
}
#eval-filter-row > div:last-child {
    flex: 0 0 auto !important;
    width: auto !important;
    align-self: center !important;
    padding: 0 !important;
}
#eval-run-btn button {
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
}
"""

# Available themes (uncomment one):
# _THEME = gr.themes.Soft()
# _THEME = gr.themes.Ocean()
# _THEME = gr.themes.Monochrome()
# _THEME = gr.themes.Glass()
_THEME = gr.themes.Soft()

with gr.Blocks(title="Washing Machine Support", css=_CSS, fill_height=True, theme=_THEME) as demo:
    session_id = gr.State(lambda: str(uuid.uuid4()))

    with gr.Tabs():
        # ── Chat tab ──────────────────────────────────────────────────────────
        with gr.Tab("Chat"):
            with gr.Column(scale=1):
                gr.Markdown("## Washing Machine Support")

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    layout="bubble",
                    scale=1,
                    show_label=False,
                    avatar_images=(
                        None,
                        str(Path(__file__).parent / "assets" / "bot_avatar2.png"),
                    ),
                    placeholder="Ask me anything about your washing machine — repairs, parts, error codes.",
                )

                inp = gr.MultimodalTextbox(
                    file_types=["image"],
                    placeholder="Ask about your washing machine or drop a part photo...",
                    show_label=False,
                    submit_btn=True,
                )

                new_btn = gr.Button("New conversation", size="sm", variant="secondary")

        # ── Debug / Eval tab ──────────────────────────────────────────────────
        with gr.Tab("Debug / Eval"):
            with gr.Accordion("Chat options", open=True):
                show_graph_cb = gr.Checkbox(
                    label="Show pipeline info in chat",
                    value=False,
                )

            with gr.Accordion("Last turn", open=False):
                debug_html = gr.HTML(
                    value="<p style='color:#9ca3af;font-size:13px;margin:0'>"
                          "No data yet — send a message.</p>"
                )

            with gr.Accordion("Eval runner", open=False):
                with gr.Row(elem_id="eval-filter-row"):
                    gr.HTML(
                        '<span style="font-size:13px;white-space:nowrap;'
                        'color:var(--body-text-color,#374151)">Filter by ID</span>',
                        scale=0,
                        min_width=90,
                    )
                    eval_ids = gr.Textbox(
                        placeholder="n001,o002 — empty = all 20 cases",
                        show_label=False,
                        scale=3,
                    )
                    eval_btn = gr.Button(
                        "Run eval", variant="primary",
                        scale=0, min_width=110,
                        elem_id="eval-run-btn",
                    )
                eval_out = gr.HTML(value="")

    inp.submit(
        respond,
        [inp, chatbot, session_id, show_graph_cb],
        [chatbot, session_id, inp, debug_html],
    )
    new_btn.click(new_chat, [], [chatbot, session_id])
    eval_btn.click(run_eval_ui, inputs=[eval_ids], outputs=[eval_out])

if __name__ == "__main__":
    demo.launch(server_port=7860)
