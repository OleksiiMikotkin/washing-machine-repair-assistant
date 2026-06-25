"""
Eval runner.

Programmatic API:
    from eval.run_eval import run_cases
    for case, response, checker_results in run_cases(api_base, ids_filter):
        ...

CLI:
    python eval/run_eval.py [--api http://localhost:8000] [--ids n001,n005]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator

import requests

EVAL_DIR = Path(__file__).parent
GOLDEN_SET = EVAL_DIR / "golden_set.json"

# Make project root importable when run as a script
if str(EVAL_DIR.parent) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR.parent))

from eval.checkers import run_checkers  # noqa: E402


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _call_api(api_base: str, session_id: str, query: str) -> tuple[str, dict]:
    """Returns (response_text, meta_dict). Uses non-streaming /query for reliability."""
    resp = requests.post(
        f"{api_base}/query",
        data={"session_id": session_id, "message": query},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    meta = {
        "agents_used": data.get("agents_used", []),
        "latency_ms": data.get("latency_ms"),
        "context": data.get("context"),
    }
    return data.get("final_answer", ""), meta


# ---------------------------------------------------------------------------
# Core generator — usable from UI and CLI
# ---------------------------------------------------------------------------

def run_cases(
    api_base: str,
    ids_filter: str = "",
) -> Iterator[tuple[dict, str, dict, dict]]:
    """
    Yields (case, response_text, checker_results, meta) for each case.
    checker_results is empty dict on API error; response_text contains the error message.
    """
    cases: list[dict] = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))

    if ids_filter:
        wanted = {x.strip() for x in ids_filter.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]

    for case in cases:
        sid = str(uuid.uuid4())
        try:
            response, meta = _call_api(api_base, sid, case["query"])
            checker_results = run_checkers(case, response, meta)
        except Exception as exc:
            response = f"ERROR: {exc}"
            meta = {}
            checker_results = {}

        yield case, response, checker_results, meta


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

_COL_W = 12


def _checker_header(checker_names: list[str]) -> str:
    cols = "  ".join(f"{n:<{_COL_W}}" for n in checker_names)
    return f"{'ID':<8} {'Type':<14} {cols} Response preview"


def _row(case_id: str, case_type: str, results: dict[str, dict], preview: str) -> str:
    def cell(r: dict) -> str:
        return f"{'PASS' if r['passed'] else 'FAIL'}({r['score']:.2f})"

    cols = "  ".join(f"{cell(results[n]):<{_COL_W}}" for n in sorted(results))
    return f"{case_id:<8} {case_type:<14} {cols} {preview[:60].replace(chr(10), ' ')}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--ids", default="", help="Comma-separated case IDs (default: all)")
    args = parser.parse_args()

    print(f"\nRunning eval against {args.api}\n")

    all_results: list[dict] = []
    checker_names_seen: set[str] = set()

    for case, response, checker_results, meta in run_cases(args.api, args.ids):
        passed_all = all(r["passed"] for r in checker_results.values()) if checker_results else False
        status = "OK  " if passed_all else "FAIL"
        latency = meta.get("latency_ms", "?")
        print(f"  [{case['id']}] {status}  ({latency} ms)  {case['query'][:50]!r}")

        checker_names_seen.update(checker_results)
        all_results.append({
            "id": case["id"],
            "type": case["type"],
            "query": case["query"],
            "response": response,
            "meta": meta,
            "checkers": checker_results,
        })

    # Summary table
    checker_names = sorted(checker_names_seen)
    print("\n" + "=" * 100)
    print(_checker_header(checker_names))
    print("-" * 100)
    for r in all_results:
        if not r["checkers"]:
            print(f"{r['id']:<8} {r['type']:<14} ERROR: {r['response'][:60]}")
        else:
            print(_row(r["id"], r["type"], r["checkers"], r.get("response", "")))
    print("=" * 100)

    total = len(all_results)
    passed = sum(
        1 for r in all_results
        if r["checkers"] and all(v["passed"] for v in r["checkers"].values())
    )
    print(f"\nResult: {passed}/{total} cases fully passed\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"results_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved → {out_path}\n")


if __name__ == "__main__":
    main()
