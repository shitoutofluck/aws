#!/usr/bin/env python3
"""CLI for the token efficiency ledger.

    python dashboard.py                                  # live dashboard
    python dashboard.py record coding --input 12000 --output 800
    python dashboard.py record coding --from-response run.json   # real API counts
    python dashboard.py summary --json                   # machine-readable
    python dashboard.py where                            # ledger location
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from token_efficiency import (
    TIERS,
    classify_task,
    format_dashboard,
    ledger_path,
    load_ledger,
    record_from_response,
    record_run,
    summarize,
)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dashboard.py",
        description="Token efficiency dashboard for Grok runs and local Ollama fallbacks.",
    )
    parser.add_argument("--profile", help="Hermes profile name (default: first-agent)")
    parser.set_defaults(recent=10)
    sub = parser.add_subparsers(dest="command")

    show = sub.add_parser("show", help="print the dashboard (default)")
    show.add_argument("-n", "--recent", type=int, default=10, help="recent runs to list")

    rec = sub.add_parser("record", help="append a run to the ledger")
    rec.add_argument("task_class", nargs="?", help="coding, research, tool-heavy, ...")
    rec.add_argument("--from-response", dest="from_response", metavar="FILE",
                     help="read real token counts from a saved API response JSON ('-' for stdin)")
    rec.add_argument("--input", "-i", type=int, dest="input_t")
    rec.add_argument("--output", "-o", type=int, dest="output_t")
    rec.add_argument("--total", type=int, default=None, help="defaults to input + output")
    rec.add_argument("--cached", type=int, default=0, help="cached input tokens")
    rec.add_argument("--turns", type=int, default=1)
    rec.add_argument("--tier", choices=TIERS, default="knife")
    rec.add_argument("--desc", default="", help="task description (also used to classify)")
    rec.add_argument("--failed", action="store_true", help="mark the run unsuccessful")
    rec.add_argument("--local", dest="local_model", nargs="?", const="ollama", default=None,
                     help="run was served locally, optionally by MODEL")

    summary = sub.add_parser("summary", help="print aggregate statistics")
    summary.add_argument("--json", action="store_true", dest="as_json")

    sub.add_parser("where", help="print the ledger path")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "show"

    if command == "where":
        print(ledger_path(args.profile))
        return 0

    if command == "record":
        task_class = args.task_class or classify_task(args.desc)

        if args.from_response:
            raw = sys.stdin.read() if args.from_response == "-" else _read(args.from_response)
            entry = record_from_response(
                json.loads(raw),
                task_class=task_class,
                task_desc=args.desc,
                tier=args.tier,
                success=not args.failed,
                turns=args.turns,
                profile=args.profile,
            )
            print(json.dumps(entry, indent=2))
            return 0

        if args.input_t is None or args.output_t is None:
            print("error: --input and --output are required without --from-response",
                  file=sys.stderr)
            return 2

        entry = record_run(
            task_class,
            input_t=args.input_t,
            output_t=args.output_t,
            total_t=args.total,
            cached=args.cached,
            success=not args.failed,
            task_desc=args.desc,
            tier=args.tier,
            local_optimized=args.local_model is not None,
            local_model=args.local_model or "",
            turns=args.turns,
            profile=args.profile,
        )
        print(json.dumps(entry, indent=2))
        return 0

    runs = load_ledger(args.profile)

    if command == "summary":
        stats = summarize(runs)
        if args.as_json:
            print(json.dumps(stats, indent=2))
        else:
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for name, bucket in value.items():
                        print(f"  {name}: {bucket}")
                else:
                    print(f"{key}: {value}")
        return 0

    print(format_dashboard(runs, recent=args.recent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
