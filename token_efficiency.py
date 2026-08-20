#!/usr/bin/env python3
"""Token efficiency tracking for Grok runs and local Ollama fallbacks.

Records every task run to an append-only JSONL ledger and renders a dashboard
of live, cumulative, per-class and per-tier efficiency.

Library use:
    from token_efficiency import record_run, show_dashboard

    record_run("coding", input_t=12_000, output_t=800, tier="knife")
    show_dashboard()

CLI use:
    python dashboard.py
    python dashboard.py record coding --input 12000 --output 800 --tier knife
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "GROK_PRICING",
    "TASK_CLASSES",
    "TIERS",
    "Pricing",
    "classify_task",
    "estimate_cost",
    "format_dashboard",
    "hermes_home",
    "ledger_path",
    "load_ledger",
    "record_run",
    "show_dashboard",
    "summarize",
]


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens for a model."""

    input_per_m: float
    output_per_m: float
    cached_input_per_m: float


# grok-build-0.1 list pricing; cached input bills at a quarter of fresh input.
GROK_PRICING = Pricing(input_per_m=1.0, output_per_m=2.0, cached_input_per_m=0.25)

TASK_CLASSES = ("coding", "research", "tool-heavy", "short-chat", "long-horizon")

# Context tiers, smallest blast radius first. Knife is the default: read only
# what the task needs. Nuke means the whole tree went into context.
TIERS = ("knife", "scalpel", "nuke")

DEFAULT_PROFILE = "first-agent"

_CLASS_KEYWORDS: Dict[str, Sequence[str]] = {
    "coding": ("refactor", "bug", "implement", "patch", "test", "compile", "lint"),
    "research": ("investigate", "compare", "explain", "survey", "summar", "why", "research"),
    "tool-heavy": ("scrape", "pipeline", "migrate", "deploy", "batch", "crawl"),
    "long-horizon": ("multi-step", "roadmap", "end-to-end", "long-horizon", "epic"),
}


def hermes_home() -> Path:
    """Resolve the Hermes data directory for the current platform."""
    override = os.environ.get("HERMES_HOME")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "hermes"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "hermes"


def ledger_path(profile: Optional[str] = None) -> Path:
    """Path to the JSONL ledger, honouring TOKEN_LEDGER for one-off redirects."""
    override = os.environ.get("TOKEN_LEDGER")
    if override:
        return Path(override).expanduser()
    profile = profile or os.environ.get("HERMES_PROFILE") or DEFAULT_PROFILE
    return hermes_home() / "profiles" / profile / "token_ledger.jsonl"


def classify_task(task_desc: str) -> str:
    """Best-effort task class from a free-text description."""
    text = task_desc.lower()
    for task_class, keywords in _CLASS_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return task_class
    return "short-chat"


def estimate_cost(
    input_t: int,
    output_t: int,
    cached: int = 0,
    pricing: Pricing = GROK_PRICING,
) -> float:
    """Cost in USD of serving a run on Grok.

    ``cached`` is the subset of ``input_t`` that hit the prompt cache and so
    bills at the discounted rate.
    """
    cached = max(0, min(cached, input_t))
    fresh_input = input_t - cached
    cost = (
        fresh_input / 1e6 * pricing.input_per_m
        + cached / 1e6 * pricing.cached_input_per_m
        + output_t / 1e6 * pricing.output_per_m
    )
    return round(cost, 6)


def record_run(
    task_class: str,
    input_t: int,
    output_t: int,
    total_t: Optional[int] = None,
    cached: int = 0,
    success: bool = True,
    task_desc: str = "",
    tier: str = "knife",
    local_optimized: bool = False,
    local_model: str = "",
    turns: int = 1,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Append one run to the ledger and return the recorded entry.

    A run with ``local_optimized=True`` was served by a local model, so it costs
    nothing but still carries a Grok-equivalent price: that difference is the
    saving the dashboard reports.
    """
    if input_t < 0 or output_t < 0:
        raise ValueError("token counts must be non-negative")
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

    total_t = input_t + output_t if total_t is None else total_t
    grok_cost = estimate_cost(input_t, output_t, cached)

    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "class": task_class,
        "input": input_t,
        "output": output_t,
        "total": total_t,
        "cached": cached,
        "success": success,
        "task_desc": task_desc[:200],
        "tier": tier,
        "local_optimized": local_optimized,
        "local_model": local_model if local_optimized else "",
        "turns": turns,
        "billed_cost": 0.0 if local_optimized else grok_cost,
        "avoided_cost": grok_cost if local_optimized else 0.0,
        "est_cost": grok_cost,
        "i_o_ratio": round(input_t / max(1, output_t), 2),
        "cache_hit_rate": round(cached / max(1, input_t), 4),
    }

    path = ledger_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def load_ledger(profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read every well-formed run from the ledger, skipping corrupt lines."""
    path = ledger_path(profile)
    if not path.exists():
        return []

    runs: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping malformed ledger line {lineno}", file=sys.stderr)
                continue
            if isinstance(entry, dict):
                runs.append(entry)
    return runs


def _billed(run: Dict[str, Any]) -> float:
    if "billed_cost" in run:
        return float(run["billed_cost"])
    # Pre-cost-split entries recorded a single est_cost field.
    return 0.0 if run.get("local_optimized") else float(run.get("est_cost", 0.0))


def _avoided(run: Dict[str, Any]) -> float:
    if "avoided_cost" in run:
        return float(run["avoided_cost"])
    return float(run.get("est_cost", 0.0)) if run.get("local_optimized") else 0.0


def _bucket(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(int(r.get("total", 0)) for r in runs)
    return {
        "runs": len(runs),
        "tokens": total,
        "avg_tokens": round(total / len(runs), 1) if runs else 0.0,
        "billed_cost": round(sum(_billed(r) for r in runs), 6),
        "success_rate": (
            round(sum(1 for r in runs if r.get("success", True)) / len(runs), 4) if runs else 0.0
        ),
    }


def summarize(runs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a ledger into cumulative, per-class and per-tier metrics."""
    runs = list(runs)
    if not runs:
        return {
            "runs": 0,
            "tokens": 0,
            "input": 0,
            "output": 0,
            "cached": 0,
            "billed_cost": 0.0,
            "avoided_cost": 0.0,
            "cache_hit_rate": 0.0,
            "avg_tokens_per_task": 0.0,
            "i_o_ratio": 0.0,
            "success_rate": 0.0,
            "knife_share": 0.0,
            "local_share": 0.0,
            "by_class": {},
            "by_tier": {},
        }

    input_t = sum(int(r.get("input", 0)) for r in runs)
    output_t = sum(int(r.get("output", 0)) for r in runs)
    cached = sum(int(r.get("cached", 0)) for r in runs)
    total = sum(int(r.get("total", 0)) for r in runs)
    local = [r for r in runs if r.get("local_optimized")]
    knife = [r for r in runs if r.get("tier") == "knife"]

    return {
        "runs": len(runs),
        "tokens": total,
        "input": input_t,
        "output": output_t,
        "cached": cached,
        "billed_cost": round(sum(_billed(r) for r in runs), 6),
        "avoided_cost": round(sum(_avoided(r) for r in runs), 6),
        "cache_hit_rate": round(cached / max(1, input_t), 4),
        "avg_tokens_per_task": round(total / len(runs), 1),
        "i_o_ratio": round(input_t / max(1, output_t), 2),
        "success_rate": round(sum(1 for r in runs if r.get("success", True)) / len(runs), 4),
        "knife_share": round(len(knife) / len(runs), 4),
        "local_share": round(len(local) / len(runs), 4),
        "by_class": {
            name: _bucket([r for r in runs if r.get("class") == name])
            for name in sorted({str(r.get("class", "unknown")) for r in runs})
        },
        "by_tier": {
            name: _bucket([r for r in runs if r.get("tier") == name])
            for name in TIERS
            if any(r.get("tier") == name for r in runs)
        },
    }


def format_dashboard(runs: Sequence[Dict[str, Any]], recent: int = 10) -> str:
    """Render the dashboard as text."""
    stats = summarize(runs)
    width = 74
    lines = [
        "=" * width,
        "Token Efficiency — Grok + local Ollama",
        "=" * width,
        f"Ledger : {ledger_path()}",
        f"Time   : {datetime.now().isoformat(timespec='seconds')}",
        f"Runs   : {stats['runs']}",
    ]

    if not stats["runs"]:
        lines += [
            "",
            "No runs recorded yet.",
            "  python dashboard.py record coding --input 12000 --output 800 --tier knife",
        ]
        return "\n".join(lines)

    lines += [
        "",
        "Cumulative",
        f"  Tokens          : {stats['tokens']:,} "
        f"(in {stats['input']:,} / out {stats['output']:,})",
        f"  Avg per task    : {stats['avg_tokens_per_task']:,.1f}",
        f"  I:O ratio       : {stats['i_o_ratio']:.2f}",
        f"  Cache hit rate  : {stats['cache_hit_rate']:.1%}",
        f"  Success rate    : {stats['success_rate']:.1%}",
        f"  Billed (Grok)   : ${stats['billed_cost']:.4f}",
        f"  Avoided (local) : ${stats['avoided_cost']:.4f}",
        f"  Knife share     : {stats['knife_share']:.1%}",
        f"  Local share     : {stats['local_share']:.1%}",
        "",
        "By tier",
        f"  {'tier':<12} {'runs':>5} {'tokens':>12} {'avg':>10} {'billed':>10}",
    ]
    for tier, bucket in stats["by_tier"].items():
        lines.append(
            f"  {tier:<12} {bucket['runs']:>5} {bucket['tokens']:>12,} "
            f"{bucket['avg_tokens']:>10,.0f} {bucket['billed_cost']:>10.4f}"
        )

    lines += [
        "",
        "By class",
        f"  {'class':<12} {'runs':>5} {'tokens':>12} {'avg':>10} {'ok':>10}",
    ]
    for name, bucket in stats["by_class"].items():
        lines.append(
            f"  {name:<12} {bucket['runs']:>5} {bucket['tokens']:>12,} "
            f"{bucket['avg_tokens']:>10,.0f} {bucket['success_rate']:>9.0%}"
        )

    lines += ["", f"Recent runs (last {min(recent, len(runs))})"]
    for run in list(runs)[-recent:]:
        served = (run.get("local_model") or "ollama") if run.get("local_optimized") else "grok"
        flag = " " if run.get("success", True) else "!"
        lines.append(
            f" {flag}{str(run.get('timestamp', ''))[:19]} "
            f"{str(run.get('class', '?')):<11} {str(run.get('tier', '?')):<7} "
            f"{served:<16} in={int(run.get('input', 0)):>8,} "
            f"out={int(run.get('output', 0)):>7,}"
        )

    lines += ["", "Knife tier and local models first — escalate only when a run fails."]
    return "\n".join(lines)


def show_dashboard(recent: int = 10, profile: Optional[str] = None) -> Dict[str, Any]:
    """Print the dashboard and return the summary statistics."""
    runs = load_ledger(profile)
    print(format_dashboard(runs, recent=recent))
    return summarize(runs)


if __name__ == "__main__":
    show_dashboard()
