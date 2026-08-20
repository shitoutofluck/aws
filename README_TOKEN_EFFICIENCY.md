# Token Efficiency Dashboard

Persistent token accounting for Grok runs and local Ollama fallbacks. Every task run is
appended to a JSONL ledger; the dashboard turns that ledger into cumulative, per-class and
per-tier efficiency numbers so you can see whether you are actually driving cheaply.

## Files

| File | Purpose |
| --- | --- |
| `token_efficiency.py` | Library: recording, loading, cost estimation, aggregation, rendering |
| `dashboard.py` | CLI entry point (`show`, `record`, `summary`, `where`) |
| `skills/token-efficiency/SKILL.md` | Hermes skill definition |
| `test_token_efficiency.py` | Test suite |

No third-party dependencies; the standard library is enough. `pytest` is only needed to run
the tests.

## Quick start

```bash
python dashboard.py                                              # dashboard
python dashboard.py record coding --input 12000 --output 800 --tier knife
python dashboard.py record --input 4000 --output 300 --local llama3.2:1b --desc "summarize diff"
python dashboard.py summary --json
python dashboard.py where
```

Omitting the task class infers it from `--desc`. Add `--cached N` for prompt-cache hits and
`--failed` for runs that did not land.

From Python:

```python
from token_efficiency import record_run, show_dashboard, summarize, load_ledger

record_run("coding", input_t=12_000, output_t=800, cached=6_000, tier="knife")
stats = summarize(load_ledger())
show_dashboard()
```

## Context tiers

Tiers describe how much context a run pulled in, and are the main lever the dashboard is
built to watch:

- `knife` — read only what the task needs. The default, and what most runs should be.
- `scalpel` — targeted expansion into neighbouring files.
- `nuke` — the whole tree went into context. Cheap to reach for, expensive to repeat.

The dashboard reports `knife_share` so tier drift is visible before the bill is.

## Cost model

Grok `grok-build-0.1` list pricing: $1/M input, $2/M output, cached input at $0.25/M. Cached
tokens are treated as a subset of input, so they bill at the discounted rate rather than
being counted twice.

A run recorded with `--local MODEL` was served by Ollama. Its `billed_cost` is zero and
`avoided_cost` holds what the same run would have cost on Grok, so the "savings" figure is
Grok spend actually avoided instead of a character-count proxy. Both numbers appear in the
dashboard and in `summary --json`.

## Ledger location

`$HERMES_HOME/profiles/$HERMES_PROFILE/token_ledger.jsonl`

- `HERMES_HOME` defaults to `%LOCALAPPDATA%\hermes` on Windows, and `$XDG_DATA_HOME/hermes`
  (falling back to `~/.local/share/hermes`) everywhere else.
- `HERMES_PROFILE` defaults to `first-agent`; `--profile` overrides it per invocation.
- `TOKEN_LEDGER` overrides the full path, which is what the tests use.

The ledger is append-only and each line is an independent JSON object, so a truncated or
corrupt write costs one run rather than the whole history — malformed lines are skipped with
a warning on stderr.

## Recorded fields

```json
{
  "timestamp": "2026-08-20T04:54:37Z",
  "class": "coding",
  "input": 12000,
  "output": 800,
  "total": 12800,
  "cached": 6000,
  "success": true,
  "task_desc": "fix parser bug",
  "tier": "knife",
  "local_optimized": false,
  "local_model": "",
  "turns": 1,
  "billed_cost": 0.0085,
  "avoided_cost": 0.0,
  "est_cost": 0.0085,
  "i_o_ratio": 15.0,
  "cache_hit_rate": 0.5
}
```

`est_cost` is retained for compatibility with older ledgers, which recorded a single cost
field; `summarize` reads those entries correctly.

## Tests

```bash
python -m pytest test_token_efficiency.py
```
