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
python dashboard.py record coding --from-response run.json       # real API counts
python dashboard.py record coding --input 12000 --output 800 --tier knife
python dashboard.py record --input 4000 --output 300 --local llama3.2:1b --desc "summarize diff"
python dashboard.py summary --json
python dashboard.py where
```

Omitting the task class infers it from `--desc`. Add `--cached N` for prompt-cache hits and
`--failed` for runs that did not land. `--from-response -` reads the JSON from stdin.

From Python:

```python
from token_efficiency import record_run, record_from_response, show_dashboard

response = client.chat.completions.create(model="grok-4.6", messages=messages)
record_from_response(response, "coding", tier="knife", task_desc="fix parser")

record_run("coding", input_t=12_000, output_t=800, cached=6_000)  # manual estimate
show_dashboard()
```

## Where the numbers come from

There are two paths, and the ledger records which one each run used in `cost_source`:

- **`record_from_response(response)` — actual usage.** Reads the counts xAI reports and, when
  present, the exact price xAI charged. This is the accurate path.
- **`record_run(...)` — self-reported.** Records whatever numbers you hand it and estimates
  cost from list pricing. Use it for local Ollama runs and for anything not made through the
  API.

`extract_usage` handles both xAI response shapes, either as SDK objects or parsed JSON:

| | Chat Completions | Responses API |
| --- | --- | --- |
| Input | `usage.prompt_tokens` | `usage.input_tokens` |
| Output | `usage.completion_tokens` | `usage.output_tokens` |
| Cached | `usage.prompt_tokens_details.cached_tokens` | `usage.input_tokens_details.cached_tokens` |
| Reasoning | `usage.completion_tokens_details.reasoning_tokens` | `usage.output_tokens_details.reasoning_tokens` |
| Cost | `usage.cost_in_usd_ticks` | `cost_in_usd_ticks` / `cost_in_nano_usd` |

Reasoning tokens are already inside the output count and bill at the output rate; they are
tracked separately so that invisible spend is visible. The dashboard labels the cost line
`provider-reported` when every Grok run came from the API, and otherwise says how many did.

## Context tiers

Tiers describe how much context a run pulled in, and are the main lever the dashboard is
built to watch:

- `knife` — read only what the task needs. The default, and what most runs should be.
- `scalpel` — targeted expansion into neighbouring files.
- `nuke` — the whole tree went into context. Cheap to reach for, expensive to repeat.

The dashboard reports `knife_share` so tier drift is visible before the bill is.

## Cost model

When the API reports a cost it is used verbatim. xAI returns `cost_in_usd_ticks`, an integer
where 1e8 ticks is one cent and 1e10 ticks is one dollar; the Responses API may instead
return `cost_in_nano_usd`. Either way the recorded figure is what you were actually charged,
so it stays correct across model and price changes.

Only when no cost is reported does the estimator run, using Grok `grok-build-0.1` list
pricing: $1/M input, $2/M output, cached input at $0.25/M. Cached tokens are treated as a
subset of input, so they bill at the discounted rate rather than being counted twice.

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
  "reasoning": 1204,
  "success": true,
  "task_desc": "fix parser bug",
  "tier": "knife",
  "model": "grok-4.6",
  "local_optimized": false,
  "local_model": "",
  "turns": 1,
  "billed_cost": 0.0085,
  "avoided_cost": 0.0,
  "est_cost": 0.0085,
  "cost_source": "api",
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
