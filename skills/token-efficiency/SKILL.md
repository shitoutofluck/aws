---
name: token-efficiency
description: Persistent token-efficiency tracking and live dashboard for Grok Build / Grok bots. Classifies runs, records metrics (tokens, cost, ratios), exposes /tokens or launch dashboard. Uses Hermes /usage, /insights, memory, file ledgers.
license: MIT
---

# Token Efficiency Tracker

## Features
- Live session tokens via `/usage` or hermes insights
- Cumulative totals since install
- Per-optimization savings estimate (Grok spend avoided by local models)
- Persistent classified runs: `coding`, `research`, `tool-heavy`, `short-chat`, `long-horizon`
- Metrics: input/output/total/cached, est cost (Grok pricing), ratios (I:O, tokens/task, cache hit, turns, success rate)
- Context tiers: `knife` (read only what the task needs), `scalpel` (targeted expansion), `nuke` (whole tree in context)
- Trends queryable per class and per tier
- Dashboard: cumulative, per-class, per-tier, recent history

## Usage
- `/tokens` or `hermes tokens` — show live dashboard
- `/efficiency` — same
- After a task: auto-classify and record, via the skill or manually
- Launch dashboard: `python dashboard.py` or `hermes skill token-efficiency dashboard`

## Commands
```bash
python dashboard.py                                             # dashboard
python dashboard.py record coding --from-response run.json      # real API counts
python dashboard.py record coding --input 12000 --output 800 --tier knife
python dashboard.py record --input 4000 --output 300 --local llama3.2:1b --desc "summarize diff"
python dashboard.py summary --json                              # machine-readable
python dashboard.py where                                       # ledger path
```

Omit the task class and it is inferred from `--desc`. Pass `--cached N` for prompt-cache
hits, `--failed` for runs that did not land, `--from-response -` to read from stdin.

## Actual vs estimated usage
`record_from_response(response)` reads the counts xAI actually reported — `prompt_tokens`,
`completion_tokens`, `prompt_tokens_details.cached_tokens`,
`completion_tokens_details.reasoning_tokens` — plus the exact price from
`cost_in_usd_ticks`. Both the Chat Completions and Responses API shapes are handled, as SDK
objects or parsed JSON.

`record_run(...)` records whatever numbers it is given and estimates cost from list pricing.
Every entry carries `cost_source` (`api` or `estimate`) so the two are never confused.

## Pricing (Grok, approx current)
Used only when the API reports no cost. `grok-build-0.1`: ~$1/M input, $2/M output, cached
input at ~$0.25/M. Reasoning tokens bill at the output rate and are already counted in
output.

## Ledger
`$HERMES_HOME/profiles/$HERMES_PROFILE/token_ledger.jsonl`, one JSON object per run.

- `HERMES_HOME` defaults to `%LOCALAPPDATA%\hermes` on Windows and
  `$XDG_DATA_HOME/hermes` (else `~/.local/share/hermes`) elsewhere.
- `HERMES_PROFILE` defaults to `first-agent`.
- `TOKEN_LEDGER` overrides the full path for one-off runs and tests.

Append-only, so it survives restarts; malformed lines are skipped rather than
aborting the read.

## Accounting
A run marked local (`--local MODEL`) is served by Ollama, so `billed_cost` is 0 and
`avoided_cost` holds what the same run would have cost on Grok. The dashboard reports
both, so "savings" is real Grok spend avoided rather than a character-count proxy.

## Implementation notes
- Query `hermes insights` for aggregates; parse `.usage.json` for live session tokens.
- Classification falls back to keyword matching on the task description.
- Record on task end.

## Quick launch
```bash
python -c "from token_efficiency import show_dashboard; show_dashboard()"
```

## Verify
```bash
python -m pytest test_token_efficiency.py
```
Record a few test runs and confirm the dashboard returns live plus historical numbers.
