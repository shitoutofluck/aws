"""Tests for the token efficiency ledger and dashboard."""

from __future__ import annotations

import json

import pytest

import dashboard
import token_efficiency as te


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    path = tmp_path / "token_ledger.jsonl"
    monkeypatch.setenv("TOKEN_LEDGER", str(path))
    return path


def test_ledger_path_follows_override(isolated_ledger):
    assert te.ledger_path() == isolated_ledger


def test_hermes_home_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    assert te.hermes_home() == tmp_path / "hermes"


def test_hermes_home_has_platform_default(monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert te.hermes_home().name == "hermes"


def test_record_run_creates_ledger_and_entry(isolated_ledger):
    entry = te.record_run("coding", 10_000, 500, tier="knife", task_desc="fix parser")

    assert isolated_ledger.exists()
    assert entry["total"] == 10_500
    assert entry["i_o_ratio"] == 20.0
    assert entry["timestamp"].endswith("Z")
    assert json.loads(isolated_ledger.read_text().strip()) == entry


def test_record_run_rejects_unknown_tier():
    with pytest.raises(ValueError):
        te.record_run("coding", 10, 10, tier="chainsaw")


def test_record_run_rejects_negative_tokens():
    with pytest.raises(ValueError):
        te.record_run("coding", -1, 10)


def test_cached_tokens_bill_at_discount():
    full = te.estimate_cost(1_000_000, 0)
    half_cached = te.estimate_cost(1_000_000, 0, cached=500_000)

    assert full == pytest.approx(1.0)
    assert half_cached == pytest.approx(0.625)


def test_cached_cannot_exceed_input():
    assert te.estimate_cost(100, 0, cached=1_000) == te.estimate_cost(100, 0, cached=100)


def test_local_runs_are_free_but_record_avoided_cost():
    entry = te.record_run(
        "coding", 1_000_000, 1_000_000, local_optimized=True, local_model="llama3.2:1b"
    )

    assert entry["billed_cost"] == 0.0
    assert entry["avoided_cost"] == pytest.approx(3.0)
    assert entry["local_model"] == "llama3.2:1b"


def test_local_model_is_dropped_for_grok_runs():
    entry = te.record_run("coding", 10, 10, local_model="llama3.2:1b")
    assert entry["local_model"] == ""


def test_load_ledger_skips_malformed_lines(isolated_ledger):
    te.record_run("coding", 100, 10)
    with open(isolated_ledger, "a", encoding="utf-8") as handle:
        handle.write("not json\n\n")
    te.record_run("research", 200, 20)

    assert [r["class"] for r in te.load_ledger()] == ["coding", "research"]


def test_load_ledger_empty_when_missing():
    assert te.load_ledger() == []


def test_summarize_empty_ledger():
    stats = te.summarize([])
    assert stats["runs"] == 0
    assert stats["by_class"] == {}


def test_summarize_aggregates_tiers_classes_and_cost():
    te.record_run("coding", 10_000, 1_000, cached=5_000, tier="knife")
    te.record_run("coding", 30_000, 2_000, tier="nuke", success=False)
    te.record_run("research", 4_000, 500, tier="knife", local_optimized=True,
                  local_model="llama3.2:1b")

    stats = te.summarize(te.load_ledger())

    assert stats["runs"] == 3
    assert stats["tokens"] == 47_500
    assert stats["cache_hit_rate"] == pytest.approx(5_000 / 44_000, rel=1e-3)
    assert stats["knife_share"] == pytest.approx(2 / 3, rel=1e-3)
    assert stats["local_share"] == pytest.approx(1 / 3, rel=1e-3)
    assert stats["success_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert stats["by_tier"]["knife"]["runs"] == 2
    assert stats["by_tier"]["nuke"]["runs"] == 1
    assert stats["by_class"]["coding"]["tokens"] == 43_000
    assert stats["avoided_cost"] == pytest.approx(te.estimate_cost(4_000, 500))
    assert stats["billed_cost"] == pytest.approx(
        te.estimate_cost(10_000, 1_000, 5_000) + te.estimate_cost(30_000, 2_000)
    )


def test_summarize_reads_legacy_entries_without_cost_split():
    legacy = [
        {"class": "coding", "input": 100, "output": 10, "total": 110, "est_cost": 0.5},
        {"class": "coding", "input": 100, "output": 10, "total": 110, "est_cost": 0.5,
         "local_optimized": True},
    ]
    stats = te.summarize(legacy)

    assert stats["billed_cost"] == pytest.approx(0.5)
    assert stats["avoided_cost"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "desc,expected",
    [
        ("refactor the parser", "coding"),
        ("investigate why latency spiked", "research"),
        ("migrate the batch pipeline", "tool-heavy"),
        ("hi", "short-chat"),
    ],
)
def test_classify_task(desc, expected):
    assert te.classify_task(desc) == expected


def test_format_dashboard_without_runs():
    assert "No runs recorded yet." in te.format_dashboard([])


def test_format_dashboard_reports_tiers_and_local_model():
    te.record_run("coding", 10_000, 1_000, tier="nuke")
    te.record_run("research", 4_000, 500, local_optimized=True, local_model="llama3.2:1b")

    output = te.format_dashboard(te.load_ledger())

    assert "nuke" in output
    assert "llama3.2:1b" in output
    assert "Local share     : 50.0%" in output


def test_cli_record_then_summary(capsys):
    assert dashboard.main(
        ["record", "coding", "--input", "1000", "--output", "100", "--tier", "scalpel"]
    ) == 0
    capsys.readouterr()

    assert dashboard.main(["summary", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)

    assert stats["runs"] == 1
    assert stats["by_tier"]["scalpel"]["runs"] == 1


def test_cli_record_classifies_when_class_omitted(capsys):
    assert dashboard.main(
        ["record", "--input", "10", "--output", "1", "--desc", "investigate the crash"]
    ) == 0
    entry = json.loads(capsys.readouterr().out)

    assert entry["class"] == "research"


def test_cli_local_flag_defaults_to_ollama(capsys):
    assert dashboard.main(["record", "coding", "--input", "10", "--output", "1", "--local"]) == 0
    entry = json.loads(capsys.readouterr().out)

    assert entry["local_optimized"] is True
    assert entry["local_model"] == "ollama"


def test_cli_show_is_the_default_command(capsys):
    assert dashboard.main([]) == 0
    assert "Token Efficiency" in capsys.readouterr().out


def test_cli_where_prints_ledger_path(capsys, isolated_ledger):
    assert dashboard.main(["where"]) == 0
    assert capsys.readouterr().out.strip() == str(isolated_ledger)
