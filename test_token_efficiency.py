"""Tests for the token efficiency ledger and dashboard."""

from __future__ import annotations

import io
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


CHAT_RESPONSE = {
    "model": "grok-4.6",
    "usage": {
        "prompt_tokens": 125,
        "completion_tokens": 48,
        "total_tokens": 173,
        "prompt_tokens_details": {"text_tokens": 125, "cached_tokens": 98},
        "completion_tokens_details": {"reasoning_tokens": 30},
        "cost_in_usd_ticks": 12_345_678_900,
    },
}

RESPONSES_API_RESPONSE = {
    "model": "grok-4.6",
    "usage": {
        "input_tokens": 125,
        "output_tokens": 48,
        "total_tokens": 173,
        "input_tokens_details": {"cached_tokens": 98},
        "output_tokens_details": {"reasoning_tokens": 30},
        "cost_in_nano_usd": 1_500_000_000,
    },
}


class _Details:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _SdkResponse:
    """Stand-in for an SDK object exposing usage as attributes."""

    def __init__(self):
        self.model = "grok-4.6"
        self.usage = _Details(
            prompt_tokens=125,
            completion_tokens=48,
            total_tokens=173,
            prompt_tokens_details=_Details(cached_tokens=98),
            completion_tokens_details=_Details(reasoning_tokens=30),
            cost_in_usd_ticks=12_345_678_900,
        )


def test_extract_usage_from_chat_completions_shape():
    usage = te.extract_usage(CHAT_RESPONSE)

    assert (usage.input, usage.output, usage.total) == (125, 48, 173)
    assert usage.cached == 98
    assert usage.reasoning == 30
    assert usage.cost_usd == pytest.approx(1.23456789)
    assert usage.model == "grok-4.6"


def test_extract_usage_from_responses_api_shape():
    usage = te.extract_usage(RESPONSES_API_RESPONSE)

    assert (usage.input, usage.output, usage.total) == (125, 48, 173)
    assert usage.cached == 98
    assert usage.reasoning == 30
    assert usage.cost_usd == pytest.approx(1.5)


def test_extract_usage_from_sdk_object():
    assert te.extract_usage(_SdkResponse()) == te.extract_usage(CHAT_RESPONSE)


def test_extract_usage_accepts_bare_usage_block():
    assert te.extract_usage(CHAT_RESPONSE["usage"]).input == 125


def test_extract_usage_without_reported_cost():
    usage = te.extract_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    assert usage.cost_usd is None
    assert usage.total == 12


def test_extract_usage_rejects_response_without_usage():
    with pytest.raises(ValueError):
        te.extract_usage({"choices": []})


def test_record_from_response_uses_reported_cost():
    entry = te.record_from_response(CHAT_RESPONSE, "coding", tier="knife")

    assert entry["input"] == 125
    assert entry["cached"] == 98
    assert entry["reasoning"] == 30
    assert entry["cost_source"] == "api"
    assert entry["billed_cost"] == pytest.approx(1.23456789)
    assert entry["model"] == "grok-4.6"


def test_record_from_response_falls_back_to_estimate():
    entry = te.record_from_response({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})

    assert entry["cost_source"] == "estimate"
    assert entry["billed_cost"] == pytest.approx(1.0)


def test_record_from_response_classifies_from_description():
    entry = te.record_from_response(CHAT_RESPONSE, task_desc="refactor the tokenizer")
    assert entry["class"] == "coding"


def test_manual_records_are_marked_as_estimates():
    entry = te.record_run("coding", 100, 10)
    assert entry["cost_source"] == "estimate"


def test_summary_counts_metered_runs():
    te.record_from_response(CHAT_RESPONSE, "coding")
    te.record_run("coding", 100, 10)

    stats = te.summarize(te.load_ledger())

    assert stats["metered_runs"] == 1
    assert stats["reasoning"] == 30


def test_dashboard_flags_mixed_cost_provenance():
    te.record_from_response(CHAT_RESPONSE, "coding")
    te.record_run("coding", 100, 10)

    assert "1/2 from API, rest estimated" in te.format_dashboard(te.load_ledger())


def test_dashboard_flags_fully_metered_costs():
    te.record_from_response(CHAT_RESPONSE, "coding")

    assert "(provider-reported)" in te.format_dashboard(te.load_ledger())


def test_cli_record_from_response_file(tmp_path, capsys):
    path = tmp_path / "response.json"
    path.write_text(json.dumps(CHAT_RESPONSE), encoding="utf-8")

    assert dashboard.main(["record", "coding", "--from-response", str(path)]) == 0
    entry = json.loads(capsys.readouterr().out)

    assert entry["input"] == 125
    assert entry["cost_source"] == "api"


def test_cli_record_from_response_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(CHAT_RESPONSE)))

    assert dashboard.main(["record", "coding", "--from-response", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["cached"] == 98


def test_cli_record_requires_tokens_without_response(capsys):
    assert dashboard.main(["record", "coding"]) == 2
    assert "--input and --output are required" in capsys.readouterr().err


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
