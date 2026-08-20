from __future__ import annotations

import json

import run_experiment as runner


def test_parse_json_and_canonicalize() -> None:
    parsed = runner.parse_json('<think>hidden</think>```json\n{"top1":"self direction"}\n```')
    assert parsed == {"top1": "self direction"}
    assert runner.canonicalize(parsed["top1"]) == "Self-Direction"


def test_scores_gold_and_all_l0_families() -> None:
    gold_summary, _ = runner.score_gold(
        [
            {
                "gold_value": "Security",
                "content": '{"top1":"Security","top2":"Power","top3":"Conformity"}',
            }
        ]
    )
    assert gold_summary["micro_acc1"] == 1.0
    assert gold_summary["macro_acc3"] == 0.1

    rows = [
        {
            "task_family": "h0_item_to_value",
            "gold_value": "Power",
            "content": '{"top1":"Power","top2":"Achievement","top3":"Security"}',
        },
        {
            "task_family": "h0_conflict_recognition",
            "gold_pair_unordered": "Power|Universalism",
            "content": '{"value_a":"Universalism","value_b":"Power"}',
        },
        {
            "task_family": "h0_contextual_relevance",
            "answer": "yes",
            "content": '{"answer":"yes"}',
        },
    ]
    l0_summary, _ = runner.score_l0(rows)
    assert l0_summary["h0_hits_at_1"] == 1.0
    assert l0_summary["h0_pair_match"] == 1.0
    assert l0_summary["h0_relv_acc"] == 1.0


def test_smoke_run_is_offline_and_resumable(monkeypatch, tmp_path) -> None:
    calls = 0

    def fake_call_api(**kwargs):
        nonlocal calls
        calls += 1
        prompt = kwargs["messages"][-1]["content"]
        if "value_a" in prompt and "value_b" in prompt:
            return '{"value_a":"Power","value_b":"Universalism"}', {}
        if '"answer"' in prompt:
            return '{"answer":"yes"}', {}
        return '{"top1":"Security","top2":"Power","top3":"Conformity"}', {}

    monkeypatch.setattr(runner, "call_api", fake_call_api)
    run_dir = tmp_path / "run"
    argv = [
        "--model",
        "mock-model",
        "--api-base",
        "http://127.0.0.1:1/v1",
        "--smoke",
        "--run-dir",
        str(run_dir),
    ]
    assert runner.main(argv) == 0
    assert calls == 10
    assert runner.main(argv) == 0
    assert calls == 10
    summary = json.loads((run_dir / "combined_summary.json").read_text())
    assert summary["gold1000"]["n_items"] == 5
    assert summary["l0_h0"]["n_items"] == 5
