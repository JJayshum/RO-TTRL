import pytest

from relational_orbit_ttrl.submission_analysis import (
    analyze_records,
    leave_one_view_out,
    mapped_predictions,
    risk_coverage,
    root_prediction,
    summarize_confirmation,
)


def make_record():
    return {
        "id": "toy:1",
        "content_sha256": "a" * 64,
        "truth": 2,
        "permutations": [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ],
        "view_samples": [
            [2, 2, 2, 1],
            [3, 3, 3, 0],
            [0, 0, 0, 3],
            [1, 1, 1, 2],
        ],
        "root_samples": [2] * 8 + [1] * 8,
        "root_correct": False,
        "mapped_correct": True,
        "gauge_correct": [False, False, True, False],
        "truth_support": True,
        "parse_failure_rate": 0.0,
    }


def test_budgeted_predictions_use_exact_maps_and_prefixes():
    record = make_record()
    assert mapped_predictions(record, 4)["log_pool"] == 2
    assert mapped_predictions(record, 16)["hard_vote"] == 2
    assert root_prediction(record, 4) == 2
    assert root_prediction(record, 16) == 1


def test_invalid_budget_and_missing_root_samples_fail_closed():
    record = make_record()
    with pytest.raises(ValueError):
        mapped_predictions(record, 6)
    del record["root_samples"]
    with pytest.raises(ValueError, match="predates"):
        root_prediction(record, 4)


def test_leave_one_view_out_is_label_free_and_stable():
    result = leave_one_view_out(make_record())
    assert result["predictions"] == [2, 2, 2, 2]
    assert result["agreement_with_full"] == 1.0
    assert result["margin"] > 0


def test_risk_coverage_uses_frozen_score_then_hash_tie_break():
    rows = [
        {"content_sha256": "b", "loo_agreement": 1.0, "margin": 0.2, "correct": True},
        {"content_sha256": "a", "loo_agreement": 0.5, "margin": 0.9, "correct": False},
        {"content_sha256": "c", "loo_agreement": 1.0, "margin": 0.1, "correct": True},
        {"content_sha256": "d", "loo_agreement": 0.5, "margin": 0.8, "correct": False},
    ]
    curve = risk_coverage(rows)
    assert curve[0]["accuracy"] == 1.0
    assert curve[1]["accuracy"] == 1.0
    assert curve[-1]["accuracy"] == 0.5


def test_analysis_reports_scaling_and_reliability():
    summary, rows = analyze_records([make_record()], seed=7)
    assert len(rows) == 1
    assert [point["budget"] for point in summary["compute_scaling"]] == [4, 8, 16]
    assert summary["reliability"]["all_views_stable_fraction"] == 1.0


def test_confirmation_summary_applies_all_frozen_gate_criteria():
    records = []
    for benchmark in ("arc_challenge", "openbookqa", "mmlu"):
        for index in range(8):
            row = make_record()
            row["id"] = f"{benchmark}:{index}"
            row["content_sha256"] = f"{len(records):064x}"
            records.append(row)
    result = summarize_confirmation(records, seed=3)
    assert result["pooled"]["mapped_minus_root"] == 1.0
    assert result["gate"]["passed"]
    assert set(result["by_benchmark"]) == {"arc_challenge", "openbookqa", "mmlu"}
