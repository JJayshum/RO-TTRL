import copy

import pytest

from relational_orbit_ttrl.generality_analysis import analyze_panel


def make_data(*, harmful_model=False):
    data = {}
    for model_index, model in enumerate(("gemma2b", "mistral7b", "phi35")):
        data[model] = {}
        for benchmark_index, benchmark in enumerate(("arc_challenge", "mmlu", "openbookqa")):
            rows = []
            for row_index in range(12):
                mapped_correct = not (harmful_model and model_index == 0)
                rows.append({
                    "content_sha256": f"{benchmark_index * 100 + row_index:064x}",
                    "root_correct": False if mapped_correct else True,
                    "mapped_correct": mapped_correct,
                    "gauge_correct": [False] * 20,
                    "truth_support": True,
                    "parse_failure_rate": 0.0,
                })
            data[model][benchmark] = rows
    return data


def test_positive_panel_passes_all_frozen_criteria():
    result = analyze_panel(make_data(), seed=9, replicates=200)
    assert result["generality_gate"]["passed"]
    assert result["overall"]["independent_question_blocks"] == 36
    assert result["overall"]["model_question_observations"] == 108
    assert all(row["individual_positive_replication"] for row in result["by_model"].values())


def test_harmful_candidate_is_retained_and_fails_panel():
    result = analyze_panel(make_data(harmful_model=True), seed=9, replicates=200)
    assert not result["generality_gate"]["passed"]
    assert not result["generality_gate"]["criteria"][
        "no_model_interval_wholly_below_zero"
    ]


def test_panel_requires_exactly_three_aligned_models():
    data = make_data()
    del data["phi35"]
    with pytest.raises(ValueError, match="exactly three"):
        analyze_panel(data, replicates=10)

    data = make_data()
    broken = copy.deepcopy(data)
    broken["phi35"]["mmlu"].pop()
    with pytest.raises(ValueError, match="shared question hashes differ"):
        analyze_panel(broken, replicates=10)
