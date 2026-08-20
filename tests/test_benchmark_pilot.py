import json

from relational_orbit_ttrl.benchmark_pilot import (
    benchmark_view,
    load_excluded_hashes,
    normalize_row,
    stable_select,
)


def test_normalize_arc_uses_source_labels_not_assumed_letter_order():
    row = {
        "id": "arc-1",
        "question": "Which one?",
        "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]},
        "answerKey": "3",
    }
    item = normalize_row("arc_challenge", row, 9)
    assert item["truth"] == 2
    assert item["options"][item["truth"]] == "y"


def test_normalize_filters_non_four_option_rows():
    row = {
        "id": "arc-2",
        "question": "Which one?",
        "choices": {"text": ["x", "y", "z"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    assert normalize_row("arc_challenge", row, 1) is None


def test_benchmark_view_applies_exact_option_permutation():
    item = {"stem": "Pick beta.", "options": ["alpha", "beta", "gamma", "delta"],
            "truth": 1}
    permutation = (2, 0, 3, 1)
    prompt = benchmark_view(item, permutation)
    assert "A. beta" in prompt
    assert "B. delta" in prompt
    assert "C. alpha" in prompt
    assert "D. gamma" in prompt


def test_stable_selection_is_order_independent():
    items = [{"content_sha256": f"{i:064x}", "id": str(i)} for i in range(10)]
    forward = stable_select(items, 4, seed=7)
    reverse = stable_select(list(reversed(items)), 4, seed=7)
    assert [item["id"] for item in forward] == [item["id"] for item in reverse]


def test_stable_selection_offset_produces_disjoint_blocks():
    items = [{"content_sha256": f"{i:064x}", "id": str(i)} for i in range(10)]
    first = stable_select(items, 4, seed=7, offset=0)
    second = stable_select(items, 4, seed=7, offset=4)
    assert {item["id"] for item in first}.isdisjoint(item["id"] for item in second)


def test_exclusion_loader_accepts_inference_and_adaptation_manifests(tmp_path):
    inference = tmp_path / "inference.json"
    inference.write_text(json.dumps({"items": [{"content_sha256": "inference"}]}))
    adaptation = tmp_path / "adaptation.json"
    adaptation.write_text(json.dumps({
        "train_items": [{"content_sha256": "train"}],
        "eval_items": [{"content_sha256": "eval"}],
    }))

    hashes = load_excluded_hashes([inference, adaptation])

    assert hashes == {"inference", "train", "eval"}
