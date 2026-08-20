import json

from relational_orbit_ttrl.compatibility_audit import audit_run


def test_audit_accepts_structurally_valid_run(tmp_path):
    bench = tmp_path / "arc_challenge"
    bench.mkdir()
    permutations = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ]
    record = {
        "id": "arc_challenge:1",
        "content_sha256": "abc",
        "truth": 0,
        "permutations": permutations,
        "view_samples": [[permutation[0]] * 4 for permutation in permutations],
        "root_samples": [0] * 16,
    }
    manifest = {
        "benchmark": "arc_challenge",
        "sample_offset": 208,
        "sample_seed": 20260816,
        "generation_seed": 31808,
        "items": [{"content_sha256": "abc"}],
    }
    summary = {"mapped_completion_tokens": 100, "root_completion_tokens": 100}
    (bench / "manifest.json").write_text(json.dumps(manifest))
    (bench / "records.jsonl").write_text(json.dumps(record) + "\n")
    (bench / "summary.json").write_text(json.dumps(summary))

    result = audit_run(tmp_path)

    assert result["stored_run_compatible"] is True
    assert result["structural_failures"] == []


def test_audit_rejects_reference_overlap(tmp_path):
    bench = tmp_path / "run" / "arc_challenge"
    bench.mkdir(parents=True)
    permutations = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
    record = {
        "id": "arc_challenge:1",
        "content_sha256": "duplicate",
        "truth": 0,
        "permutations": permutations,
        "view_samples": [[permutation[0]] * 4 for permutation in permutations],
        "root_samples": [0] * 16,
    }
    manifest = {
        "benchmark": "arc_challenge", "sample_offset": 208,
        "sample_seed": 20260816, "generation_seed": 31808,
        "items": [{"content_sha256": "duplicate"}],
    }
    (bench / "manifest.json").write_text(json.dumps(manifest))
    (bench / "records.jsonl").write_text(json.dumps(record) + "\n")
    (bench / "summary.json").write_text(json.dumps({
        "mapped_completion_tokens": 100, "root_completion_tokens": 100,
    }))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"items": [{"content_sha256": "duplicate"}]}))

    result = audit_run(tmp_path / "run", [reference])

    assert result["stored_run_compatible"] is False
    assert result["reference_overlap_count"] == 1
