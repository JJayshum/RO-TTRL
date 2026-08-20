from relational_orbit_ttrl.paper_analysis import analyze_shared


def _row(content_hash, root, mapped, gauge):
    return {
        "content_sha256": content_hash,
        "root_correct": root,
        "mapped_correct": mapped,
        "gauge_correct": [gauge] * 4,
        "truth_support": True,
        "parse_failure_rate": 0.0,
    }


def test_shared_analysis_preserves_question_pairing_across_models():
    data = {
        model: {
            "arc": [_row("a", False, True, False), _row("b", False, True, False)],
            "mmlu": [_row("c", False, True, False), _row("d", False, True, False)],
        }
        for model in ("llama3b", "qwen3b", "qwen7b")
    }

    result = analyze_shared(data, seed=1, replicates=100)

    assert result["overall"]["independent_question_blocks"] == 4
    assert result["overall"]["model_question_observations"] == 12
    assert result["overall"]["fixed_model_average_mapped_minus_root"] == 1.0
    assert result["paper_gate"]["passed"] is True


def test_shared_analysis_rejects_nonidentical_question_sets():
    data = {
        "qwen3b": {"arc": [_row("a", False, True, False)]},
        "qwen7b": {"arc": [_row("b", False, True, False)]},
    }

    try:
        analyze_shared(data, replicates=10)
    except ValueError as error:
        assert "shared question hashes differ" in str(error)
    else:
        raise AssertionError("expected mismatched shared hashes to fail")
