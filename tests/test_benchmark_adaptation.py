from relational_orbit_ttrl.benchmark_adaptation import policy_effect, summarize


def test_benchmark_adaptation_summary_and_paired_effect():
    root = [
        {"policy_pass1": 0.25, "root_majority": False, "mapped": False,
         "truth_support": True, "parse_failure": 0.0},
        {"policy_pass1": 0.50, "root_majority": True, "mapped": True,
         "truth_support": True, "parse_failure": 0.0},
    ]
    mapped = [dict(root[0], policy_pass1=0.50), dict(root[1], policy_pass1=0.75)]
    assert summarize(root)["policy_pass1"] == 0.375
    assert policy_effect(root, mapped, seed=1)["effect"] == 0.25
