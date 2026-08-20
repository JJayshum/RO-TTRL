from relational_orbit_ttrl.families import build_orbit, validate_orbit


def test_all_family_maps_are_bijective_and_truth_is_supported():
    for family in ("option", "boolean", "affine"):
        for seed in range(10):
            orbit = build_orbit(family, seed, 4, "easy")
            validate_orbit(orbit)
            assert len(orbit.prompts) == len(orbit.inverse_maps) == 4


def test_affine_maps_recover_root_truth_from_transformed_answers():
    orbit = build_orbit("affine", 44, 4, "easy")
    modulus = orbit.metadata["modulus"]
    for inverse, (c, d) in zip(orbit.inverse_maps, orbit.metadata["transforms"]):
        transformed = (c * orbit.truth + d) % modulus
        assert inverse(transformed) == orbit.truth


def test_boolean_negation_views_flip_back_to_root():
    orbit = build_orbit("boolean", 12, 4, "easy")
    transformed_truths = [orbit.truth, 1 - orbit.truth, orbit.truth, 1 - orbit.truth]
    assert [inv(x) for inv, x in zip(orbit.inverse_maps, transformed_truths)] == [orbit.truth] * 4
