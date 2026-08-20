from relational_orbit_ttrl.model_pilot import (
    make_item, orbit_permutations, parse_answer, permuted_view, randomized_inverse,
)


def test_generated_item_and_permutation_are_exact():
    item = make_item(12)
    value = item["options"][item["truth"]]
    x = int(item["stem"].split("Start with x=")[1].split(".")[0])
    for a, b in item["steps"]:
        x = (a * x + b) % item["modulus"]
    assert x == value
    p = (2, 0, 3, 1)
    text = permuted_view(item, p)
    assert f"{chr(65 + p[item['truth']])}. {value}" in text


def test_answer_parser():
    assert parse_answer("work... Answer: C") == 2
    assert parse_answer("no final answer") is None


def test_cyclic_views_disperse_every_fixed_position():
    permutations = orbit_permutations(4, seed=1)
    for fixed_node_answer in range(4):
        roots = {p.index(fixed_node_answer) for p in permutations}
        assert roots == set(range(4))


def test_randomized_gauge_inverse_matches_composition():
    p = (2, 0, 3, 1)  # canonical -> node
    rho = (1, 3, 0, 2)  # node -> gauged node
    inverse = randomized_inverse(p, rho)
    for canonical in range(4):
        observed = rho[p[canonical]]
        assert inverse(observed) == canonical
