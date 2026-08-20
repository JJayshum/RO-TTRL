from relational_orbit_ttrl.pooling import canonical_pool, leave_one_out_rewards
from relational_orbit_ttrl.transforms import OptionPermutation, validate_map


def test_option_map_round_trip_and_pooling():
    m = OptionPermutation((2, 0, 1))
    validate_map(m, range(3))
    # view 1 contains the same root answer 0 at transformed index 2
    q = canonical_pool([[0, 0, 1], [2, 2, 0]], [lambda x: x, m.inverse],
                       support=(0, 1, 2), alpha=1.0)
    assert max(q, key=q.get) == 0


def test_leave_one_out_shape_and_finite_values():
    rewards = leave_one_out_rewards([[0, 1, 0], [0, 0, 1]],
                                    [lambda x: x, lambda x: x], support=(0, 1), alpha=1.0)
    assert [len(x) for x in rewards] == [3, 3]
    assert all(0 < r < 1 for row in rewards for r in row)
