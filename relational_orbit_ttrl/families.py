"""Executable task families and exact answer maps for broader validation."""

from dataclasses import dataclass
import random
import re

from .model_pilot import make_item, orbit_permutations, parse_answer, permuted_view


@dataclass
class VerifiedOrbit:
    family: str
    truth: int
    prompts: list[str]
    inverse_maps: list
    support: tuple
    parser: object
    metadata: dict


def option_orbit(seed: int, views: int = 4, difficulty: str = "easy") -> VerifiedOrbit:
    item = make_item(seed, difficulty)
    permutations = orbit_permutations(views, seed)
    prompts = [permuted_view(item, p) for p in permutations]
    maps = [lambda a, p=p: None if a is None else p.index(a) for p in permutations]
    return VerifiedOrbit("option", item["truth"], prompts, maps, (None, 0, 1, 2, 3),
                         parse_answer, {"permutations": permutations})


def _boolean_expression(rng, names, values, depth):
    expression = names[0]
    value = values[0]
    operations = []
    for i in range(1, depth + 1):
        name = names[i % len(names)]
        other = values[i % len(values)]
        if rng.random() < 0.3:
            name = f"NOT {name}"
            other = not other
        op = rng.choice(("AND", "OR", "XOR"))
        operations.append(op)
        expression = f"({expression} {op} {name})"
        if op == "AND": value = value and other
        elif op == "OR": value = value or other
        else: value = bool(value) ^ bool(other)
    return expression, int(bool(value)), operations


def boolean_orbit(seed: int, views: int = 4, difficulty: str = "easy") -> VerifiedOrbit:
    if views != 4:
        raise ValueError("Boolean pilot uses four certified views")
    rng = random.Random(seed)
    names = ["P", "Q", "R", "S"]
    values = [bool(rng.getrandbits(1)) for _ in names]
    depth = {"easy": 2, "medium": 3, "hard": 4, "extreme": 5}[difficulty]
    expression, truth, operations = _boolean_expression(rng, names, values, depth)
    renamed = {name: f"V{i + 1}" for i, name in enumerate(names)}
    renamed_expression = expression
    for old, new in renamed.items():
        renamed_expression = re.sub(rf"\b{old}\b", new, renamed_expression)

    def render(expr, shown_names, negate):
        assignments = ", ".join(f"{shown_names[n]}={'True' if v else 'False'}"
                                for n, v in zip(names, values))
        query = f"NOT ({expr})" if negate else expr
        return (f"Boolean variables are assigned as follows: {assignments}. Evaluate {query}.\n"
                "A. True\nB. False\nEvaluate privately. Output only 'Answer: A' or 'Answer: B'.")

    identity_names = {n: n for n in names}
    prompts = [render(expression, identity_names, False),
               render(expression, identity_names, True),
               render(renamed_expression, renamed, False),
               render(renamed_expression, renamed, True)]
    maps = [lambda a: a,
            lambda a: None if a is None else 1 - a,
            lambda a: a,
            lambda a: None if a is None else 1 - a]

    def parser(text):
        answer = parse_answer(text)
        return answer if answer in (0, 1) else None

    return VerifiedOrbit("boolean", truth, prompts, maps, (None, 0, 1), parser,
                         {"operations": operations, "values": values})


def affine_orbit(seed: int, views: int = 4, difficulty: str = "easy") -> VerifiedOrbit:
    if views != 4:
        raise ValueError("Affine pilot uses four certified views")
    rng = random.Random(seed)
    moduli = {"easy": (7, 11, 13), "medium": (11, 13, 17),
              "hard": (17, 19, 23), "extreme": (23, 29, 31)}[difficulty]
    steps_range = {"easy": (1, 2), "medium": (2, 3), "hard": (3, 4),
                   "extreme": (4, 5)}[difficulty]
    modulus = rng.choice(moduli)
    start = rng.randrange(modulus)
    value = start
    steps = []
    for _ in range(rng.randint(*steps_range)):
        a = rng.randrange(2, modulus)
        b = rng.randrange(1, modulus)
        steps.append((a, b)); value = (a * value + b) % modulus
    # Characterization uses the certified affine-shift subset first. It changes
    # the answer while keeping transformed arithmetic difficulty comparable.
    transforms = [(1, d) for d in range(min(views, modulus))]
    operations = "; ".join(f"{i}. x=({a}x+{b}) mod {modulus}"
                           for i, (a, b) in enumerate(steps, 1))
    prompts, maps = [], []
    for c, d in transforms:
        target = "x" if (c, d) == (1, 0) else f"z=({c}x+{d}) mod {modulus}"
        prompts.append(
            f"Start with x={start}. Apply exactly {len(steps)} operations once in order: "
            f"{operations}. Report {target}. Show one short calculation line per operation and "
            "one line for the target conversion. Use at most six lines and fewer than 80 words. "
            "End with exactly 'Answer: N', where N is an integer.")
        inv_c = pow(c, -1, modulus)
        maps.append(lambda answer, inv_c=inv_c, d=d, m=modulus:
                    None if answer is None else (inv_c * (answer - d)) % m)

    def parser(text):
        matches = re.findall(r"ANSWER\s*:\s*(-?\d+)", text.upper())
        if not matches:
            return None
        answer = int(matches[-1]) % modulus
        return answer

    return VerifiedOrbit("affine", value, prompts, maps, tuple([None, *range(modulus)]),
                         parser, {"modulus": modulus, "steps": steps,
                                  "transforms": transforms})


def build_orbit(family: str, seed: int, views: int = 4,
                difficulty: str = "easy") -> VerifiedOrbit:
    builders = {"option": option_orbit, "boolean": boolean_orbit, "affine": affine_orbit}
    return builders[family](seed, views, difficulty)


def validate_orbit(orbit: VerifiedOrbit):
    valid = [x for x in orbit.support if x is not None]
    for inverse in orbit.inverse_maps:
        mapped = [inverse(x) for x in valid]
        if set(mapped) != set(valid):
            raise ValueError(f"{orbit.family} answer map is not bijective")
    if orbit.truth not in valid:
        raise ValueError("truth outside declared support")
