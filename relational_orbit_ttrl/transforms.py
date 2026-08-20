"""Small, deterministic answer-map primitives used by the pilot."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping


class TransformationError(ValueError):
    pass


@dataclass(frozen=True)
class OptionPermutation:
    """Map old option indices to their new positions (zero based)."""
    permutation: tuple[int, ...]

    def __post_init__(self):
        n = len(self.permutation)
        if set(self.permutation) != set(range(n)):
            raise TransformationError("option map must be a bijection over 0..n-1")

    def forward(self, answer: int) -> int:
        return self.permutation[answer]

    def inverse(self, answer: int) -> int:
        return self.permutation.index(answer)


@dataclass(frozen=True)
class BooleanNegation:
    def forward(self, answer: int) -> int:
        if answer not in (0, 1):
            raise TransformationError("Boolean answers must be 0 or 1")
        return 1 - answer

    inverse = forward


@dataclass(frozen=True)
class AffineMap:
    scale: Fraction
    shift: Fraction

    def __post_init__(self):
        if self.scale == 0:
            raise TransformationError("affine scale cannot be zero")

    def forward(self, answer: Fraction) -> Fraction:
        return self.scale * answer + self.shift

    def inverse(self, answer: Fraction) -> Fraction:
        return (answer - self.shift) / self.scale


def validate_map(mapping, domain: Iterable, *, require_round_trip: bool = True) -> None:
    """Validate a map without using model outputs or target labels."""
    domain = tuple(domain)
    mapped = tuple(mapping.forward(a) for a in domain)
    if len(set(mapped)) != len(mapped):
        raise TransformationError("map is not injective on the declared domain")
    if require_round_trip:
        for a in domain:
            if mapping.inverse(mapping.forward(a)) != a:
                raise TransformationError("inverse round trip failed")


def compose_maps(first, second):
    """Return a callable composition second(first(x))."""
    class Composition:
        def forward(self, x):
            return second.forward(first.forward(x))
        def inverse(self, x):
            return first.inverse(second.inverse(x))
    return Composition()
