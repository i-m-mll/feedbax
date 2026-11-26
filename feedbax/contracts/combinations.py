"""Ordered combinations shared by sweeps and authored member families."""

from collections.abc import Iterator, Mapping, Sequence
from itertools import product


DEFAULT_COMBINATION_LIMIT = 10_000

type Dimension = tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]
"""Axes that vary together, and the value index of each axis at each step."""


def cross_dimension(name: str, size: int) -> Dimension:
    return (name,), tuple((index,) for index in range(size))


def zip_length(sizes: Mapping[str, int]) -> int:
    if len(set(sizes.values())) != 1:
        raise ValueError(f"zip requires equal axis lengths, got {dict(sizes)}")
    return next(iter(sizes.values()))


def zip_dimension(sizes: Mapping[str, int]) -> Dimension:
    return tuple(sizes), tuple((index,) * len(sizes) for index in range(zip_length(sizes)))


def select_combinations[Choice](
    dimensions: Sequence[Dimension], choices: Mapping[str, Sequence[Choice]],
) -> Iterator[dict[str, Choice]]:
    for steps in product(*(steps for _, steps in dimensions)):
        yield {
            name: choices[name][index]
            for (names, _), indices in zip(dimensions, steps, strict=True)
            for name, index in zip(names, indices, strict=True)
        }
