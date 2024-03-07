from typing import Callable, Protocol

import jax
from ring import base
from tree_utils import PyTree

PRNGKey = jax.Array
InputExtras = base.System
OutputExtras = tuple[PRNGKey, jax.Array, jax.Array, base.System]
Xy = tuple[PyTree, PyTree]
BatchedXy = tuple[PyTree, PyTree]
GeneratorWithInputExtras = Callable[[PRNGKey, InputExtras], Xy]
GeneratorWithOutputExtras = Callable[[PRNGKey], tuple[Xy, OutputExtras]]
GeneratorWithInputOutputExtras = Callable[
    [PRNGKey, InputExtras], tuple[Xy, OutputExtras]
]
Generator = Callable[[PRNGKey], Xy]
BatchedGenerator = Callable[[PRNGKey], BatchedXy]
SETUP_FN = Callable[[PRNGKey, base.System], base.System]
FINALIZE_FN = Callable[[PRNGKey, jax.Array, base.Transform, base.System], Xy]


class GeneratorTrafo(Protocol):
    def __call__(  # noqa: E704
        self,
        gen: (
            GeneratorWithInputOutputExtras
            | GeneratorWithOutputExtras
            | GeneratorWithInputExtras
        ),
    ) -> (
        GeneratorWithInputOutputExtras
        | GeneratorWithOutputExtras
        | GeneratorWithInputExtras
        | Generator
    ): ...
