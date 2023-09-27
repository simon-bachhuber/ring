from typing import Callable, Protocol

import jax
from tree_utils import PyTree

from ... import base

PRNGKey = jax.Array
InputExtras = base.System
OutputExtras = tuple[PRNGKey, jax.Array, jax.Array, base.System]
Xy = PyTree
BatchedXy = PyTree
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
    def __call__(
        self,
        gen: GeneratorWithInputOutputExtras
        | GeneratorWithOutputExtras
        | GeneratorWithInputExtras,
    ) -> (
        GeneratorWithInputOutputExtras
        | GeneratorWithOutputExtras
        | GeneratorWithInputExtras
        | Generator
    ):
        ...
