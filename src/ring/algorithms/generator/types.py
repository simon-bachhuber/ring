from typing import Callable

import jax
from tree_utils import PyTree

from ring import base

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
