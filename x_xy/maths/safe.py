from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_jvp


@partial(jnp.vectorize, signature="(k)->(1)")
def safe_norm(x):
    """Grad-safe for x=0.0. Norm along last axis."""
    assert x.ndim == 1

    is_zero = jnp.all(jnp.isclose(x, 0.0), axis=-1, keepdims=False)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.array([0.0], dtype=x.dtype),
        lambda x: jnp.linalg.norm(x, keepdims=True),
        x,
    )


@partial(jnp.vectorize, signature="(k)->(k)")
def safe_normalize(x):
    """Execution- and Grad-safe for x=0.0. Normalizes along last axis."""
    assert x.ndim == 1

    is_zero = jnp.allclose(x, 0.0)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.zeros_like(x),
        lambda x: x / jnp.where(is_zero, 1.0, safe_norm(x)),
        x,
    )


@custom_jvp
def safe_arccos(x: jnp.ndarray) -> jnp.ndarray:
    """Trigonometric inverse cosine, element-wise with safety clipping in grad."""
    return jnp.arccos(x)


@safe_arccos.defjvp
def _safe_arccos_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arccos(x)
    tangent_out = -x_dot / jnp.sqrt(1.0 - jnp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out


@custom_jvp
def safe_arcsin(x: jnp.ndarray) -> jnp.ndarray:
    """Trigonometric inverse sine, element-wise with safety clipping in grad."""
    return jnp.arcsin(x)


@safe_arcsin.defjvp
def _safe_arcsin_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arccos(x)
    tangent_out = x_dot / jnp.sqrt(1.0 - jnp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out
