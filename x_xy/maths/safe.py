from functools import partial

import jax
import jax.numpy as jnp


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


@jax.custom_jvp
def safe_normalize_custom_jvp(x):
    """Execution- and Grad-safe for x=0.0. Normalizes along last axis."""
    assert x.ndim == 1

    is_zero = jnp.allclose(x, 0.0)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.zeros_like(x),
        lambda x: x / jnp.where(is_zero, 1.0, safe_norm(x)),
        x,
    )


@safe_normalize_custom_jvp.defjvp
def safe_normalize_custom_jvp_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    ans = safe_normalize_custom_jvp(x)
    eps = 1e-5
    ans_dot = (jnp.clip(1 / (jnp.linalg.norm(x) + eps), 1e-4, 1)) * x_dot
    return ans, ans_dot
