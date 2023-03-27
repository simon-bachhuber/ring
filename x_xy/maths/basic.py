import jax.numpy as jnp


def wrap_to_pi(phi):
    "Wraps angle `phi` (radians) to interval [-pi, pi]."
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi
