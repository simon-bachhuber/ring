import jax.numpy as jnp


def wrap_to_pi(phi):
    "Wraps angle `phi` (radians) to interval [-pi, pi]."
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


x_unit_vector = jnp.array([1.0, 0, 0])
y_unit_vector = jnp.array([0.0, 1, 0])
z_unit_vector = jnp.array([0.0, 0, 1])


def unit_vectors(xyz: int | str):
    if isinstance(xyz, str):
        xyz = {"x": 0, "y": 1, "z": 2}[xyz]
    return [x_unit_vector, y_unit_vector, z_unit_vector][xyz]
