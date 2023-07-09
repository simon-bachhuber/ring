from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

import x_xy
from x_xy import base
from x_xy.algebra import transform_mul

X = Y = PARAMS = STATE = dict


class HaikuTransformedWithState:
    "Holds a pair of pure functions."

    init: Callable[[jax.random.PRNGKey, X], Tuple[PARAMS, STATE]]
    apply: Callable[[PARAMS, STATE, X], Tuple[Y, STATE]]


RNNO_FN = Callable[[base.System], HaikuTransformedWithState]


def predict(
    sys: base.System,
    rnno_fn: RNNO_FN,
    X: X,
    y: Optional[Y] = None,
    xs: Optional[base.Transform] = None,
    sys_xs: Optional[base.System] = None,
    params: Optional[PARAMS] = None,
    error_warmup_time: float = 2.0,
    plot: bool = False,
    render: bool = False,
    render_prediction: bool = True,
    render_path: str = "animation.mp4",
    verbose: bool = True,
    T_plot: float = 20.0,
):
    import matplotlib.pyplot as plt

    rnno = rnno_fn(sys)

    # TODO
    # right now the key used in init can be hardcoded because
    # i am always using a state which is non-stochastic and
    # rather it is just zeros
    initial_params, state = rnno.init(
        jax.random.PRNGKey(
            1,
        ),
        X,
    )

    if params is None:
        params = initial_params

    yhat, _ = rnno.apply(params, state, X)
    metrics = {}
    warmup = int(error_warmup_time / sys.dt)
    if y is not None:
        for link_name in y:
            metrics[f"mae_deg_{link_name}"] = jnp.mean(
                jnp.rad2deg(x_xy.maths.angle_error(y[link_name], yhat[link_name]))[
                    warmup:
                ]
            )
        if verbose:
            print(metrics)

    if plot:
        ts = jnp.arange(0.0, T_plot, step=sys.dt)
        N = len(ts)
        fig, axes = plt.subplots(len(yhat), 3, figsize=(10, 3 * len(yhat)))
        for row, link_name in enumerate(yhat.keys()):
            euler_angles_hat = jnp.rad2deg(quat_to_euler(yhat[link_name]))
            euler_angles = (
                jnp.rad2deg(quat_to_euler(y[link_name])) if y is not None else None
            )

            for col, xyz in enumerate(["x", "y", "z"]):
                axis = axes[row, col]
                axis.plot(ts, euler_angles_hat[:N, col], label="prediction")
                if euler_angles is not None:
                    axis.plot(ts, euler_angles[:N, col], label="truth")
                axis.grid(True)
                axis.set_title(link_name + "_" + xyz)
                axis.set_xlabel("time [s]")
                axis.set_ylabel("euler angles [deg]")
        fig.tight_layout()

    if render:
        assert xs is not None
        assert sys_xs is not None
        assert y is not None

        sys_render = sys_xs
        xs_render = xs
        if render_prediction:
            # replace render color of geoms for render of predicted motion
            prediction_color = jnp.array([78.0, 163, 243, 255]) / 255
            sys_newcolor = _geoms_replace_color(sys, prediction_color)
            sys_render = x_xy.utils.inject_system(sys_xs, sys_newcolor, prefix="hat_")

            xshat = _forward_kinematics_mix_in_transform2(sys, xs, yhat, sys_xs)

            # TODO
            # concatenate `xs` from both systems to get `xs` corresponding
            # to `sys_render`. This uses the fact that `inject_system` always
            # exactly appends at the index end.
            xs_render_simple = xs.concatenate(xshat, axis=1)

            # this creates an identical version of `xs_render_simple` but
            # using a more "correct" yet less efficient algorithm
            # for now it is commented out
            if False:
                # swap time axis, and link axis
                xs, xshat = xs.transpose((1, 0, 2)), xshat.transpose((1, 0, 2))
                # create mapping from `name` -> Transform
                xs_dict = dict(
                    zip(
                        ["hat_" + name for name in sys.link_names],
                        [xshat[i] for i in range(sys.num_links())],
                    )
                )
                xs_dict.update(
                    dict(
                        zip(
                            sys_xs.link_names,
                            [xs[i] for i in range(sys_xs.num_links())],
                        )
                    )
                )

                xs_render = []
                for name in sys_render.link_names:
                    xs_render.append(xs_dict[name])
                xs_render = xs_render[0].batch(*xs_render[1:])
                xs_render = xs_render.transpose((1, 0, 2))

                from x_xy.utils import tree_equal

                assert tree_equal(xs_render, xs_render_simple)
            xs_render = xs_render_simple

        x_xy.render.animate(
            render_path, sys_render, xs_render, fps=25, vispy_backend="pyqt6"
        )

    return yhat, metrics


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, None))
def _forward_kinematics_mix_in_transform2(
    sys: base.System, xs: base.Transform, y: Y, sys_xs: base.System
) -> base.Transform:
    eps_to_l = {-1: base.Transform.zero()}

    def f(_, __, i, p, name, type, transform1):
        if p == -1:
            transform = xs[sys_xs.name_to_idx(name)]
        else:
            assert (
                type[0] == "r"
            ) or type == "free", f"The link `{name}` is of type `{type}` "
            "but `y` only contains quaternions."
            transform2 = base.Transform.create(rot=y[name])
            transform = transform_mul(transform2, transform1)
        eps_to_l[i] = transform_mul(transform, eps_to_l[p])
        return eps_to_l[i]

    return x_xy.scan.tree(
        sys,
        f,
        "lllll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_names,
        sys.link_types,
        sys.links.transform1,
    )


def euler_to_quat(v: jnp.ndarray) -> jnp.ndarray:
    """Converts euler rotations in degrees to quaternion."""
    # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
    c1, c2, c3 = jnp.cos(v * jnp.pi / 360)
    s1, s2, s3 = jnp.sin(v * jnp.pi / 360)
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * c2 * c3 + c1 * s2 * s3
    y = c1 * s2 * c3 - s1 * c2 * s3
    z = c1 * c2 * s3 + s1 * s2 * c3
    return jnp.array([w, x, y, z])


@jax.jit
@jax.vmap
def quat_to_euler(q: jnp.ndarray) -> jnp.ndarray:
    """Converts quaternions to euler rotations in radians."""
    # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''

    z = jnp.arctan2(
        -2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2],
    )
    # TODO: Investigate why quaternions go so big we need to clip.
    y = x_xy.maths.safe_arcsin(jnp.clip(2 * q[1] * q[3] + 2 * q[0] * q[2], -1.0, 1.0))
    x = jnp.arctan2(
        -2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0],
    )

    return jnp.array([x, y, z])


def _geoms_replace_color(sys, color):
    def _update_color_in_dict(vispy_kwargs):
        vispy_kwargs = vispy_kwargs.copy()
        vispy_kwargs["color"] = color
        return vispy_kwargs

    geoms = [
        g.replace(vispy_kwargs=_update_color_in_dict(g.vispy_kwargs)) for g in sys.geoms
    ]
    return sys.replace(geoms=geoms)
