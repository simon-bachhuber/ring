from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tree_utils

import x_xy
from x_xy import base, maths
from x_xy.subpkgs import sim2real, sys_composer

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
    **vispy_kwargs,
):
    """
    NOTE: If you are encountering the error that the rendered frames are 1x1 pixels
    large and this function is executed from a jupyter notebook, you may have to force
    vispy to use the PyQT6 backend prior to calling this function, i.e. call
    >>> import vispy; vispy.use("pyqt6")
    in the first cell of the notebook.
    """
    import matplotlib.pyplot as plt

    rnno = rnno_fn(sys)

    X_3d = tree_utils.to_3d_if_2d(X, strict=True)
    # TODO
    # right now the key used in init can be hardcoded because
    # i am always using a state which is non-stochastic and
    # rather it is just zeros
    initial_params, state = rnno.init(
        jax.random.PRNGKey(
            1,
        ),
        X_3d,
    )

    if params is None:
        params = initial_params

    yhat = rnno.apply(params, tree_utils.add_batch_dim(state), X_3d)[0]
    yhat = tree_utils.to_2d_if_3d(yhat, strict=True)
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
        T = tree_utils.tree_shape(y) * sys.dt
        ts = jnp.arange(0.0, T, step=sys.dt)
        fig, axes = plt.subplots(len(yhat), 3, figsize=(10, 3 * len(yhat)))
        axes = np.atleast_2d(axes)
        for row, link_name in enumerate(yhat.keys()):
            euler_angles_hat = jnp.rad2deg(maths.quat_to_euler(yhat[link_name]))
            euler_angles = (
                jnp.rad2deg(maths.quat_to_euler(y[link_name]))
                if y is not None
                else None
            )

            for col, xyz in enumerate(["x", "y", "z"]):
                axis = axes[row, col]
                axis.plot(ts, euler_angles_hat[:, col], label="prediction")
                if euler_angles is not None:
                    axis.plot(ts, euler_angles[:, col], label="truth")
                axis.grid(True)
                axis.set_title(link_name + "_" + xyz)
                axis.set_xlabel("time [s]")
                axis.set_ylabel("euler angles [deg]")
                axis.legend()
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
            sys_render = sys_composer.inject_system(sys_xs, sys_newcolor, prefix="hat_")

            # `yhat` are child-to-parent transforms, but we need parent-to-child
            # this dictonary has now all links that don't connect to worldbody
            transform2hat_rot = jax.tree_map(
                lambda quat: x_xy.maths.quat_inv(quat), yhat
            )

            transform1, transform2 = sim2real.unzip_xs(
                sys, sim2real.match_xs(sys, xs, sys_xs)
            )

            # we add the missing links in transform2hat, links that connect to worldbody
            transform2hat = []
            for i, name in enumerate(sys.link_names):
                if name in transform2hat_rot:
                    transform2_name = x_xy.base.Transform.create(
                        rot=transform2hat_rot[name]
                    )
                else:
                    transform2_name = transform2.take(i, axis=1)
                transform2hat.append(transform2_name)

            # after transpose shape is (n_timesteps, n_links, ...)
            transform2hat = (
                transform2hat[0].batch(*transform2hat[1:]).transpose((1, 0, 2))
            )

            xshat = sim2real.zip_xs(sys, transform1, transform2hat)

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
            render_path,
            sys_render,
            xs_render,
            fps=25,
            show_pbar=True,
            verbose=verbose,
            **vispy_kwargs,
        )

    return yhat, metrics


def _geoms_replace_color(sys, color):
    geoms = [g.replace(color=color) for g in sys.geoms]
    return sys.replace(geoms=geoms)
