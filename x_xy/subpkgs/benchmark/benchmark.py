import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import tree_utils

import x_xy
from x_xy import maths
from x_xy.io.xml.from_xml import _load_xml
from x_xy.subpkgs import exp
from x_xy.subpkgs import sim2real
from x_xy.subpkgs import sys_composer
from x_xy.subpkgs.exp.exp import _id2xml
from x_xy.subpkgs.exp.exp import _relative_to_this_file

from .base import Filter

_dt = 0.01


def double_hinge_joint(
    filter: Filter,
    sparse_segments: list[str] = ["seg3"],
    rigid_imus: bool = True,
    joint_axes_from_sys: bool = False,
    warmup: int = 500,
    plot: bool = False,
    render: bool = False,
    render_kwargs: dict = dict(),
):
    sys = exp.load_sys("S_06", morph_yaml_key="seg2", delete_after_morph="seg5")
    return _S06_double_triple_hinge_joint(
        sys,
        sparse_segments,
        rigid_imus,
        joint_axes_from_sys,
        filter,
        warmup,
        plot,
        render,
        render_kwargs,
    )


def triple_hinge_joint(
    filter: Filter,
    sparse_segments: list[str] = ["seg2", "seg3"],
    rigid_imus: bool = True,
    joint_axes_from_sys: bool = False,
    warmup: int = 500,
    plot: bool = False,
    render: bool = False,
    render_kwargs: dict = dict(),
):
    sys = exp.load_sys("S_06", morph_yaml_key="seg5", delete_after_morph="seg1")
    return _S06_double_triple_hinge_joint(
        sys,
        sparse_segments,
        rigid_imus,
        joint_axes_from_sys,
        filter,
        warmup,
        plot,
        render,
        render_kwargs,
    )


def _S06_double_triple_hinge_joint(
    sys,
    sparse_segments: list[str],
    rigid: bool,
    from_sys: bool,
    filter: Filter,
    warmup: int,
    plot: bool,
    render: bool,
    render_kwargs: dict,
) -> dict:
    sys = sys_composer.make_sys_noimu(sys)[0]

    imu_key = "imu_rigid" if rigid else "imu_flex"

    motion_start = ["slow", "fast", "fast_slow_fast", "fast_slow_fast", "fast_slow"]
    motion_stop = ["slow", "fast", "fast_slow", "slow_end", "slow_end"]

    results = dict()
    for sta, sto in zip(motion_start, motion_stop):
        data = exp.load_data("S_06", sta, sto)

        xml_str = _load_xml(_relative_to_this_file(_id2xml["S_06"]))
        xs = sim2real.xs_from_raw(
            sys, exp.link_name_pos_rot_data(data, xml_str), qinv=True
        )
        X = x_xy.joint_axes(sys, xs, sys, from_sys=from_sys)
        if from_sys:
            X_xs = x_xy.joint_axes(sys, xs, sys, from_sys=False)
            for seg in X:
                if sys.link_parents[sys.name_to_idx(seg)] == -1:
                    X[seg].update(X_xs[seg])

        for seg in X:
            imu_data = data[seg][imu_key]
            imu_data.pop("mag")
            if seg in sparse_segments:
                imu_data = tree_utils.tree_zeros_like(imu_data)
            X[seg].update(imu_data)

        y = x_xy.rel_pose(sys, xs)

        filter.init(sys, tree_utils.tree_slice(X, 0))
        yhat = tree_utils.tree_slice(filter.predict(tree_utils.add_batch_dim(X)), 0)

        key = f"{sta}->{sto}"
        results[key] = dict()
        for seg in y:
            results[key][f"mae_deg_{seg}"] = jnp.mean(
                jnp.rad2deg(maths.angle_error(y[seg], yhat[seg]))[warmup:]
            )

        if plot:
            path = x_xy.utils.parse_path(
                f"~/xxy_benchmark/{filter.identifier()}/{key}.png"
            )
            _plot_3x3(path, y, yhat, results=results[key], dt=_dt)

        if render:
            path = x_xy.utils.parse_path(
                f"~/xxy_benchmark/{filter.identifier()}/{key}.mp4"
            )
            _render(path, sys, xs, yhat, **render_kwargs)

    return results


def _render(path, sys, xs, yhat, **kwargs):
    # replace render color of geoms for render of predicted motion
    prediction_color = (78 / 255, 163 / 255, 243 / 255, 1.0)
    sys_newcolor = _geoms_replace_color(sys, prediction_color)
    sys_render = sys_composer.inject_system(sys, sys_newcolor.add_prefix_suffix("hat_"))

    # `yhat` are child-to-parent transforms, but we need parent-to-child
    # this dictonary has now all links that don't connect to worldbody
    transform2hat_rot = jax.tree_map(lambda quat: maths.quat_inv(quat), yhat)

    transform1, transform2 = sim2real.unzip_xs(sys, xs)

    # we add the missing links in transform2hat, links that connect to worldbody
    transform2hat = []
    for i, name in enumerate(sys.link_names):
        if name in transform2hat_rot:
            transform2_name = x_xy.Transform.create(rot=transform2hat_rot[name])
        else:
            transform2_name = transform2.take(i, axis=1)
        transform2hat.append(transform2_name)

    # after transpose shape is (n_timesteps, n_links, ...)
    transform2hat = transform2hat[0].batch(*transform2hat[1:]).transpose((1, 0, 2))

    xshat = sim2real.zip_xs(sys, transform1, transform2hat)

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
                sys.link_names,
                [xs[i] for i in range(sys.num_links())],
            )
        )
    )

    xs_render = []
    for name in sys_render.link_names:
        xs_render.append(xs_dict[name])
    xs_render = xs_render[0].batch(*xs_render[1:])
    xs_render = xs_render.transpose((1, 0, 2))
    xs_render = [xs_render[t] for t in range(xs_render.shape())]

    frames = x_xy.render(
        sys_render, xs_render, camera="target", show_pbar=False, **kwargs
    )

    mediapy.write_video(path, frames)


def _geoms_replace_color(sys, color):
    geoms = [g.replace(color=color) for g in sys.geoms]
    return sys.replace(geoms=geoms)


def _plot_3x3(path, y, yhat, results: dict, dt: float):
    T = tree_utils.tree_shape(y) * dt
    ts = jnp.arange(0.0, T, step=dt)
    fig, axes = plt.subplots(len(yhat), 3, figsize=(10, 3 * len(yhat)))
    axes = np.atleast_2d(axes)
    for row, link_name in enumerate(yhat.keys()):
        euler_angles_hat = jnp.rad2deg(maths.quat_to_euler(yhat[link_name]))
        euler_angles = (
            jnp.rad2deg(maths.quat_to_euler(y[link_name])) if y is not None else None
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

    title = ""
    for seg, mae in results.items():
        title += "{}={:.2f}_".format(seg, mae)
    plt.title(title[:-1])

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
