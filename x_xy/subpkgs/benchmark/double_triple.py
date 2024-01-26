import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import tree_utils

import x_xy
from x_xy import maths
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml
from x_xy.subpkgs import sim2real
from x_xy.subpkgs import sys_composer

Filter = ml.AbstractFilter


def double_hinge_joint(
    filter: Filter,
    exp_id,
    motion_start,
    motion_stop,
    delete_imus: list[str] = ["imu3"],
    rigid_imus: bool = True,
    warmup: int = 500,
    plot: bool = False,
    render: bool = False,
    render_kwargs: dict = dict(),
    debug: bool = False,
    ja: bool = True,
    attitude: bool = False,
):
    sys = exp.load_sys(
        exp_id, morph_yaml_key="seg2", delete_after_morph=["seg5"] + delete_imus
    )
    return _double_triple_hinge_joint(
        sys,
        exp_id,
        motion_start,
        motion_stop,
        rigid_imus,
        filter,
        warmup,
        plot,
        render,
        render_kwargs,
        debug,
        ja,
        attitude,
    )


def triple_hinge_joint(
    filter: Filter,
    exp_id,
    motion_start,
    motion_stop,
    delete_imus: list[str] = ["imu2", "imu3"],
    rigid_imus: bool = True,
    warmup: int = 500,
    plot: bool = False,
    render: bool = False,
    render_kwargs: dict = dict(),
    debug: bool = False,
    ja: bool = True,
    attitude: bool = False,
):
    sys = exp.load_sys(
        exp_id, morph_yaml_key="seg5", delete_after_morph=["seg1"] + delete_imus
    )
    return _double_triple_hinge_joint(
        sys,
        exp_id,
        motion_start,
        motion_stop,
        rigid_imus,
        filter,
        warmup,
        plot,
        render,
        render_kwargs,
        debug,
        ja,
        attitude,
    )


def _double_triple_hinge_joint(
    sys,
    exp_id,
    motion_start,
    motion_stop,
    rigid: bool,
    filter: Filter,
    warmup: int,
    plot: bool,
    render: bool,
    render_kwargs: dict,
    debug: bool,
    ja: bool,
    attitude: bool,
) -> dict:
    debug_dict = dict()

    X, y, xs = ml.convenient.pipeline_load_data(
        sys, exp_id, motion_start, motion_stop, not rigid, False, ja, attitude, False
    )
    yhat = filter.predict(X, sys_composer.make_sys_noimu(sys)[0])

    if debug:
        print(f"_double_triple_hinge_joint: `y.keys()`={list(y.keys())}")
        print(f"_double_triple_hinge_joint: `yhat.keys()`={list(yhat.keys())}")

    if not attitude:
        for name in yhat:
            if name not in y:
                yhat.pop(name)

    key = (
        f"{motion_start}->{str(motion_stop)}_rigid_{int(rigid)}_ja_{int(ja)}_att_"
        f"{int(attitude)}"
    )
    results = dict()
    _results = {key: results}
    for seg in y:
        results[f"mae_deg_{seg}"] = jnp.mean(
            jnp.rad2deg(maths.angle_error(y[seg], yhat[seg]))[warmup:]
        )

    if plot:
        path = x_xy.utils.parse_path(f"~/xxy_benchmark/{filter.name}/{key}.png")
        _plot_3x3(path, y, yhat, results=results, dt=float(sys.dt))

    if render:
        path = x_xy.utils.parse_path(f"~/xxy_benchmark/{filter.name}/{key}.mp4")
        _render(path, sys, xs, yhat, debug, **render_kwargs)

    if debug:
        return _results, debug_dict
    else:
        return _results


def _render(path, sys, xs, yhat, debug, **kwargs):
    offset_truth = kwargs.pop("offset_truth", [0, 0, 0])
    offset_pred = kwargs.pop("offset_pred", [0, 0, 0])
    assert sys.dt == 0.01

    sys_noimu = sys_composer.make_sys_noimu(sys)[0]
    xs_noimu = sim2real.match_xs(sys_noimu, xs, sys)

    seg_to_root = sys.link_names[sys.link_parents.index(-1)]
    attitude = seg_to_root in yhat

    if debug:
        print(f"_render: `sys.link_parents`={sys.link_parents}")
        print(f"_render: `sys.link_types`={sys.link_types}")
        print(f"_render: `attitude`={attitude}")
        print(f"_render: `seg_to_root`={seg_to_root}")

    # `yhat` are child-to-parent transforms, but we need parent-to-child
    transform2hat_rot = jax.tree_map(lambda quat: maths.quat_inv(quat), yhat)
    # well apart from the to_root connection that one was already parent-to-child
    if attitude:
        transform2hat_rot[seg_to_root] = maths.quat_inv(transform2hat_rot[seg_to_root])

    transform1, transform2 = sim2real.unzip_xs(sys_noimu, xs_noimu)

    # if not attitude:
    #   we add the missing links in transform2hat, links that connect to worldbody
    # if attitude:
    #   we add the truth heading
    transform2hat = []
    for i, name in enumerate(sys_noimu.link_names):
        rot_truth = transform2.take(i, axis=1).rot

        if name in transform2hat_rot:
            rot_hat = transform2hat_rot[name]
            if name == seg_to_root:
                rot_hat = maths.quat_transfer_heading(rot_truth, rot_hat)
            rot_i = rot_hat
        else:
            rot_i = rot_truth
        transform2hat.append(x_xy.Transform.create(rot=rot_i))

    # after transpose shape is (n_timesteps, n_links, ...)
    transform2hat = transform2hat[0].batch(*transform2hat[1:]).transpose((1, 0, 2))

    xshat_noimu = sim2real.zip_xs(sys_noimu, transform1, transform2hat)

    add_offset = lambda x, offset: x_xy.transform_mul(
        x, x_xy.Transform.create(pos=jnp.array(offset, dtype=jnp.float32))
    )

    # swap time axis, and link axis
    xs, xshat_noimu = xs.transpose((1, 0, 2)), xshat_noimu.transpose((1, 0, 2))
    # create mapping from `name` -> Transform
    xs_dict = dict(
        zip(
            ["hat_" + name for name in sys_noimu.link_names],
            [
                add_offset(xshat_noimu[i], offset_pred)
                for i in range(sys_noimu.num_links())
            ],
        )
    )

    xs_dict.update(
        dict(
            zip(
                sys.link_names,
                [add_offset(xs[i], offset_truth) for i in range(sys.num_links())],
            )
        )
    )

    # replace render color of geoms for render of predicted motion
    prediction_color = (78 / 255, 163 / 255, 243 / 255, 1.0)
    sys_newcolor = _geoms_replace_color(sys_noimu, prediction_color)
    sys_render = sys_composer.inject_system(sys_newcolor.add_prefix_suffix("hat_"), sys)

    xs_render = []
    for name in sys_render.link_names:
        xs_render.append(xs_dict[name])
    xs_render = xs_render[0].batch(*xs_render[1:])
    xs_render = xs_render.transpose((1, 0, 2))
    xs_render = [xs_render[t] for t in range(xs_render.shape())]

    frames = x_xy.render(
        sys_render,
        xs_render,
        camera="target",
        show_pbar=False,
        render_every_nth=2,
        **kwargs,
    )

    mediapy.write_video(path, frames, fps=50)


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
