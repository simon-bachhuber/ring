from functools import cache

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import qmt

import x_xy
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml
from x_xy.subpkgs import sys_composer


@cache
def _get_system(exp_id, seg_femur, seg_tibia):
    sys_render = exp.load_sys(exp_id, morph_yaml_key=seg_femur)
    chain = sys_render.findall_segments()
    delete = list(set(chain) - set([seg_femur, seg_tibia]))
    sys_render = sys_composer.delete_subsystem(sys_render, delete, strict=False)
    return sys_render


def _error_fn(y, yhat, warmup: int):
    ae_deg = jax.tree_map(
        lambda q, qhat: jnp.rad2deg(x_xy.maths.angle_error(q, qhat)), y, yhat
    )
    return {
        "mae_deg": jax.tree_map(lambda arr: jnp.mean(arr[warmup:]), ae_deg),
        "rmse_deg": jax.tree_map(
            lambda arr: jnp.sqrt(jnp.mean(arr[warmup:] ** 2)), ae_deg
        ),
    }


def single_hinge_joint(
    exp_id: str,
    motion_start: str,
    motion_stop: str | None,
    filter: ml.AbstractFilter,
    seg_femur="seg3",
    seg_tibia="seg4",
    flex=False,
    render: bool = False,
    ja: bool = False,
    warmup: int = 500,
):
    mag = True
    if isinstance(filter, ml.InitApplyFnFilter):
        mag = False

    sys = _get_system(exp_id, seg_femur, seg_tibia)
    sys_noimu = sys_composer.make_sys_noimu(sys)[0]

    X, y, xs = ml.convenient.pipeline_load_data(
        sys=sys,
        exp_id=exp_id,
        motion_start=motion_start,
        motion_stop=motion_stop,
        flex=flex,
        mag=mag,
        jointaxes=ja,
        rootincl=False,
        rootfull=True,
    )

    yhat = filter.predict(X, sys_noimu)
    yhat[seg_femur] = x_xy.maths.quat_transfer_heading(y[seg_femur], yhat[seg_femur])

    errors = _error_fn(y, yhat, warmup)

    if render:
        yhat_render = x_xy.utils.pytree_deepcopy(yhat)
        yhat_render[seg_femur] = x_xy.maths.quat_inv(yhat_render[seg_femur])
        frames = x_xy.render_prediction(
            sys,
            xs,
            yhat_render,
            stepframe=4,
            width=1280,
            height=720,
            camera="target",
            transparent_segment_to_root=False,
            show_floor=False,
        )
        key = (
            f"{exp_id}_{motion_start}_{motion_stop}_{seg_femur}_{seg_tibia}_"
            + f"flex_{int(flex)}_ja"
        )
        path = x_xy.utils.parse_path(f"~/xxy_benchmark/{filter.name}/{key}.mp4")
        media.write_video(path, frames, fps=25)

    y_euler, yhat_euler = jax.tree_map(
        lambda q: np.rad2deg(qmt.eulerAngles(q)), (y, yhat)
    )
    return y, yhat, y_euler, yhat_euler, errors
