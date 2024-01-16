from functools import cache

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import qmt
import tree_utils

import x_xy
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml
from x_xy.subpkgs import sys_composer


def _second_segment(chain: list[str], seg: str) -> str:
    assert len(chain) > 1

    seg_i = chain.index(seg)
    if seg_i == (len(chain) - 1):
        return chain[seg_i - 1]
    else:
        return chain[seg_i + 1]


@cache
def _get_system(exp_id, seg):
    sys_render = exp.load_sys(exp_id, morph_yaml_key=seg)
    chain = sys_render.findall_segments()
    delete = list(set(chain) - set([seg, _second_segment(chain, seg)]))
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


def attitude(
    exp_id: str,
    motion_start: str,
    motion_stop: str | None,
    filter: ml.AbstractFilter,
    seg="seg3",
    flex=False,
    render: bool = False,
    warmup: int = 500,
):
    mag = True
    if isinstance(filter, ml.InitApplyFnFilter):
        mag = False

    sys = _get_system(exp_id, seg)
    second_seg = sys.findall_segments()[1]
    sys_noimu = sys_composer.make_sys_noimu(sys)[0]

    X, y, _ = ml.convenient.pipeline_load_data(
        sys=sys,
        exp_id=exp_id,
        motion_start=motion_start,
        motion_stop=motion_stop,
        flex=flex,
        mag=mag,
        jointaxes=False,
        rootincl=False,
        rootfull=True,
    )
    y.pop(second_seg)

    sys_render = sys_composer.delete_subsystem(sys, second_seg)
    *_, xs_render = ml.convenient.pipeline_load_data(
        sys=sys,
        exp_id=exp_id,
        motion_start=motion_start,
        motion_stop=motion_stop,
        flex=flex,
        mag=mag,
        jointaxes=False,
        rootincl=False,
        rootfull=True,
    )

    X[second_seg] = tree_utils.tree_zeros_like(X[second_seg])
    # let joint-axes just be a unit x-axis
    X[second_seg]["joint_axes"] = X[second_seg]["joint_axes"].at[:, 0].set(1.0)

    yhat = filter.predict(X, sys_noimu)
    yhat.pop(second_seg)

    yhat[seg] = x_xy.maths.quat_transfer_heading(y[seg], yhat[seg])

    errors = _error_fn(y, yhat, warmup)

    if render:
        yhat_render = x_xy.utils.pytree_deepcopy(yhat)
        yhat_render[seg] = x_xy.maths.quat_inv(yhat_render[seg])
        frames = x_xy.render_prediction(
            sys_render,
            xs_render,
            yhat_render,
            stepframe=4,
            width=1280,
            height=720,
            camera="target",
            transparent_segment_to_root=False,
            show_floor=False,
        )
        key = f"{exp_id}_{motion_start}_{motion_stop}_{seg}_" + f"flex_{int(flex)}_ja"
        path = x_xy.utils.parse_path(f"~/xxy_benchmark/{filter.name}/{key}.mp4")
        media.write_video(path, frames, fps=25)

    y_euler, yhat_euler = jax.tree_map(
        lambda q: np.rad2deg(qmt.eulerAngles(q)), (y, yhat)
    )
    return y, yhat, y_euler, yhat_euler, errors
