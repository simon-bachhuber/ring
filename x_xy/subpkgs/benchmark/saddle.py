import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import qmt
import tree_utils

import x_xy
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml

from .single import _error_fn

Filter = ml.AbstractFilter

_sys = """
<x_xy>
    <worldbody>
        <body name="seg5" joint="free">
            <body name="aux" joint="px">
                <body name="outer" joint="px">
                </body>
            </body>
        </body>
    </worldbody>
</x_xy>
"""


def saddle(
    filter: Filter,
    exp_id,
    motion_start,
    motion_stop,
    left_or_right: str = "left",
    flex: bool = False,
    warmup: int = 500,
    render: bool = False,
    render_kwargs: dict = dict(),
    ja_inner: list[float] = None,
    ja_outer: list[float] = None,
    model_as_1DOF: bool = False,
    factory: bool = False,
    dt: bool = False,
):
    assert exp.load_arm_or_gait(exp_id) == "gait"

    delete, outer = {
        "left": (["seg4", "seg2"], "seg1"),
        "right": (["seg1", "seg3"], "seg2"),
    }[left_or_right]

    sys_exp = exp.load_sys(
        exp_id, morph_yaml_key="seg5", delete_after_morph=tuple(delete)
    )

    if not model_as_1DOF:
        sys_noimu = x_xy.io.load_sys_from_str(_sys).change_link_name("outer", outer)
    else:
        sys_noimu = sys_exp.make_sys_noimu()[0]

    mag = True
    if isinstance(filter, ml.InitApplyFnFilter):
        mag = False

    X, y, xs = ml.convenient.pipeline_load_data(
        sys=sys_exp,
        exp_id=exp_id,
        motion_start=motion_start,
        motion_stop=motion_stop,
        flex=flex,
        mag=mag,
        jointaxes=False,
        rootincl=True,
        rootfull=False,
        dt=dt,
    )

    if not model_as_1DOF:
        N = xs.shape()
        X["aux"] = tree_utils.tree_zeros_like(X[outer])

        if dt:
            X["aux"]["dt"] = X[outer]["dt"]

        ja_inner = jnp.zeros((3,)) if ja_inner is None else jnp.array(ja_inner)
        ja_outer = jnp.zeros((3,)) if ja_outer is None else jnp.array(ja_outer)
        X["aux"]["joint_axes"] = jnp.repeat(ja_inner[None], N, axis=0)
        X[outer]["joint_axes"] = jnp.repeat(ja_outer[None], N, axis=0)

    filter.set_sys(sys_noimu)

    def _params_to_errors_yhat(params):
        yhat = filter.predict(X, sys=None, params=params)

        if not model_as_1DOF:
            q_outer_to_aux = yhat.pop(outer)
            q_aux_to_seg5 = yhat.pop("aux")
            yhat[outer] = x_xy.transform_mul(
                x_xy.Transform.create(rot=q_aux_to_seg5),
                x_xy.Transform.create(rot=q_outer_to_aux),
            ).rot

        yhat["seg5"] = x_xy.maths.quat_transfer_heading(y["seg5"], yhat["seg5"])

        errors = _error_fn(y, yhat, warmup)
        if factory:
            errors.pop("rmse_deg")
            return errors
        return errors, yhat

    if factory:
        return _params_to_errors_yhat

    errors, yhat = _params_to_errors_yhat(filter.params)

    if render:
        yhat_render = x_xy.utils.pytree_deepcopy(yhat)
        yhat_render["seg5"] = x_xy.maths.quat_inv(yhat_render["seg5"])

        frames = sys_exp.render_prediction(
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
            f"saddle_{exp_id}_{motion_start}_{motion_stop}_{left_or_right}_"
            + f"flex_{int(flex)}"
        )
        path = x_xy.utils.parse_path(f"~/xxy_benchmark/{filter.name}/{key}.mp4")
        media.write_video(path, frames, fps=25)

    y_euler, yhat_euler = jax.tree_map(
        lambda q: np.rad2deg(qmt.eulerAngles(q)), (y, yhat)
    )
    return y, yhat, y_euler, yhat_euler, errors
