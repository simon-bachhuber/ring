from dataclasses import dataclass
from dataclasses import replace
from functools import cache
from typing import Optional

import jax.numpy as jnp
import numpy as np

from x_xy import algorithms
from x_xy import base
from x_xy import exp
from x_xy import maths
from x_xy import ml
from x_xy import sim2real
from x_xy import utils


@cache
def _get_sys(exp_id, anchor: str, include_links: tuple[str]):
    sys = exp.load_sys(exp_id).morph_system(new_anchor=anchor)
    delete = list(set(sys.link_names) - set(include_links))
    return sys.delete_system(delete, strict=False)


def _max_coords_after_omc_pos_offset(sys: base.System, data: dict) -> dict:

    data_out = dict()
    for link_name, max_cord in zip(sys.link_names, sys.omc):
        if max_cord is None:
            continue
        cs_name, marker, pos_offset = (
            max_cord.coordinate_system_name,
            max_cord.pos_marker_number,
            max_cord.pos_marker_constant_offset,
        )
        pos = data[cs_name][f"marker{marker}"]
        quat = data[cs_name]["quat"]
        pos_with_offset = pos + maths.rotate(pos_offset, quat)
        data_out[link_name] = dict(pos=pos_with_offset, quat=quat)

    return data_out


@dataclass
class IMTP:
    segments: list[str]
    mag: bool = False
    flex: bool = False
    sparse: bool = False
    joint_axes: bool = False
    joint_axes_field: bool = True
    hz: float = 100.0
    dt: bool = True

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def imus(self) -> list[str]:
        segs = self.segments
        if self.sparse:
            segs = segs[0:1] + [segs[-1]]
        return [f"imu{seg[-1]}" for seg in segs]

    def sys(self, exp_id: str):
        include_links = self.segments + self.imus()
        sys = _get_sys(exp_id, self.segments[0], tuple(include_links))
        sys = sys.change_model_name(suffix=f"_{len(self.segments)}Seg")
        return sys

    def sys_noimu(self, exp_id: str):
        sys_noimu, imu_attachment = self.sys(exp_id).make_sys_noimu()
        assert sys_noimu.link_parents == self.lam
        assert sys_noimu.link_names == self.segments
        return sys_noimu, imu_attachment

    @property
    def lam(self):
        return list(range(-1, len(self.segments) - 1))

    @property
    def N(self):
        return len(self.segments)

    @property
    def F(self):
        F = 6
        if self.mag:
            F += 3
        if self.joint_axes_field:
            F += 3
        if self.dt:
            F += 1
        return F

    def name(self, exp_id: str, motion_start: str, motion_stop: Optional[str] = None):
        if motion_stop is None:
            motion_stop = ""
        model_name = self.sys(exp_id).model_name
        flex, mag, ja = int(self.flex), int(self.mag), int(self.joint_axes)
        return (
            f"{model_name}_{exp_id}_{motion_start}_{motion_stop}_flex_{flex}_"
            + f"mag_{mag}_ja_{ja}"
        )


def _build_Xy_xs_xsnoimu(
    exp_id: str, motion_start: str, motion_stop: str | None, imtp: IMTP
) -> tuple[np.ndarray, np.ndarray, base.Transform, base.Transform]:

    data = exp.load_data(exp_id, motion_start, motion_stop, resample_to_hz=imtp.hz)
    sys = imtp.sys(exp_id)
    sys_noimu, imu_attachment = imtp.sys_noimu(exp_id)

    max_coords = _max_coords_after_omc_pos_offset(sys, data)
    xs = sim2real.xs_from_raw(
        sys,
        max_coords,
        qinv=True,
    )
    xs_noimu = sim2real.xs_from_raw(
        sys_noimu,
        max_coords,
        qinv=True,
    )

    T = xs.shape()
    N, F = imtp.N, imtp.F

    X, y = np.zeros((T, N, F)), np.zeros((T, N, 4))

    imu_key = "imu_flex" if imtp.flex else "imu_rigid"
    for i, seg in enumerate(imtp.segments):
        if seg in list(imu_attachment.values()):
            X_seg = data[seg][imu_key]
            X[:, i, :3] = X_seg["acc"]
            X[:, i, 3:6] = X_seg["gyr"]
            if imtp.mag:
                X[:, i, 6:9] = X_seg["acc"]

    if imtp.joint_axes:
        X_joint_axes = algorithms.joint_axes(sys_noimu, xs, sys)
        for i, seg in enumerate(imtp.segments):
            F_i = 9 if imtp.mag else 6
            X[:, i, F_i : (F_i + 3)] = X_joint_axes[seg]["joint_axes"]

    if imtp.dt:
        repeated_dt = np.repeat(np.array([[1 / imtp.hz]]), T, axis=0)
        for i, seg in enumerate(imtp.segments):
            X[:, i, (F - 1) : F] = repeated_dt

    y_dict = algorithms.rel_pose(sys_noimu, xs, sys)
    y_rootfull = algorithms.sensors.root_full(sys_noimu, xs, sys)
    y_dict = utils.dict_union(y_dict, y_rootfull)
    for i, seg in enumerate(imtp.segments):
        y[:, i] = y_dict[seg]

    return X, y, xs, xs_noimu


_mae_metrices = dict(
    mae_deg=lambda q, qhat: jnp.rad2deg(jnp.mean(maths.angle_error(q, qhat)[:, 2500:]))
)


def benchmark(
    filter: ml.AbstractFilter,
    imtp: IMTP,
    exp_id: str,
    motion_start: str,
    motion_stop: Optional[str] = None,
    warmup: float = 0.0,
    return_cb: bool = False,
):

    X, y, xs, xs_noimu = _build_Xy_xs_xsnoimu(exp_id, motion_start, motion_stop, imtp)

    if return_cb:
        return ml.callbacks.EvalXyTrainingLoopCallback(
            filter,
            _mae_metrices,
            X,
            y,
            imtp.lam,
            imtp.name(exp_id, motion_start, motion_stop),
            link_names=imtp.segments,
        )

    yhat, _ = filter.apply(X=X, y=y, lam=tuple(imtp.lam))

    errors = dict()
    for i, seg in enumerate(imtp.segments):
        ae = np.rad2deg(maths.angle_error(y[:, i], yhat[:, i])[warmup:])
        errors[seg] = {}
        errors[seg]["mae"] = np.mean(ae)
        errors[seg]["std"] = np.std(ae)

    return errors, X, y, yhat, xs, xs_noimu
