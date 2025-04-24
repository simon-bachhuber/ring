"""This module exports a loss function `loss_fn` for training neural networks that
output quaternions in PyTorch"""

from typing import Sequence

import torch


def quat_mul(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    "Multiplies two quaternions."
    q = torch.stack(
        [
            u[..., 0] * v[..., 0]
            - u[..., 1] * v[..., 1]
            - u[..., 2] * v[..., 2]
            - u[..., 3] * v[..., 3],
            u[..., 0] * v[..., 1]
            + u[..., 1] * v[..., 0]
            + u[..., 2] * v[..., 3]
            - u[..., 3] * v[..., 2],
            u[..., 0] * v[..., 2]
            - u[..., 1] * v[..., 3]
            + u[..., 2] * v[..., 0]
            + u[..., 3] * v[..., 1],
            u[..., 0] * v[..., 3]
            + u[..., 1] * v[..., 2]
            - u[..., 2] * v[..., 1]
            + u[..., 3] * v[..., 0],
        ],
        dim=-1,
    )
    return q


def quat_inv(q: torch.Tensor):
    return torch.concat([q[..., :1], -q[..., 1:]], dim=-1)


def wrap_to_pi(phi):
    "Wraps angle `phi` (radians) to interval [-pi, pi]."
    return (phi + torch.pi) % (2 * torch.pi) - torch.pi


def quat_angle(q: torch.Tensor):
    phi = 2 * torch.arctan2(torch.norm(q[..., 1:], dim=-1), q[..., 0])
    return wrap_to_pi(phi)


def safe_normalize(x):
    return x / (1e-6 + torch.norm(x, dim=-1, keepdim=True))


def quat_qrel(q1, q2):
    "q1^-1 * q2"
    return quat_mul(quat_inv(q1), q2)


@torch.jit.script
def angle_error(q, qhat):
    "Absolute angle error in radians"
    return torch.abs(quat_angle(quat_qrel(q, qhat)))


@torch.jit.script
def inclination_error(q, qhat):
    "Absolute inclination error in radians. `q`s are from body-to-eps"
    q_rel = quat_mul(q, quat_inv(qhat))
    phi_pri = 2 * torch.arctan2(q_rel[..., 3], q_rel[..., 0])
    q_pri = torch.zeros_like(q)
    q_pri[..., 0] = torch.cos(phi_pri / 2)
    q_pri[..., 3] = torch.sin(phi_pri / 2)
    q_res = quat_mul(q_rel, quat_inv(q_pri))
    return torch.abs(quat_angle(q_res))


def loss_fn(lam: Sequence[int], q: torch.Tensor, qhat: torch.Tensor) -> torch.Tensor:
    "(..., N, 4) -> (..., N)"
    *batch_dims, N, F = q.shape
    assert q.shape == qhat.shape
    assert F == 4
    assert N == len(lam)
    permu = list(reversed(range(q.ndim - 1)))
    loss_incl = inclination_error(q, qhat).permute(*permu)
    loss_mae = angle_error(q, qhat).permute(*permu)
    lam = torch.tensor(lam, device=q.device)
    return torch.where(
        lam.reshape(-1, *[1] * len(batch_dims)) == -1, loss_incl, loss_mae
    ).permute(*permu)


def quat_rand(*size: tuple[int]):
    qs = torch.randn(size=size + (4,))
    return qs / torch.norm(qs, dim=-1, keepdim=True)
