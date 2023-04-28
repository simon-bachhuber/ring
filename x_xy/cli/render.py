import argparse

import jax
import jax.numpy as jnp

import x_xy


def _parse_xml(p: argparse.ArgumentParser):
    p.add_argument("path_xml", help="Path of the xml-file that defines the system.")


def _simulate(
    p: argparse.ArgumentParser,
):
    p.add_argument("path_video")
    p.add_argument("--format", default="mp4")
    p.add_argument("T", help="Duration of simulation", type=float)
    p.add_argument(
        "--rcmg",
        action="store_true",
        help="Use the `RCMG` during the simulation and"
        " does not dynamically simulate the system.",
    )

    p.add_argument("--rcmg-t-min", default=0.15)
    p.add_argument("--rcmg-t-max", default=0.75)
    p.add_argument("--rcmg-dang-min", default=0.0)
    p.add_argument("--rcmg-dang-max", default=120.0)
    p.add_argument("--rcmg-dang-min-free", default=0.0)
    p.add_argument("--rcmg-dang-max-free", default=60)
    p.add_argument("--rcmg-dpos-min", default=0.001)
    p.add_argument("--rcmg-dpos-max", default=0.2)
    p.add_argument("--rcmg-pos-min", default=-2.5)
    p.add_argument("--rcmg-pos-max", default=2.5)
    p.add_argument(
        "--rcmg-rand-interp", action="store_true", help="Randomize interpolation."
    )
    p.add_argument(
        "--rcmg-rom-hinge",
        action="store_true",
        help="Restrict hinge joints to not pass through the 180° point.",
    )
    p.add_argument("--rcmg-rom-hinge-method", default="uniform")
    p.add_argument("--rcmg-seed", default=1)

    p.add_argument(
        "--dyn-q",
        metavar="DOF",
        nargs="+",
        type=float,
        help="Configuration vector `q` of size equal to DOFs where `free`"
        " and `spherical` joint types are of size 4 (due to encoding as quaternion).",
        default=[],
    )

    p.add_argument(
        "--dyn-qd",
        metavar="DOF",
        nargs="+",
        type=float,
        help="Configuration velocity vector `qd` of size equal to DOFs.",
        default=[],
    )


def main():
    x_xy.utils.disable_jit_warn()

    p = argparse.ArgumentParser(
        "cli-render", description="Simulates and renders a system."
    )
    _parse_xml(p)
    _simulate(p)

    p = p.parse_args()

    sys = x_xy.io.load_sys_from_xml(p.path_xml)

    if p.rcmg:
        d2r = lambda degree: float(jnp.deg2rad(degree))
        config = x_xy.algorithms.RCMG_Config(
            p.T,
            sys.dt,
            p.rcmg_t_min,
            p.rcmg_t_max,
            d2r(p.rcmg_dang_min),
            d2r(p.rcmg_dang_max),
            d2r(p.rcmg_dang_min_free),
            d2r(p.rcmg_dang_max_free),
            p.rcmg_dpos_min,
            p.rcmg_dpos_max,
            p.rcmg_pos_min,
            p.rcmg_pos_max,
            p.rcmg_rand_interp,
            p.rcmg_rom_hinge,
            p.rcmg_rom_hinge_method,
        )
        _, xs = x_xy.algorithms.build_generator(sys, config)(
            jax.random.PRNGKey(
                p.rcmg_seed,
            )
        )
    else:
        q = jnp.zeros((sys.q_size())) if len(p.dyn_q) == 0 else jnp.array(p.dyn_q)
        qd = jnp.zeros((sys.qd_size())) if len(p.dyn_qd) == 0 else jnp.array(p.dyn_qd)

        state = x_xy.base.State.create(sys, q, qd)
        xs = []
        tau = jnp.zeros_like(state.qd)
        for _ in range(int(p.T / sys.dt)):
            state = jax.jit(x_xy.algorithms.step)(sys, state, tau)
            xs.append(state.x)
        xs = xs[0].batch(*xs[1:])

    scene = x_xy.render.VispyScene(sys.geoms)
    x_xy.render.animate(p.path_video, scene, xs, sys.dt, fmt=p.format)


if __name__ == "__main__":
    main()
