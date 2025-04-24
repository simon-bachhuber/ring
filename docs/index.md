<p align="center">
<img src="https://raw.githubusercontent.com/simon-bachhuber/ring/main/docs/img/concept_v4.png" height="200" />
</p>


# Recurrent Inertial Graph-based Estimator (RING)
<img src="https://raw.githubusercontent.com/simon-bachhuber/ring/main/docs/img/coverage_badge.svg" height="20" />

RING provides a pluripotent, problem-unspecific plug-and-play IMT solution that, in contrast to conventional IMT solutions, eliminates the need for expert knowledge to identify, select, and parameterize the appropriate method. RING's pluripotency is enabled by a novel online-capable neural network architecture that uses a decentralized network of message-passing, parameter-sharing recurrent neural networks, which map local IMU measurements and nearest-neighbour messages to local orientations. This architecture enables RING to address a broad range of IMT problems that vary greatly in aspects such as the number of attached sensors, or the number of segments in the kinematic chain, and even generalize to previously unsolved IMT problems, including the challenging combination of magnetometer-free and sparse sensing with unknown sensor-to-segment parameters. Remarkably, RING is trained solely on simulated data, yet evaluated on experimental data, which indicates its exceptional ability to zero-shot generalize from simulation to experiment, while outperforming several state-of-the-art problem-specific solutions. For example, RING can, for the first time, accurately track a four-segment kinematic chain (which requires estimating four orientations) using only two magnetometer-free inertial measurement units.

> **ℹ️ Tip:**
> 
> Check out my new plug-and-play interface for inertial motion tracking (RING included) [here](https://github.com/simon-bachhuber/imt.git).

## Installation

Supports `Python=3.10/3.11/3.12` (tested).

Install with `pip` using

`pip install imt-ring`

Typically, this will install `jax` as cpu-only version. For GPU install instructions for `jax` see https://github.com/jax-ml/jax?tab=readme-ov-file#instructions.

## Documentation

Available [here](https://simon-bachhuber.github.io/ring/).

## Quickstart Example
```python
import ring
import numpy as np

T  : int       = 30         # sequence length     [s]
Ts : float     = 0.01       # sampling interval   [s]
B  : int       = 1          # batch size
lam: list[int] = [-1, 0, 1] # parent array
N  : int       = len(lam)   # number of bodies
T_i: int       = int(T/Ts)  # number of timesteps

X              = np.zeros((B, T_i, N, 9))
# where X is structured as follows:
# X[..., :3]   = acc
# X[..., 3:6]  = gyr
# X[..., 6:9]  = jointaxis

# let's assume we have an IMU on each outer segment of the
# three-segment kinematic chain
X[..., 0, :3]  = acc_segment1
X[..., 2, :3]  = acc_segment3
X[..., 0, 3:6] = gyr_segment1
X[..., 2, 3:6] = gyr_segment3

ringnet = ring.RING(lam, Ts)
yhat, _ = ringnet.apply(X)
# yhat: unit quaternions, shape = (B, T_i, N, 4)
```

### Known fixes

#### Offscreen rendering with Mujoco

> mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called

Solution:

```python
import os
os.environ["MUJOCO_GL"] = "egl"
```

#### Windows-related: ImportError: DLL load failed while importing ...

> ImportError: DLL load failed while importing _multiarray_umath: Das angegebene Modul wurde nicht gefunden.

Solution:
1. `pip uninstall -y jax jaxlib`
2. `conda install -c conda-forge jax`

## Publications

The main publication is:

- [*Recurrent Inertial Graph-Based Estimator (RING): A Single Pluripotent Inertial Motion Tracking Solution*](https://openreview.net/pdf?id=h2C3rkn0zR)

The following publications also utilize this software library, and refer to it as the *Random Chain Motion Generator (RCMG)* (more specifically the function `ring.RCMG`):

- [*RNN-based Observability Analysis for Magnetometer-Free Sparse Inertial Motion Tracking*](https://ieeexplore.ieee.org/document/9841375)
- [*Plug-and-Play Sparse Inertial Motion Tracking With Sim-to-Real Transfer*](https://ieeexplore.ieee.org/document/10225275)
- [*RNN-based State and Parameter Estimation for Sparse Magnetometer-free Inertial Motion Tracking*](https://www.journals.infinite-science.de/index.php/automed/article/view/745)

### Other useful ressources

Particularly useful is the following publication from *Roy Featherstone*
- [*A Beginner’s Guide to 6-D Vectors (Part 2)*](https://ieeexplore.ieee.org/document/5663690)

## Contact

Simon Bachhuber (simon.bachhuber@fau.de)

### How to bump verion in this python package

1) commit and *push* your code changes. Make sure to also update the version in `pyproject.toml`
2) create the tag and push the tag