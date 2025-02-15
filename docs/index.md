<p align="center">
<img src="https://raw.githubusercontent.com/simon-bachhuber/ring/main/docs/img/icon.svg" height="200" />
</p>

# Recurrent Inertial Graph-based Estimator (RING)
<img src="https://raw.githubusercontent.com/simon-bachhuber/ring/main/docs/img/coverage_badge.svg" height="20" />

> **ℹ️ Tip:**
> 
> Check out my new plug-and-play interface for inertial motion tracking (RING included) [here](https://github.com/simon-bachhuber/imt.git).

## Installation

Supports `Python=3.10/3.11/3.12` (tested).

Install with `pip` using

`pip install imt-ring`

Typically, this will install `jax` as cpu-only version. Afterwards, gpu-enabled version can be installed with
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Documentation

Available [here](https://simipixel.github.io/ring/).

## Quickstart Example
```python
import ring
import numpy as np

T  : int       = 30        # sequence length     [s]
Ts : float     = 0.01      # sampling interval   [s]
B  : int       = 1         # batch size
lam: list[int] = [0, 1, 2] # parent array
N  : int       = len(lam)  # number of bodies
T_i: int       = int(T/Ts) # number of timesteps

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

The following publications utilize this software library, and refer to it as the *Random Chain Motion Generator (RCMG)* (more specifically the function `ring.RCMG`):

- [*RNN-based Observability Analysis for Magnetometer-Free Sparse Inertial Motion Tracking*](https://ieeexplore.ieee.org/document/9841375)
- [*Plug-and-Play Sparse Inertial Motion Tracking With Sim-to-Real Transfer*](https://ieeexplore.ieee.org/document/10225275)
- [*RNN-based State and Parameter Estimation for Sparse Magnetometer-free Inertial Motion Tracking*](https://www.journals.infinite-science.de/index.php/automed/article/view/745)

### Other useful ressources

Particularly useful is the following publication from *Roy Featherstone*
- [*A Beginner’s Guide to 6-D Vectors (Part 2)*](https://ieeexplore.ieee.org/document/5663690)

## Contact

Simon Bachhuber (simon.bachhuber@fau.de)
