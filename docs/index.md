# `x_xy` - Documentation Home

This is the documentation for the `x_xy` software library.

<p align="center">
<img src="img/icon.svg" height="200" />
</p>

## `x_xy_v2` -- A *tiny* Kinematic Tree Simulator
<img src="img/coverage_badge.svg" height="20" />

## Installation

Supports `Python=3.10` and `Python=3.11`.

Install with `pip` using

`pip install git+https://github.com/SimiPixel/x_xy_v2.git`

Additionally,
- `render.py` requires a vispy backend (e.g. `pip install pyqt6`).

    Note 1: On a headless node with a Nvidia GPU it works without any backend.

    Note 2: More info: https://vispy.org/installation.html

Typically, this will install `jax` as cpu-only version. CUDA version can be installed with
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Publications

The following publications utilize this software library, and refer to it as the *Random Chain Motion Generator (RCMG)* (more specifically the function `x_xy.build_generator`):

- [*RNN-based Observability Analysis for Magnetometer-Free Sparse Inertial Motion Tracking*](https://ieeexplore.ieee.org/document/9841375)
- [*Plug-and-Play Sparse Inertial Motion Tracking With Sim-to-Real Transfer*](https://ieeexplore.ieee.org/document/10225275)
- [*RNN-based State and Parameter Estimation for Sparse Magnetometer-free Inertial Motion Tracking*](https://www.journals.infinite-science.de/index.php/automed/article/view/745)

## Contact

Simon Bachhuber (simon.bachhuber@fau.de)
