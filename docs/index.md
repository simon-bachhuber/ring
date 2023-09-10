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