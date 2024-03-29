<p align="center">
<img src="https://raw.githubusercontent.com/SimiPixel/ring/main/docs/img/icon.svg" height="200" />
</p>

# Recurrent Inertial Graph-based Estimator (RING)
<img src="https://raw.githubusercontent.com/SimiPixel/ring/main/docs/img/coverage_badge.svg" height="20" />

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

### Known fixes

#### Offscreen rendering with Mujoco

> mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called

Solution:

```python
import os
os.environ["MUJOCO_GL"] = "egl"
```

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
