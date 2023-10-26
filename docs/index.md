<p align="center">
<img src="https://raw.githubusercontent.com/SimiPixel/x_xy_v2/main/docs/img/icon.svg" height="200" />
</p>

# `x_xy_v2`
<img src="https://raw.githubusercontent.com/SimiPixel/x_xy_v2/main/docs/img/coverage_badge.svg" height="20" />

## Installation

Supports `Python=3.10/3.11` (tested).

Install with `pip` using

`pip install 'x_xy[all] @ git+https://github.com/SimiPixel/x_xy_v2'`

Typically, this will install `jax` as cpu-only version. Afterwards, gpu-enabled version can be installed with
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

or with (if NVIDIA driver is a little older)
```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Installation of Extras

Seperates dependencies neatly per subpackage. E.g. the dependencies for subpkg `omc` can be installed via

`pip install 'x_xy[omc] @ git+https://github.com/SimiPixel/x_xy_v2'`

Also available installs are

`pip install 'x_xy @ git+https://github.com/SimiPixel/x_xy_v2'` (base; only subpackages `sys_composer` and `sim2real` work, no rendering)

`pip install 'x_xy[omc] @ git+https://github.com/SimiPixel/x_xy_v2'` (base+omc)

`pip install 'x_xy[ml] @ git+https://github.com/SimiPixel/x_xy_v2'` (base+ml)

`pip install 'x_xy[exp] @ git+https://github.com/SimiPixel/x_xy_v2'` (base+exp)

`pip install 'x_xy[muj] @ git+https://github.com/SimiPixel/x_xy_v2'` (base+ mujoco rendering backend)

`pip install 'x_xy[vis] @ git+https://github.com/SimiPixel/x_xy_v2'` (base+ vispy rendering backend)

`pip install 'x_xy[all] @ git+https://github.com/SimiPixel/x_xy_v2'` (everything works)

`pip install 'x_xy[all_muj] @ git+https://github.com/SimiPixel/x_xy_v2'` (everything works but not vispy rendering only mujoco rendering)

`pip install 'x_xy[dev] @ git+https://github.com/SimiPixel/x_xy_v2'` (base+ development dependencies)

Can also be combined, e.g.
`pip install 'x_xy[ml,omc] @ git+https://github.com/SimiPixel/x_xy_v2'` (base+ml+omc)

## Documentation

Available [here](https://simipixel.github.io/x_xy_v2/).

### Known fixes

#### Offscreen rendering with Mujoco

> mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called

Solution:

```python
import os
os.environ["MUJOCO_GL"] = "egl"
```

## Publications

The following publications utilize this software library, and refer to it as the *Random Chain Motion Generator (RCMG)* (more specifically the function `x_xy.build_generator`):

- [*RNN-based Observability Analysis for Magnetometer-Free Sparse Inertial Motion Tracking*](https://ieeexplore.ieee.org/document/9841375)
- [*Plug-and-Play Sparse Inertial Motion Tracking With Sim-to-Real Transfer*](https://ieeexplore.ieee.org/document/10225275)
- [*RNN-based State and Parameter Estimation for Sparse Magnetometer-free Inertial Motion Tracking*](https://www.journals.infinite-science.de/index.php/automed/article/view/745)

### Other useful ressources

Particularly useful is the following publication from *Roy Featherstone*
- [*A Beginner’s Guide to 6-D Vectors (Part 2)*](https://ieeexplore.ieee.org/document/5663690)

## Contact

Simon Bachhuber (simon.bachhuber@fau.de)
