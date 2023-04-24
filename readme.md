<p align="center">
<img src="figures/icon.svg" height="200" />
</p>

# `x_xy_v2` -- A *tiny* Kinematic Tree Simulator

`x_xy` is a small package for performing
- forward dynamics
- inverse dynamics
- forward kinematics

on a general Kinematic Tree structure. 

It uses nothing but JAX (and flax.struct).

It's meant to be minimalistic and simple. It uses spatial vectors and implements algorithms as proposed by Roy Featherstone. Nameing is heavily inspired by `brax`.

It currently does *not* support
- collisions (i.e. every body is (sort of) transparent)

and probably won't in the near future.

## Installation

`pip install git+https://github.com/SimiPixel/x_xy.git`

Additionally,
- `render.py` requires a vispy-backend (one is enough). Good options are
    on linux:
        - PyQT5 (via e.g. pip)
        - EGL (headless) (via e.g. apt)
    on m1 mac:
        - PyQT6 (via e.g. pip)
        - OS Mesa (headless) (via brew install mesa)

    More info: https://vispy.org/installation.html
