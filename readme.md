<p align="center">
<img src="figures/icon.svg" height="200" />
</p>

# `x_xy_v2` -- A *tiny* Kinematic Tree Simulator
<img src="figures/coverage_badge.svg" height="20" />

`x_xy` is a small package for performing
- forward dynamics
- inverse dynamics
- forward kinematics

on a general Kinematic Tree structure. 

## Installation

`pip install git+https://github.com/SimiPixel/x_xy_v2.git`

Additionally,
- `render.py` requires a vispy backend (e.g. `pip install pyqt6`).

    Note 1: On a headless node with a Nvidia GPU it works without any backend.

    Note 2: More info: https://vispy.org/installation.html

- `subpkgs/omc` requires `pandas`.

    

## Examples
Example systems can be found under `x_xy/io/examples`. 

Additionally, there are some simple example scripts available on the [wiki](https://github.com/SimiPixel/x_xy_v2/wiki). 