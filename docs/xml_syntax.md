# XML Syntax Documentation

## Overview
This document describes the XML syntax used to define a physical system. The structure includes elements for defining bodies, joints, geometries, simulation parameters, and rendering properties.

## Root Element
```xml
<model name="example_model">
    ...
</model>
```
- `name` (string, required): The name of the model.

## Simulation Parameters
```xml
<options gravity="0 0 -9.81" dt="0.01" />
```
- `gravity` (x y z, optional): Global gravity vector.
- `dt` (float, optional): Simulation time step.

## Default Parameters for `geom` and `body`
Changes the default values. For example:

```xml
<defaults>
    <geom mass="1.0"/>
</defaults>
```
- `mass` (float, optional): Default mass of geometries.

## World Definition
```xml
<worldbody>
    <body name="base" pos="0 0 0">
        ...
    </body>
</worldbody>
```
- `<worldbody>`: The root container for all bodies.
- `<body>`: Defines a physical body.
  - `name` (string, required): Unique identifier for the body.
  - `pos` (x y z, optional): Position in world coordinates.
  - for more see section `Bodies` below

## Geometry Definition and Rendering Properties
```xml
<geom type="box" mass="1" size="0.5 0.5 0.5" color="0.8 0.2 0.2" />
```
- `mass` (float, required): Mass of geometry.
- `type` (string, required): Shape type (`box`, `sphere`, `cylinder`, `xyz`, `capsule`).
- `dim` (Vector of floats, required): Dimensions of the geometry. Its dimensionality depends on the type of the geometry. 
    - `box`: length_x, length_y, length_z
    - `sphere`: radius
    - `cylinder`: radius, height
    - `xyz`: unit_vector_length
    - `capsule`: radius, length
- `color` (rgb or string, optional): RGB color (normalised from 0 to 1) of the object or string identifier of a color such as green, blue, red, orange, ...
- `pos` (x y z, optional): Position of geometry in coordinate system of surrouning body. Points to the center of mass of the geometry. Defaults to zeros.
- `euler` (x y z, optional): Euler angles in degree. Orientation of geometry in coordinate system of surrouning body. Mutually exclusive with field `quat`. Defaults to zeros.
- `quat` (u x y z, optional): Orientation of geometry in coordinate system of surrouning body. Mutually exclusive with field `euler`. Defaults to 1 0 0 0.

## Bodies
```xml
<body name="hinge" type="rx" pos="0 0 1" euler="90 0 0"/>
```
- `name` (string, required): Identifier for the body.
- `type` (string, required): Type of joint. Possible values:
  - `free`: 6D free joint
  - `cor`: 9D free joint, center of rotation also moves
  - `free_2d`: 3D free joint (1D rotation + 2D translations)
  - `frozen`: 0D joint
  - `spherical`: 3D rotational joint
  - `px`, `py`, `pz` (prismatic joints): 1D translational joints around x/y/z
  - `rx`, `ry`, `rz` (revolute joints): 1D rotational joints around x/y/z
  - `saddle`: 2D rotational joint
  - `p3d`: 3D translational joint
- `pos` (x y z, optional): Position relative to parent body. Defaults to zeros.
- `euler` (x y z, optional): Euler angles in degree. Orientation relative to parent body. Mutually exclusive with field `quat`. Defaults to zeros.
- `quat` (u x y z, optional): Orientation relative to parent body. Mutually exclusive with field `euler`. Defaults to 1 0 0 0.
- `pos_min` (x y z, optional): Lower bound for randomization of the `pos` value. Defaults to zeros.
- `pos_max` (x y z, optional): Upper bound for randomization of the `pos` value. Defaults to zeros.
- `damping` (Vector of floats, optional): Damping of the joint. It's dimensionality depends on the `qd` size of the joint type. So for a 1D joint, this is a single float, for a 3D joint it is three floats. Defaults to zeros.
- `armature` (Vector of floats, optional): Armature of the joint. It's dimensionality depends on the `qd` size of the joint type. So for a 1D joint, this is a single float, for a 3D joint it is three floats. Defaults to zeros.
- `spring_stiff` (Vector of floats, optional): Spring stiffness of the joint. It's dimensionality depends on the `qd` size. Defaults to zeros.
- `spring_zero` (Vector of floats, optional): Zero point for the spring force of the joint. It's dimensionality depends on the `q` size of the joint type. Defaults to 1 0 0 0 for `spherical`, `cor`, and `free`, and to zeros else.

## Example Model
```xml
<x_xy model="inv_pendulum">
    <options gravity="0 0 9.81" dt="0.01"/>
    <defaults>
        <geom edge_color="black" color="white"/>
    </defaults>
    <worldbody>
        <body name="cart" joint="px" damping="0.01">
            <geom type="box" mass="1" dim="0.4 0.1 0.1"/>
            <body name="pendulum" joint="ry" euler="0 -90 0" damping="0.01">
                <geom type="box" mass="0.5" pos="0.5 0 0" dim="1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</x_xy>
```

This syntax provides a structured way to define physical systems for simulation and rendering.
