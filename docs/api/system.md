# The System object

The system object (in source code shortened to `sys`) captures all relevant physical information about the system. 

This includes (not exhaustively):

- All defined coordinate systems in the system (also called links), and their associated names (`sys.link_names`)
- All fixed translation and rotation between links (`sys.links.transform1`)
- All degrees of freedom between links (`sys.link_types`)
- All massive objects (`sys.links.inertia`) and their spatial volume (`sys.geoms`)
- The timestep size (default: 0.01s or 100Hz)

$$\frac{\mathrm{d}y}{\mathrm{d}t} = f(t, y(t))$$

!!! info
    Something awesome!

::: x_xy.base.System
---
::: x_xy.io.xml.from_xml.load_sys_from_str
---
::: x_xy.io.xml.from_xml.load_sys_from_xml
---
::: x_xy.io.xml.to_xml.save_sys_to_str
---
::: x_xy.io.xml.to_xml.save_sys_to_xml