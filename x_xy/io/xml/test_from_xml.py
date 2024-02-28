import jax.numpy as jnp

import x_xy

sys_str = """
<x_xy model="model">
<options gravity=".1 2 3" dt=".03"/>
    <worldbody>
        <body name="name" joint="rx" pos="1 2 3" euler="30 30 30" damping=".7" armature=".8" spring_stiff="1" spring_zero=".9">
            <geom type="box" mass="2.7" dim="0.2 0.3 0.4" color="black" edge_color="pink"/>
        </body>
    </worldbody>
</x_xy>
"""  # noqa: E501


def test_from_xml():
    pos = jnp.array([1.0, 2, 3])
    sys1 = x_xy.System(
        [-1],
        x_xy.base.Link(
            x_xy.base.Transform(
                pos=pos,
                rot=x_xy.maths.quat_euler(
                    jnp.array([jnp.deg2rad(30), jnp.deg2rad(30), jnp.deg2rad(30)])
                ),
            ),
            pos_min=pos,
            pos_max=pos,
        ).batch(),
        ["rx"],
        link_damping=jnp.array([0.7]),
        link_armature=jnp.array([0.8]),
        link_spring_zeropoint=jnp.array([0.9]),
        link_spring_stiffness=jnp.array([1.0]),
        dt=0.03,
        geoms=[
            x_xy.base.Box(
                jnp.array(2.7),
                x_xy.Transform.zero(),
                0,
                "black",
                "pink",
                jnp.array(0.2),
                jnp.array(0.3),
                jnp.array(0.4),
            )
        ],
        gravity=jnp.array([0.1, 2, 3.0]),
        link_names=["name"],
        model_name="model",
        omc=[None],
    )
    sys1 = sys1.parse()
    sys2 = x_xy.load_sys_from_str(sys_str)

    assert x_xy.utils.sys_compare(sys1, sys2)
