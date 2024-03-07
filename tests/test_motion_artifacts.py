import ring
from ring.algorithms.generator import motion_artifacts

sys_start = """
<x_xy model="knee_flexible_imus">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="femur" joint="free" pos="0.5 0.5 0.3" damping="5 5 5 25 25 25">
            <geom type="xyz" dim="0.1"/>
            <geom type="capsule" mass="1" euler="0 90 0" pos="0.2 0 0" dim="0.05 0.4"/>
            <body name="imu1" joint="frozen" pos="0.2 0 0" pos_min="0.05 0 0" pos_max="0.35 0 0">
                <geom type="xyz" dim="0.05"/>
                <geom type="box" mass="0.1" dim="0.05 0.05 0.02" color="orange"/>
            </body>
            <body name="tibia" joint="ry" pos="0.4 0 0" damping="3">
                <geom type="xyz" dim="0.1"/>
                <geom type="capsule" mass="1" euler="0 90 0" pos="0.2 0 0" dim="0.04 0.4"/>
                <body name="imu2" joint="frozen" pos="0.2 0 0" pos_min="0.05 0 0" pos_max="0.35 0 0">
                    <geom type="xyz" dim="0.05"/>
                    <geom type="box" mass="0.1" dim="0.05 0.05 0.02" color="orange"/>
                </body>
                <geom type="box" mass="0" pos="0.45 0 .1" dim="0.025 0.05 0.2"/>
            </body>
        </body>
    </worldbody>
</x_xy>
"""  # noqa: E501

sys_target = """
<x_xy model="knee_flexible_imus">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="femur" joint="free" pos="0.5 0.5 0.3" damping="5 5 5 25 25 25">
            <geom type="xyz" dim="0.1"/>
            <geom type="capsule" mass="1" euler="0 90 0" pos="0.2 0 0" dim="0.05 0.4"/>
            <body name="_imu1" joint="spherical" pos="0.2 0 0" pos_min="0.05 0 0" pos_max="0.35 0 0" damping="0.03 0.03 0.03" spring_stiff="0.3 0.3 0.3">
                <body name="imu1" joint="p3d" pos_min="-.03 -.03 -.03" pos_max=".03 .03 .03" damping="5 5 5" spring_stiff="50 50 50">
                    <geom type="xyz" dim="0.05"/>
                    <geom type="box" mass="0.1" dim="0.05 0.05 0.02" color="orange"/>
                </body>
            </body>
            <body name="tibia" joint="ry" pos="0.4 0 0" damping="3">
                <geom type="xyz" dim="0.1"/>
                <geom type="capsule" mass="1" euler="0 90 0" pos="0.2 0 0" dim="0.04 0.4"/>
                <body name="_imu2" joint="spherical" pos="0.2 0 0" pos_min="0.05 0 0" pos_max="0.35 0 0" damping="0.03 0.03 0.03" spring_stiff="0.3 0.3 0.3">
                    <body name="imu2" joint="p3d" pos_min="-.03 -.03 -.03" pos_max=".03 .03 .03" damping="5 5 5" spring_stiff="50 50 50">
                        <geom type="xyz" dim="0.05"/>
                        <geom type="box" mass="0.1" dim="0.05 0.05 0.02" color="orange"/>
                    </body>
                </body>
                <geom type="box" mass="0" pos="0.45 0 .1" dim="0.025 0.05 0.2"/>
            </body>
        </body>
    </worldbody>
</x_xy>
"""  # noqa: E501


def test_inject_subsystems():
    sys_motion_artifacts = motion_artifacts.inject_subsystems(
        ring.io.load_sys_from_str(sys_start), pos_min_max=0.03
    )
    # inject_subsystems appends all injected systems at the end; saving to xml str
    # fixes this
    sys_motion_artifacts = ring.io.load_sys_from_str(
        ring.io.save_sys_to_str(sys_motion_artifacts)
    )
    assert ring.utils.sys_compare(
        sys_motion_artifacts,
        ring.io.load_sys_from_str(sys_target),
    )
