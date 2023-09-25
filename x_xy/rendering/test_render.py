import jax

import x_xy


def test_vispy_render():
    for sys in x_xy.io.list_load_examples():
        x_xy.render(sys, x_xy.State.create(sys).x, show_pbar=False, backend="vispy")


def test_mujoco_render():
    for sys in x_xy.io.list_load_examples():
        print(sys.model_name)
        x_xy.render(sys, x_xy.State.create(sys).x, show_pbar=False, backend="mujoco")


def test_shapes():
    sys_str = """
<x_xy model="shape_test">
    <options gravity="0 0 9.81" dt="0.01" />
    <worldbody>
        <geom type="sphere" mass="1" pos="0 0 0" dim="0.3" color="white" />
        <geom type="box" mass="1" pos="-1 0 0" quat="1 0 1 0" dim="1 0.3 0.2" color="0.8 0.3 1 0" />
        <geom type="cylinder" mass="1" pos="1 0 0.5" quat="0.75 0 0 0.25" dim="0.3 1" color="0.2 0.8 0.5" />
        <geom type="capsule" mass="1" pos="0 0 -1" dim="0.3 2" color="self"/>

        <body name="dummy" pos="0 0 0" quat="1 0 0 0" joint="ry" />
    </worldbody>
</x_xy>
    """  # noqa: E501

    sys = x_xy.load_sys_from_str(sys_str)
    state = x_xy.State.create(sys)
    step_fn = jax.jit(x_xy.step)
    state = step_fn(sys, state)

    import mediapy

    for backend in ["mujoco", "vispy"]:
        frame = x_xy.render(sys, state.x, show_pbar=False, backend=backend)[0]
        mediapy.write_image(f"docs/img/example_{backend}.png", frame)
