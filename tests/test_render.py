import jax
import pytest

import ring


@pytest.mark.render
def test_vispy_render():
    for sys in ring.io.list_load_examples():
        sys.render(ring.State.create(sys).x, show_pbar=False, backend="vispy")


@pytest.mark.render
def test_mujoco_render():
    for sys in ring.io.list_load_examples():
        print(sys.model_name)
        sys.render(ring.State.create(sys).x, show_pbar=False, backend="mujoco")


WRITE_IMAGE = False


@pytest.mark.render
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

    sys = ring.io.load_sys_from_str(sys_str)
    state = ring.State.create(sys)
    step_fn = jax.jit(ring.step)
    state = step_fn(sys, state)

    mediapy = ring.utils.import_lib("mediapy")

    for backend in ["mujoco", "vispy"]:
        frame = sys.render(state.x, show_pbar=False, backend=backend)[0]
        if WRITE_IMAGE:
            mediapy.write_image(f"docs/img/example_{backend}.png", frame)
