import multiprocessing
import time
from typing import Optional

import fire
import jax.numpy as jnp
import numpy as np

import ring
from ring import System


class InteractiveViewer:
    def __init__(self, sys: ring.System, **scene_kwargs):
        self._mp_dict = multiprocessing.Manager().dict()
        self._geom_dict = multiprocessing.Manager().dict()
        self.update_q(np.array(ring.State.create(sys).q))
        self.process = multiprocessing.Process(
            target=self._worker,
            args=(self._mp_dict, self._geom_dict, sys.to_str(), scene_kwargs),
        )
        self.process.start()

    def update_q(self, q: np.ndarray):
        self._mp_dict["q"] = q

    def make_geometry_transparent(self, body_number: int, geom_number: int):
        geom_name = f"body{body_number}_geom{geom_number}"
        # the value is not used
        self._geom_dict[geom_name] = None

    def _worker(self, mp_dict, geom_dict, sys_str, scene_kwargs):
        from ring.rendering import base_render

        sys = System.from_str(sys_str)
        while base_render._scene is None or base_render._scene._renderer.is_alive:
            sys.render(jnp.array(mp_dict["q"]), interactive=True, **scene_kwargs)

            if len(geom_dict) > 0:
                model = base_render._scene._model
                processed = []
                for geom_name in list(geom_dict.keys()):
                    # Get the geometry ID
                    geom_id = model.geom(geom_name).id
                    # Set transparency to 0 (fully transparent)
                    model.geom_rgba[geom_id, 3] = 0
                    print(f"Made geom with name={geom_name} transparent (worker)")
                    processed.append(geom_name)

                for geom_name in processed:
                    geom_dict.pop(geom_name)

    def __enter__(self):
        return self

    def close(self):
        self.process.terminate()
        self.process.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _fire_main(path_sys_xml: str, path_qs_np: Optional[str] = None, **scene_kwargs):
    """View motion given by trajectory of minimal coordinates in interactive viewer.

    Args:
        path_sys_xml (str): Path to xml file defining the system.
        path_qs_np (str | None, optional): Path to numpy array containing the timeseries of minimal coordinates with
            shape (T, DOF) where DOF is equal to `sys.q_size()`. Each minimal coordiante is from parent
            to child. So for example a `spherical` joint that connects the first body to the worldbody
            has a minimal coordinate of a quaternion that gives from worldbody to first body. The sampling
            rate of the motion is inferred from the `sys.dt` attribute. If `None` (default), then simply renders the
            unarticulated pose of the system.
    """  # noqa: E501

    sys = ring.System.from_xml(path_sys_xml)
    if path_qs_np is None:
        qs = np.array(ring.State.create(sys).q)[None]
    else:
        qs: np.ndarray = np.load(path_qs_np)

    assert qs.ndim == 2, f"qs.shape = {qs.shape}"
    T, Q = qs.shape
    assert Q == sys.q_size(), f"Q={Q} != sys.q_size={sys.q_size()}"
    dt_target = sys.dt

    with InteractiveViewer(sys, width=640, height=480, **scene_kwargs) as viewer:
        dt = dt_target
        last_t = time.time()
        t = -1

        while True:
            t = (t + 1) % T

            while dt < dt_target:
                time.sleep(0.001)
                dt = time.time() - last_t

            last_t = time.time()
            viewer.update_q(qs[t])
            dt = time.time() - last_t

            # process will be stopped if the window is closed
            if not viewer.process.is_alive():
                break


def main():
    fire.Fire(_fire_main)


if __name__ == "__main__":
    main()
