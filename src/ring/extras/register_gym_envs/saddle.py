from gymnasium import spaces
import gymnasium as gym
import jax
import numpy as np

import ring

xml = """
<x_xy model="lam2">
  <options dt="0.01" gravity="0.0 0.0 9.81"/>
  <worldbody>
    <body joint="free" name="seg1" pos="0.4 0.0 0.0" pos_min="0.2 -0.05 -0.05" pos_max="0.55 0.05 0.05" damping="5.0 5.0 5.0 25.0 25.0 25.0">
      <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_blue" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
      <geom pos="0.05 0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
      <geom pos="0.15 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
      <body joint="frozen" name="imu1" pos="0.099999994 0.0 0.035" pos_min="0.050000012 -0.05 -0.05" pos_max="0.15 0.05 0.05">
        <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
      </body>
      <body joint="saddle" name="seg2" pos="0.20000002 0.0 0.0" pos_min="0.0 -0.05 -0.05" pos_max="0.35 0.05 0.05" damping="3.0 3.0">
        <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_blue" edge_color="black" type="box" dim="0.2 0.05 0.05"/>
        <geom pos="0.1 0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
        <geom pos="0.15 -0.05 0.0" mass="0.1" color="black" edge_color="black" type="box" dim="0.01 0.1 0.01"/>
        <body joint="frozen" name="imu2" pos="0.100000024 0.0 0.035" pos_min="0.050000012 -0.05 -0.05" pos_max="0.14999998 0.05 0.05">
          <geom mass="0.1" color="dustin_exp_orange" edge_color="black" type="box" dim="0.05 0.03 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</x_xy>
"""  # noqa: E501


class Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, T: float = 60):
        self._sys = ring.System.create(xml)
        self._generator = ring.RCMG(
            self._sys,
            ring.MotionConfig(T=T, pos_min=0),
            add_X_imus=1,
            # child-to-parent
            add_y_relpose=1,
            cor=True,
            disable_tqdm=True,
            keep_output_extras=True,
        ).to_lazy_gen()
        # warmup jit compile
        self._generator(jax.random.PRNGKey(1))

        self.observation_space = spaces.Box(-float("inf"), float("inf"), shape=(12,))
        # quaternion; from seg2 to seg1, so child-to-parent
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,))
        self.reward_range = (-float("inf"), 0.0)

        self._action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        jax_seed = self.np_random.integers(1, int(1e18))
        (X, y), (_, _, xs, _) = self._generator(jax.random.PRNGKey(jax_seed))
        self._xs = xs[0]
        self._truth = y["seg2"][0]
        self._T = self._truth.shape[0]
        self._observations = np.zeros((self._T, 12), dtype=np.float32)
        self._observations[:, :3] = X["seg1"]["acc"][0]
        self._observations[:, 3:6] = X["seg1"]["gyr"][0]
        self._observations[:, 6:9] = X["seg2"]["acc"][0]
        self._observations[:, 9:12] = X["seg2"]["gyr"][0]
        self._t = 0

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return self._observations[self._t]

    def _get_info(self):
        return {"truth": self._truth[self._t]}

    def step(self, action):
        self._t += 1

        # convert to unit quaternion
        self._action = action / np.linalg.norm(action)
        reward = -self._abs_angle(self._truth[self._t - 1], self._action)

        terminated = False
        truncated = self._t >= (self._T - 1)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _abs_angle(self, q, qhat) -> float:
        return float(jax.jit(ring.maths.angle_error)(q, qhat))

    def render(self):
        light = '<light pos="0 0 3" dir="0 0 -1" directional="false"/>'
        render_kwargs = dict(
            show_pbar=False,
            camera="target",
            width=640,
            height=480,
            add_lights={-1: light},
        )
        x = [self._xs[self._t]]
        if self._action is None:
            return self._sys.render(x, **render_kwargs)[0]
        yhat = {"seg1": np.array([[1.0, 0, 0, 0]]), "seg2": self._action[None]}
        return self._sys.render_prediction(x, yhat, **render_kwargs)[0]
