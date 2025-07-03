import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_so100.constants import (
    SO100_ACTIONS,
    ASSETS_DIR,
    DT,
    SO100_JOINTS,
)
from gym_so100.tasks.single_arm import BOX_POSE, SO100TouchCubeTask, SO100TouchCubeSparseTask

from gym_so100.utils import fixed_so100_box_pose, sample_so100_box_pose


class SO100Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self._env = self._make_env_task(self.task)

        if self.obs_type == "so100_pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                                low=0,
                                high=255,
                       shape=(self.observation_height, self.observation_width, 3),
                       dtype=np.uint8,
                   )
                ,
               "agent_pos": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(len(SO100_JOINTS),),
                        dtype=np.float32,
                    ),
                }
            )
        elif self.obs_type == "so100_state":
            self.observation_space = spaces.Box(
                low=-100.0,
                high=100.0,
                shape=(len(SO100_JOINTS) + 3 * 3,),  # joints + box + bin + ee
                dtype=np.float32,
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(SO100_ACTIONS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "so100_touch_cube":
            xml_path = ASSETS_DIR / "so100_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = SO100TouchCubeTask(observation_width=self.observation_width, observation_height=self.observation_height)
        elif task_name == "so100_touch_cube_sparse":
            xml_path = ASSETS_DIR / "so100_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = SO100TouchCubeSparseTask(observation_width=self.observation_width, observation_height=self.observation_height)
        
        else:
            raise NotImplementedError(task_name)

        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "so100_pixels_agent_pos":
            rgb = raw_obs["images"]["top"].copy()
            obs = {
                "pixels": rgb,
                "agent_pos": raw_obs["qpos"].astype(np.float32),      # SO100 uses float32,
            }
        elif self.obs_type == "so100_state":
            obs = np.concatenate([
                raw_obs["box_position"],
                raw_obs["bin_position"],
                raw_obs["ee_position"],
                raw_obs["qpos"].astype(np.float32),    # SO100 uses float32,
            ])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        if self.task == "so100_touch_cube":
            BOX_POSE[0] = sample_so100_box_pose(seed)  # used in sim reset
        elif self.task == "so100_touch_cube_sparse":
            BOX_POSE[0] = sample_so100_box_pose(seed)  # used in sim reset
        else:
            raise ValueError(self.task)

        raw_obs = self._env.reset()

        observation = self._format_raw_obs(raw_obs.observation)

        info = {"is_success": False}
        return observation, info

    def step(self, action):
        assert action.ndim == 1
        _, reward, _, raw_obs = self._env.step(action)
        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        pass
