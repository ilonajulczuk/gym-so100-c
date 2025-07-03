import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_so100  # noqa: F401


@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("SO100TouchCube-v0", "so100_pixels_agent_pos"),
        ("SO100TouchCube-v0", "so100_state"),
    ],
)
def test_aloha(env_task, obs_type):
    env = gym.make(f"gym_so100/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)
