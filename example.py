# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_so100

env = gym.make(
    "gym_so100/SO100TouchCube-v0",
    obs_type="so100_pixels_agent_pos", 
    observation_width=64,
    observation_height=48,
)
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(observation["pixels"])

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("outputs/example.mp4", np.stack(frames), fps=25)