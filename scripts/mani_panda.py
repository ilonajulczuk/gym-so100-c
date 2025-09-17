import gymnasium as gym
import mani_skill.envs

shader = "minimal"

    # env_kwargs = dict(
    #     obs_mode=args.obs_mode,
    #     reward_mode=args.reward_mode,
    #     control_mode=args.control_mode,
    #     render_mode=args.render_mode,
    #     sensor_configs=dict(shader_pack=args.shader),
    #     human_render_camera_configs=dict(shader_pack=args.shader),
    #     viewer_camera_configs=dict(shader_pack=args.shader),
    #     num_envs=args.num_envs,
    #     sim_backend=args.sim_backend,
    #     render_backend=args.render_backend,
    #     enable_shadow=True,
    #     parallel_in_single_scene=parallel_in_single_scene,
    # )

import time

env = gym.make(
    "PickCube-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    robot_uids="panda_wristcam",
    obs_mode="rgbd",  # there is also "state_dict", "rgbd", ...
    # control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
    # control_mode="pd_ee_target_delta_pose",
    render_mode="human",
    sensor_configs=dict(shader_pack=shader),
    human_render_camera_configs=dict(shader_pack=shader),
    viewer_camera_configs=dict(shader_pack=shader),
    sim_backend="auto",
    render_backend="gpu",

)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # a display is required to render
    time.sleep(1)


env.close()
