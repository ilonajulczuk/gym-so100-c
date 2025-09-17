""" """

from pathlib import Path

import gym_so100  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch
import argparse
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main(video_output_dir, policy_type, policy_path, num_episodes, task, video_name, prompt, normalize, robot_type):
    # Create a directory to store the video of the evaluation
    video_output_dir.mkdir(parents=True, exist_ok=True)
    # Select your device
    device = "mps"
    if policy_type == "act":
        policy = ACTPolicy.from_pretrained(policy_path)
    elif policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(policy_path)
    elif policy_type == "pi0fast":
        policy = PI0FASTPolicy.from_pretrained(policy_path)
    elif policy_type == "smolVLA":
        policy = SmolVLAPolicy.from_pretrained(policy_path)

    print("Policy loaded from:", policy_path)

    torch.set_default_dtype(torch.float32)

    policy.eval()
    policy.to(device)

    env = gym.make(
        task,
        obs_type="so100_pixels_agent_pos",
        observation_width=640,
        observation_height=480,
    )

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    print(policy.config.input_features)
    print(env.observation_space)

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    print(policy.config.output_features)
    print(env.action_space)

    # Reset the policy and environments to prepare for rollout
    policy.reset()
    numpy_observation, info = env.reset(seed=41)

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render())

    best_reward = 0
    total_rewards = []

    for num_episode in range(num_episodes):
        print(f"Starting episode {num_episode + 1}")
        step = 0
        total_reward = 0
        done = False
        env.reset()
        while not done:
            # Prepare observation for the policy running in Pytorch
            agent_pos = numpy_observation["agent_pos"]
            if normalize:
                agent_pos = gym_so100.constants.normalize_gym_so100_to_lerobot(agent_pos.copy())
            state = torch.from_numpy(agent_pos)
            image = torch.from_numpy(numpy_observation["pixels"])

            # Convert to float32 with image from channel first in [0,255]
            # to channel last in [0,1]
            state = state.to(torch.float32)
            image = image.to(torch.float32) / 255
            image = image.permute(2, 0, 1)


            # Send data tensors from CPU to GPU
            state = state.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)

            # Add extra (empty) batch dimension, required to forward the policy
            state = state.unsqueeze(0)
            image = image.unsqueeze(0)

            # Create the policy input dictionary            
            observation = {
                "observation.state": state,
                "observation.images.top": image,
            }
            if policy_type == "pi0fast":
                observation["observation.images.main"] = image
                observation["task"] = [prompt]
            elif policy_type == "smolVLA":
                observation["task"] = prompt
            if robot_type:
                observation["robot_type"] = robot_type
            # Predict the next action with respect to the current observation
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Prepare the action for the environment
            numpy_action = action.squeeze(0).to("cpu").numpy()
            if normalize:
                numpy_action = gym_so100.constants.normalize_lerobot_to_gym_so100(numpy_action.copy())

            # Step through the environment and receive a new observation
            numpy_observation, reward, terminated, truncated, info = env.step(
                numpy_action
            )
            print(f"{step=} {reward=} {terminated=}")

            # Keep track of all the rewards and frames
            rewards.append(reward)
            frames.append(env.render())

            total_reward += reward
            if total_reward > best_reward:
                best_reward = total_reward

            # The rollout is considered done when the success state is reached (i.e. terminated is True),
            # or the maximum number of iterations is reached (i.e. truncated is True)
            done = terminated | truncated | done
            step += 1

        if terminated:
            print("Success!")
        else:
            print("Failure!")
        total_rewards.append(total_reward)
        print(f"Episode {num_episode + 1} finished with total reward: {total_reward}")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = video_output_dir / video_name
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
    print(f"Best reward in {num_episodes} episodes: {best_reward}")
    print(f"Average reward in {num_episodes} episodes: {numpy.mean(total_rewards)}")
    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Record expert demonstrations via teleoperation"
    )

    parser.add_argument(
        "--policy_type",
        type=str,
        default="act",
        help="Type of policy to use (e.g., 'act', 'diffusion')",
    )

    parser.add_argument(
        "--policy_path",
        type=str,
        default="attilczuk/bin_sim_act1_colab",
        help="Path to the HF policy to use (e.g., 'attilczuk/bin_sim_act1_colab')",
    )

    parser.add_argument(
        "--video_output_dir",
        type=str,
        default=None,
        help="Directory to save video outputs (default: './outputs/videos')",
    )

    parser.add_argument(
        "--robot_type",
        type=str,
        default=None,
        help="Type of robot to use (default: None)",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="rollout.mp4",
        help="Name of the video file to save (default: 'rollout.mp4')",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up the red cube and put it in the bin",
        help="Prompt for the task (default: 'Pick up the red cube and put it in the bin')",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )

    parser.add_argument(
        "--normalize",
        type=bool,
        default=False,
        help="Whether to normalize actions (default: False)",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="gym_so100/SO100CubeToBin-v0",
        help="Task to run (default: 'gym_so100/SO100CubeToBin-v0')",
    )
    args = parser.parse_args()
    video_output_dir = Path(args.video_output_dir or f"./outputs/videos/{args.policy_path.replace('/', '_')}")

    main(
        video_output_dir=video_output_dir,
        policy_type=args.policy_type,
        num_episodes=args.num_episodes,
        policy_path=args.policy_path,
        video_name=args.video_name,
        task=args.task,
        prompt=args.prompt,
        normalize=args.normalize,
        robot_type=args.robot_type,
    )
