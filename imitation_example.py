#!/usr/bin/env python3
"""
Imitation Learning Example using demonstrations from teleoperation.

This script loads expert demonstrations recorded by teleop_example.py
and trains a Behavioral Cloning (BC) model using the imitation library.
"""

import pickle
import numpy as np
import gymnasium as gym
import gym_so100
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.algorithms import bc
from imitation.data.types import DictObs
import argparse
import os
import torch
import imageio

from collections import deque
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

import argparse


def create_single_env(task):
    """Create a single environment for evaluation."""
    env = gym.make(
        task,
        obs_type="so100_pixels_agent_pos",
        observation_width=64,
        observation_height=48,
    )

    env = RecordEpisodeStatistics(env)
    return env


def create_environment(num_envs, task):
    """Create environment with macOS subprocess fixes."""
    vec_env = make_vec_env(
        create_single_env,
        n_envs=num_envs,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"task": task},
    )
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    return vec_env


def load_demonstrations(filename):
    """Load expert demonstrations from pickle file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Demonstrations file not found: {filename}")

    with open(filename, "rb") as f:
        demonstrations = pickle.load(f)

    print(f"Loaded {len(demonstrations)} episodes from {filename}")
    return demonstrations


def convert_episodes_to_trajectories(episodes):
    """
    Convert teleoperation episodes to imitation.data.types.Trajectory format.

    Args:
        episodes: List of episodes from teleoperation recording

    Returns:
        List of Trajectory objects
    """
    trajectories = []

    for i, episode in enumerate(episodes):
        obs = episode["observations"]

        acts = np.array(episode["actions"][:-1])
        rews = np.array(episode["rewards"])
        infos = episode["infos"]

        # Check if episode has valid data
        if len(obs) == 0 or len(acts) == 0:
            print(f"Warning: Episode {i} has no data, skipping")
            continue

        # For trajectory format, we need:
        # - obs: observations including final state
        # - acts: actions (one less than observations)
        # - infos: info dicts (same length as actions)
        # - terminal: whether episode terminated naturally

        # Handle the case where obs and acts might have different lengths
        min_length = min(len(obs), len(acts))
        if min_length == 0:
            print(f"Warning: Episode {i} has no valid transitions, skipping")
            continue

        # Truncate to matching lengths
        episode_obs = obs[: min_length + 1]  # +1 because we need final observation
        episode_obs = DictObs.from_obs_list(episode_obs)
        episode_acts = acts[:min_length]
        episode_infos = (
            infos[:min_length] if len(infos) >= min_length else [{}] * min_length
        )

        # Ensure observations are numpy arrays
        # Create trajectory
        trajectory = Trajectory(
            obs=episode_obs,
            acts=episode_acts,
            infos=episode_infos,
            terminal=True,  # Assume all recorded episodes are complete
        )

        trajectories.append(trajectory)
        print(f"Converted episode {i}: {len(episode_acts)} transitions")

    print(f"Successfully converted {len(trajectories)} episodes to trajectories")
    return trajectories


def save_bc_policy(bc_trainer, filepath, env):
    """Manually save BC policy with all needed info"""
    save_data = {
        'policy_state_dict': bc_trainer.policy.state_dict(),
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'policy_class': type(bc_trainer.policy).__name__,
    }
    torch.save(save_data, filepath)
    print(f"Saved BC policy to {filepath}")

def load_bc_policy(filepath, env):
    """Load BC policy and return as SAC model"""
    from stable_baselines3 import SAC
    
    data = torch.load(filepath)
    
    # Create new SAC model
    sac_model = SAC("MultiInputPolicy", env, verbose=0)
    
    # Load the weights
    sac_model.policy.load_state_dict(data['policy_state_dict'])
    
    return sac_model

def train_bc_model(trajectories, env, save_path="bc_model"):
    """
    Train a Behavioral Cloning model on the trajectories.

    Args:
        trajectories: List of Trajectory objects
        env: Gymnasium environment
        save_path: Path to save the trained model
    """
    # Set up random number generator
    rng = np.random.default_rng(seed=42)

    # Convert trajectories to transitions for BC training
    print("Converting trajectories to transitions...")
    transitions = rollout.flatten_trajectories(trajectories)
    print(f"Created {len(transitions)} transitions from trajectories")

    if len(transitions) == 0:
        raise ValueError(
            "No transitions available for training. Check your demonstration data."
        )

    # Create BC trainer
    print("Initializing BC trainer...")
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    # Train the model
    print("Starting BC training...")
    bc_trainer.train(n_epochs=500)  # Adjust epochs as needed

    # Save the trained model
    print(f"Saving trained model to {save_path}")
    save_bc_policy(bc_trainer, save_path, env)

    return bc_trainer


def evaluate_model(bc_trainer, env, n_episodes=5, eval_video_path="outputs/bc_eval_video.mp4"):
    """
    Evaluate the trained BC model.

    Args:
        bc_trainer: Trained BC model
        env: Gymnasium environment
        n_episodes: Number of episodes to evaluate
    """
    print(f"Evaluating model over {n_episodes} episodes...")

    total_rewards = []

    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            # Get action from trained policy
            action, _ = bc_trainer.policy.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1

            image = env.render()
            frames.append(image)
            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: {episode_reward} reward, {step_count} steps")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\nEvaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")

    imageio.mimsave(eval_video_path, np.stack(frames), fps=25)

    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(
        description="Train BC model from teleoperation demonstrations"
    )
    parser.add_argument(
        "--demonstrations",
        type=str,
        default="expert_demonstrations.pkl",
        help="Path to demonstrations pickle file",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="gym_so100/SO100CubeToBin-v0",
        help="Environment ID",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="bc_model",
        help="Path to save trained BC model",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=5, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--no_eval", action="store_true", help="Skip evaluation after training"
    )

    args = parser.parse_args()

    try:
        # Load demonstrations
        print("Loading expert demonstrations...")
        episodes = load_demonstrations(args.demonstrations)

        if len(episodes) == 0:
            print("Error: No episodes found in demonstrations file")
            return

        # Create environment (same as used in teleoperation)
        print("Creating environment...")
        env = create_environment(1, args.env_id)

        # Convert episodes to trajectories
        print("Converting episodes to trajectories...")
        trajectories = convert_episodes_to_trajectories(episodes)

        if len(trajectories) == 0:
            print("Error: No valid trajectories could be created from episodes")
            return

        # Train BC model
        print("Training BC model...")
        bc_trainer = train_bc_model(trajectories, env, args.model_save_path)

        # Evaluate model (optional)
        if not args.no_eval:
            print("Evaluating trained model...")
            evaluate_model(bc_trainer, env, args.eval_episodes)

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        raise

    finally:
        if "env" in locals():
            env.close()


if __name__ == "__main__":
    main()
