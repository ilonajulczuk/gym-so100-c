#!/usr/bin/env python3
"""
Imitation learning (Behavior Cloning) using demonstrations from teleoperation.

This script loads expert demonstrations recorded by teleop_example.py
and trains a Behavioral Cloning (BC) model using the imitation library.

Can be used as pretraining for SAC and can be used to train the newly pretrained SAC model.

# Just BC training:
python train_bc.py --demonstrations expert_demonstrations.pkl

# BC + SAC training:
python train_bc.py --demonstrations expert_demonstrations.pkl --continue_with_sac --sac_timesteps 50000

# Skip evaluation:
python train_bc.py --demonstrations expert_demonstrations.pkl --continue_with_sac --no_eval

# Custom policy architecture:
python train_bc.py --demonstrations expert_demonstrations.pkl --net_arch 512 512 256

# Custom BC epochs and architecture:
python train_bc.py --demonstrations expert_demonstrations.pkl --bc_epochs 200 --net_arch 128 128
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
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy

from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.logger import configure


device = "cpu"
if torch.backends.mps.is_available():
    print("Using MPS backend for PyTorch")
    device = "mps"
elif torch.cuda.is_available():
    print("Using CUDA backend for PyTorch")
    device = "cuda"
else:
    print("Using CPU backend for PyTorch")
    device = "cpu"

# Set default dtype to float32 to avoid MPS compatibility issues
torch.set_default_dtype(torch.float32)


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
        acts = np.array(episode["actions"][:-1], dtype=np.float32)
        rews = np.array(episode["rewards"], dtype=np.float32)
        infos = episode["infos"]

        # Check if episode has valid data
        if len(obs) == 0 or len(acts) == 0:
            print(f"Warning: Episode {i} has no data, skipping")
            continue

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


def create_custom_policy(observation_space, action_space, policy_kwargs=None):
    """
    Create a custom policy with configurable architecture for BC training.

    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        policy_kwargs: Dict with policy configuration (net_arch, activation_fn, etc.)

    Returns:
        ActorCriticPolicy instance with custom architecture
    """
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": [256, 256],
            "activation_fn": torch.nn.ReLU,
        }

    # Determine policy class based on observation space
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy
    from gymnasium.spaces import Dict

    if isinstance(observation_space, Dict):
        # For dict observations (like image + state)
        policy_class = MultiInputActorCriticPolicy
    else:
        # For simple observations
        policy_class = ActorCriticPolicy

    print(f"Using policy class: {policy_class.__name__}")

    # Create a policy with the specified architecture
    policy = policy_class(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,  # Dummy learning rate schedule
        net_arch=policy_kwargs.get("net_arch", [256, 256]),
        activation_fn=policy_kwargs.get("activation_fn", torch.nn.ReLU),
    )

    # Ensure policy is on the correct device and uses float32
    policy = policy.to(device)
    if hasattr(policy, "float"):
        policy = policy.float()  # Ensure float32 precision

    return policy


def save_bc_policy(bc_trainer, filepath, env):
    """Save BC policy with all needed info"""
    save_data = {
        "policy_state_dict": bc_trainer.policy.state_dict(),
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "policy_class": type(bc_trainer.policy).__name__,
    }
    torch.save(save_data, filepath)
    print(f"Saved BC policy to {filepath}")


def train_bc_model(
    trajectories, env, save_path="bc_model", policy_kwargs=None, n_epochs=100
):
    """
    Train a Behavioral Cloning model with customizable architecture for SAC compatibility.

    Args:
        trajectories: List of Trajectory objects
        env: Gymnasium environment
        save_path: Path to save the trained model
        policy_kwargs: Dict with policy configuration (net_arch, activation_fn, etc.)
        n_epochs: Number of training epochs
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

    # Set default policy kwargs to match SAC architecture
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": [256, 256],  # Default SAC architecture
            "activation_fn": torch.nn.ReLU,
        }

    print(f"Using policy architecture: {policy_kwargs}")

    # Create custom policy
    print("Creating custom policy...")
    policy = create_custom_policy(
        env.observation_space, env.action_space, policy_kwargs
    )

    # Create BC trainer with custom policy
    print("Initializing BC trainer...")
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=policy,
        device=device,
    )

    # Train the BC model
    print(f"Starting BC training for {n_epochs} epochs...")
    bc_trainer.train(n_epochs=n_epochs)

    # Save the trained model
    print(f"Saving trained BC model to {save_path}")
    save_bc_policy(bc_trainer, save_path, env)

    return bc_trainer


def save_bc_as_sac(
    bc_trainer, save_path, env, log_dir="logs/sac_bc_init", policy_kwargs=None
):
    """Transfer BC policy weights to SAC model with matching architecture"""

    # Use the same policy kwargs as BC if not provided
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": [256, 256],  # Match BC architecture
            "activation_fn": torch.nn.ReLU,
        }

    print(f"Creating SAC model with architecture: {policy_kwargs}")
    sac_model = SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        buffer_size=50_000,  # Increase this (big stability gain)
        batch_size=256,  # Increase this (stability)
        ent_coef="auto",
        target_entropy=-2.0,  # Fix entropy (stop the chaos)
        device=device,
    )

    new_logger = configure(log_dir, ["tensorboard", "stdout"])
    sac_model.set_logger(new_logger)

    # Transfer compatible weights from BC to SAC
    print("Transferring compatible weights from BC to SAC...")
    bc_state_dict = bc_trainer.policy.state_dict()
    sac_state_dict = sac_model.policy.state_dict()

    # Transfer only the compatible layers (features extractor and shared layers)
    transferred_keys = []
    for key in bc_state_dict:
        if (
            key in sac_state_dict
            and bc_state_dict[key].shape == sac_state_dict[key].shape
        ):
            sac_state_dict[key] = bc_state_dict[key]
            transferred_keys.append(key)

    print(f"Transferred {len(transferred_keys)} compatible layers:")
    for key in transferred_keys:
        print(f"  - {key}")

    # Load the updated state dict
    sac_model.policy.load_state_dict(sac_state_dict)

    # Save the SAC model
    sac_model.save(save_path)
    print(f"Saved BC-initialized SAC model to {save_path}")

    return sac_model


def continue_with_sac_training(
    bc_sac_model, env, total_timesteps=100000, save_path="final_sac_model"
):
    """Continue training the BC-initialized SAC model"""
    print(f"Starting SAC fine-tuning for {total_timesteps} timesteps...")

    # Continue training with SAC
    bc_sac_model.learn(total_timesteps=total_timesteps)

    # Save the final SAC model
    bc_sac_model.save(save_path)
    print(f"Saved final SAC model to {save_path}")

    return bc_sac_model


def evaluate_model(
    bc_trainer, env, n_episodes=5, eval_video_path="../outputs/bc_eval_video.mp4"
):
    """
    Evaluate the trained BC model.

    Args:
        bc_trainer: Trained BC model
        env: Gymnasium environment
        n_episodes: Number of episodes to evaluate
    """
    print(f"Evaluating BC model over {n_episodes} episodes...")

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

    print(f"\nBC Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")

    if frames:
        imageio.mimsave(eval_video_path, np.stack(frames), fps=25)
        print(f"Saved evaluation video to {eval_video_path}")

    return mean_reward, std_reward


def evaluate_sac_model(
    sac_model, env, n_episodes=5, eval_video_path="../outputs/sac_eval_video.mp4"
):
    """Evaluate SAC model"""
    print(f"Evaluating SAC model over {n_episodes} episodes...")

    total_rewards = []
    frames = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            # SAC prediction
            action, _ = sac_model.predict(obs, deterministic=True)

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

    print(f"\nSAC Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")

    if frames:
        imageio.mimsave(eval_video_path, np.stack(frames), fps=25)
        print(f"Saved evaluation video to {eval_video_path}")

    return mean_reward, std_reward


def main():
    parser = argparse.ArgumentParser(
        description="Train BC model from teleoperation demonstrations and optionally continue with SAC"
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
    parser.add_argument(
        "--continue_with_sac",
        action="store_true",
        help="Continue training with SAC after BC",
    )
    parser.add_argument(
        "--sac_timesteps", type=int, default=100000, help="SAC training timesteps"
    )
    parser.add_argument(
        "--sac_save_path",
        type=str,
        default="bc_sac_model",
        help="Path to save SAC model",
    )
    parser.add_argument(
        "--bc_epochs", type=int, default=100, help="Number of BC training epochs"
    )
    parser.add_argument(
        "--net_arch",
        nargs="+",
        type=int,
        default=[256, 256],
        help="Network architecture (list of layer sizes)",
    )

    args = parser.parse_args()

    # Create policy kwargs from arguments
    policy_kwargs = {
        "net_arch": args.net_arch,
        "activation_fn": torch.nn.ReLU,
    }

    print(f"Using policy architecture: {policy_kwargs}")

    try:
        # Load demonstrations
        print("Loading expert demonstrations...")
        episodes = load_demonstrations(args.demonstrations)

        if len(episodes) == 0:
            print("Error: No episodes found in demonstrations file")
            return

        # Create environment
        print("Creating environment...")
        env = create_environment(1, args.env_id)

        # Convert episodes to trajectories
        print("Converting episodes to trajectories...")
        trajectories = convert_episodes_to_trajectories(episodes)

        if len(trajectories) == 0:
            print("Error: No valid trajectories could be created from episodes")
            return

        # Train BC model with custom architecture
        print("Training BC model...")
        bc_trainer = train_bc_model(
            trajectories,
            env,
            args.model_save_path,
            policy_kwargs=policy_kwargs,
            n_epochs=args.bc_epochs,
        )

        # Evaluate BC model (optional)
        if not args.no_eval:
            print("Evaluating trained BC model...")
            evaluate_model(bc_trainer, env, args.eval_episodes)

        # Transfer BC weights to SAC model with matching architecture
        print("Creating SAC model with BC initialization...")
        bc_sac_model = save_bc_as_sac(
            bc_trainer,
            args.sac_save_path + "_bc_init",
            env,
            policy_kwargs=policy_kwargs,
        )

        # Continue with SAC training (optional)
        if args.continue_with_sac:
            print("Continuing with SAC training...")
            final_sac_model = continue_with_sac_training(
                bc_sac_model,
                env,
                total_timesteps=args.sac_timesteps,
                save_path=args.sac_save_path + "_final",
            )

            # Evaluate final SAC model
            if not args.no_eval:
                print("Evaluating final SAC model...")
                evaluate_sac_model(final_sac_model, env, args.eval_episodes)

        print("Training completed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        raise

    finally:
        if "env" in locals():
            env.close()


if __name__ == "__main__":
    main()
