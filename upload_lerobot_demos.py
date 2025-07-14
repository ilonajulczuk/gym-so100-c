import argparse
import pickle
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from gym_so100.constants import FPS, SO100_JOINTS

DEFAULT_FEATURES = {
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "seed": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
}


def load_demonstrations(filename):
    """Load expert demonstrations from pickle file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Demonstrations file not found: {filename}")

    with open(filename, "rb") as f:
        demonstrations = pickle.load(f)

    print(f"Loaded {len(demonstrations)} episodes from {filename}")
    return demonstrations


def convert_demos_to_dataset(
    demonstrations_file: str, user_id: str, env_name: str, **kwargs
):
    """Convert expert demonstrations to LeRobot dataset"""

    dataset_dir = kwargs["root"] + "/" + user_id + "/" + env_name

    # Load expert demonstrations
    print(f"Loading demonstrations from {demonstrations_file}")
    episodes = load_demonstrations(demonstrations_file)

    if len(episodes) == 0:
        print("Error: No episodes found in demonstrations file")
        return

    # setup the environment to get observation/action spaces
    env = gym.make(env_name, obs_type="so100_pixels_agent_pos")

    features = DEFAULT_FEATURES

    # For expert demonstrations, pixels is a numpy array with shape (1, 3, 48, 64)
    # We need to squeeze the first dimension and use it as "top" image
    features["observation.images.top"] = {
        "dtype": "video",
        "shape": (3, 48, 64),  # C, H, W format
        "names": ["channel", "height", "width"],
    }

    # states features - use agent_pos for state (squeeze the batch dimension)
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (
            6,
        ),  # Based on examination: agent_pos has shape (1, 6), we'll squeeze it
        "names": SO100_JOINTS,
    }

    # action feature (actions also need to be squeezed from (1, 6) to (6,))
    features["action"] = {
        "dtype": "float32",
        "shape": (6,),  # Actions have shape (1, 6), we'll squeeze them
        "names": SO100_JOINTS,
    }

    dataset = LeRobotDataset.create(
        repo_id="attilczuk/gym-so100experiment2",
        fps=FPS,
        root=dataset_dir,
        features=features,
        image_writer_processes=1,
        image_writer_threads=4,
    )

    successful_episodes = 0

    for episode_idx, episode in enumerate(episodes):
        print(f"Processing episode {episode_idx + 1}/{len(episodes)}")

        observations = episode["observations"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        infos = episode.get("infos", [{}] * len(observations))

        # Check if episode has valid data
        if len(observations) == 0 or len(actions) == 0:
            print(f"Warning: Episode {episode_idx} has no data, skipping")
            continue

        # Handle length mismatch between observations and actions
        min_length = min(len(observations), len(actions))
        if min_length == 0:
            print(f"Warning: Episode {episode_idx} has no valid transitions, skipping")
            continue

        # Truncate to matching lengths (actions should be one less than observations)
        episode_observations = observations[:min_length]
        episode_actions = actions[:min_length]
        episode_rewards = (
            rewards[:min_length] if len(rewards) >= min_length else [0] * min_length
        )
        episode_infos = (
            infos[:min_length] if len(infos) >= min_length else [{}] * min_length
        )

        episode_reward_sum = 0
        for i in range(len(episode_observations)):
            obs = episode_observations[i]
            action = episode_actions[i]
            reward = episode_rewards[i]
            info = episode_infos[i]

            # Extract image data - handle the actual format (1, 3, 48, 64) -> (3, 48, 64)
            if "pixels" in obs:
                pixels = obs["pixels"]
                if isinstance(pixels, np.ndarray):
                    # Squeeze the batch dimension: (1, 3, 48, 64) -> (3, 48, 64)
                    if len(pixels.shape) == 4 and pixels.shape[0] == 1:
                        image_data = pixels.squeeze(0)
                    else:
                        image_data = pixels

                    if image_data.dtype != np.uint8 and image_data.min() <= 0:
                        # Poor woman's normalization - clip to [0, 255] and convert to uint8.
                        # This is a hack for observations after vecenvnorm normalization.
                        image_data = np.clip(image_data * 255, 0, 255).astype(np.uint8)
                else:
                    print(f"Warning: Unexpected pixels type {type(pixels)} at step {i}")
                    break
            else:
                print(
                    f"Warning: No 'pixels' key in observation at step {i}, skipping episode"
                )
                break

            # Extract agent position for state - squeeze batch dimension (1, 6) -> (6,)
            if "agent_pos" in obs:
                agent_pos = obs["agent_pos"]
                if (
                    isinstance(agent_pos, np.ndarray)
                    and len(agent_pos.shape) == 2
                    and agent_pos.shape[0] == 1
                ):
                    state_data = agent_pos.squeeze(0).astype(np.float32)
                else:
                    state_data = np.array(agent_pos, dtype=np.float32)
            else:
                print(
                    f"Warning: No 'agent_pos' key in observation at step {i}, skipping episode"
                )
                break

            # Handle action format - squeeze if needed (1, 6) -> (6,)
            if (
                isinstance(action, np.ndarray)
                and len(action.shape) == 2
                and action.shape[0] == 1
            ):
                action_data = action.squeeze(0).astype(np.float32)
            else:
                action_data = np.array(action, dtype=np.float32)

            data_frame = {
                "observation.state": state_data,
                "observation.images.top": image_data,
                "action": action_data,
                "next.reward": np.array([reward], dtype=np.float32)[0],
                "next.success": np.array(
                    [reward.squeeze() >= 4], dtype=np.bool_
                ),  # Assuming reward >= 4 means success
                "seed": np.array([0], dtype=np.int64),
            }

            dataset.add_frame(data_frame, task=env_name)
            episode_reward_sum += reward


        # Save the episode
        print(f"Episode {episode_idx}: Total reward = {episode_reward_sum}")
       
    
        dataset.image_writer.wait_until_done()
        dataset.save_episode()
        dataset.clear_episode_buffer()

        successful_episodes += 1

    dataset.push_to_hub()
    print(f"Conversion completed!")
    print(f"Processed {successful_episodes}/{len(episodes)} episodes successfully")
    print(f"Dataset saved in {dataset_dir}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demonstrations",
        type=str,
        default="expert_demonstrations.pkl",
        help="Path to expert demonstrations pickle file",
    )
    parser.add_argument(
        "--user-id", type=str, default="test_user", help="User ID for the dataset"
    )
    parser.add_argument(
        "--root", type=str, default="dataset", help="Root dir to save recordings"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="gym_so100/SO100CubeToBin-v0",
        help="Environment name",
    )

    """
    Example usage:
    python upload_lerobot_demos.py --demonstrations expert_demonstrations.pkl --env-name gym_so100/SO100CubeToBin-v0
    """
    args = parser.parse_args()
    kwargs = vars(args)

    convert_demos_to_dataset(
        demonstrations_file=args.demonstrations,
        user_id=args.user_id,
        env_name=args.env_name,
        root=args.root,
    )
