from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from gymnasium.wrappers import RecordEpisodeStatistics

import torch
import gym_so100
from gym_so100.env import SO100GoalEnv  # Import your goal environment

def create_sac_her_model():
    # Your base environment
    # Wrap with GoalEnv (choose one approach)
    # Option 1: Nested Dict observation
    goal_env = SO100GoalEnv()

    # Option 2: Flattened observation (might be easier)
    # goal_env = CubeManipulationGoalEnvFlat(base_env)

    goal_env = RecordEpisodeStatistics(goal_env)
    # Vectorize
    vec_env = DummyVecEnv([lambda: goal_env])
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    new_logger = configure(log_dir, ["tensorboard", "stdout"])


    # Create SAC with HER
    model = SAC(
        'MultiInputPolicy',  # Required for Dict observation spaces
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            
        ),
        learning_rate=1e-4,
        buffer_size=2_000,
        batch_size=256,
        verbose=1,
        device=device,
        learning_starts=1000,
    )
    model.set_logger(new_logger)

    return model, vec_env


if __name__ == "__main__":
    log_dir = "logs/sac_so100"  # will hold TB files
    
    # Create environment
    model, vec_env = create_sac_her_model()

    # Create or load model
    start_steps = 0
    model.learn(10000)