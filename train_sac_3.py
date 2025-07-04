import imageio
import gymnasium as gym
import numpy as np
import gym_so100
import torch
import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.env_util import make_vec_env


# Apply same wrappers as training (but single env)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from stable_baselines3.common.env_util import make_vec_env

class EvaluationVideoCallback(BaseCallback):
    """
    Custom callback that uses your existing evaluate function to get mean_reward and video frames
    """
    
    def __init__(
        self,
        evaluate_function,
        eval_freq=1000,
        best_model_save_path="./outputs/checkpoints_from_eval/",
        video_folder="./outputs/videos/",
        callback_on_new_best=None,
        num_episodes=3,
        verbose=1,
        prefix="sac_so100_get_cube_new",
        task=None,
    ):
        """
        Args:
            evaluate_function: Your function that takes model and returns (mean_reward, video_frames)
            eval_freq: Evaluate every eval_freq timesteps
            best_model_save_path: Path to save best model
            video_folder: Folder to save videos
            callback_on_new_best: Callback to call when new best model is found
            verbose: Verbosity level
            prefix: Prefix for naming saved files
        """
        super().__init__(verbose)
        
        self.evaluate_function = evaluate_function
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.callback_on_new_best = callback_on_new_best
        self.num_episodes = num_episodes
        self.prefix = prefix
        self.task = task
        
        # Paths
        self.best_model_save_path = best_model_save_path
        self.video_folder = video_folder
        
        # Create directories
        if self.best_model_save_path:
            os.makedirs(best_model_save_path, exist_ok=True)
        if self.video_folder:
            os.makedirs(video_folder, exist_ok=True)
        
        # Evaluation counter
        self.eval_count = 0

    def _save_video(self, frames, filename):
        """Save video frames as MP4"""
        if frames and len(frames) > 0:
            filepath = os.path.join(self.video_folder, f"{filename}.mp4")
            imageio.mimsave(filepath, frames, fps=30)
            if self.verbose > 0:
                print(f"Video saved to {filepath}")

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Use your custom evaluate function
            mean_reward, video_frames = self.evaluate_function(
                self.model, num_episodes=self.num_episodes, deterministic=True, task=self.task
            )
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"mean_reward={mean_reward:.2f}")
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/timesteps", self.num_timesteps)
            
            # Save video
            if self.video_folder and video_frames:
                video_filename = f"{self.prefix}_eval_step_{self.num_timesteps}_reward_{mean_reward:.2f}"
                self._save_video(video_frames, video_filename)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f} (previous: {self.best_mean_reward:.2f})")
                
                if self.best_model_save_path:
                    # Save like CheckpointCallback does
                    model_path = os.path.join(self.best_model_save_path, f"{self.prefix}_best_model_{self.num_timesteps}_steps")
                    self.model.save(model_path)
                    
                    # Save replay buffer if it exists (like CheckpointCallback)
                    if hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                        replay_buffer_path = os.path.join(self.best_model_save_path, f"{self.prefix}_best_model_{self.num_timesteps}_steps_replay_buffer.pkl")
                        self.model.save_replay_buffer(replay_buffer_path)
                    
                    # Save VecNormalize if it exists (like CheckpointCallback)
                    if self.model.get_vec_normalize_env() is not None:
                        vec_normalize_path = os.path.join(self.best_model_save_path, f"{self.prefix}_best_model_{self.num_timesteps}_steps_vecnormalize.pkl")
                        self.model.get_vec_normalize_env().save(vec_normalize_path)
                    
                    if self.verbose > 0:
                        print(f"Best model saved to {model_path}")
                
                self.best_mean_reward = mean_reward
                
                # Save best video separately
                if self.video_folder and video_frames:
                    best_video_filename = f"{self.prefix}_best_model_reward_{mean_reward:.2f}"
                    self._save_video(video_frames, best_video_filename)
                
                # Trigger callback for new best model
                if self.callback_on_new_best is not None:
                    return self.callback_on_new_best(locals(), globals())
            
            self.eval_count += 1
        
        return True


def evaluate(model, num_episodes=10, deterministic=True, frames=None, task=None):
    """
    Evaluate using a single environment (recreated from the original env)
    """
    if frames is None:
        frames = []
    
    # eval_env = gym.make(
    #     "gym_so100/SO100TouchCube-v0",
    #     obs_type="so100_pixels_agent_pos",
    #     observation_width=64,
    #     observation_height=48,
    # )
    
    eval_env = make_vec_env(create_single_env, n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs={"task": task})
    eval_env = VecTransposeImage(eval_env)
    # eval_env = Monitor(eval_env)
    # eval_env = DummyVecEnv([lambda: eval_env])
    # eval_env = VecTransposeImage(eval_env)
    
    # Apply VecNormalize with training stats
    train_env = model.get_env()
    if hasattr(train_env, 'obs_rms'):  # If using VecNormalize
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_obs=10.0,)
        eval_env.obs_rms = train_env.obs_rms  # Use training normalization stats
        eval_env.ret_rms = train_env.ret_rms
    
    all_episode_rewards = []
    
    for i in range(num_episodes):
        episode_reward = 0.0
        done = False
        obs = eval_env.reset()
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = eval_env.step(action)
            
            # Record frame
            try:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass
                
            episode_reward += reward[0]  # reward is array for vec env
        
        all_episode_rewards.append(episode_reward)
    
    eval_env.close()
    
    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f}, Num episodes: {num_episodes}")
    
    return mean_episode_reward, frames


def evaluate_old(model, num_episodes=10, deterministic=True, frames=None):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    if frames is None:
        frames = []
    # This function will only work for a single Environment
    vec_env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            # also note that the step only returns a 4-tuple, as the env that is returned
            # by model.get_env() is an sb3 vecenv that wraps the >v0.26 API
            obs, reward, done, info = vec_env.step(action)
            frame = vec_env.render()
            if frame is not None:
                frames.append(frame)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward, frames


def create_environment_old():
    """Create and configure the training environment."""
    def make_env():
        env = gym.make(
            "gym_so100/SO100TouchCube-v0",
            obs_type="so100_pixels_agent_pos",
            observation_width=64,
            observation_height=48,
        )

        env = RecordEpisodeStatistics(env)
        return env
    # 1) vectorise
    vec_env = SubprocVecEnv([make_env for _ in range(6)])
    # vec_env = DummyVecEnv([lambda: env], create_env=False)  # for SB3 ≥ 2.4
    # 2) move channel first for ALL dict image keys
    vec_env = VecTransposeImage(vec_env)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    
    return vec_env

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
    vec_env = make_vec_env(create_single_env, n_envs=num_envs, vec_env_cls=DummyVecEnv, env_kwargs={"task": task})
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    
    return vec_env


def create_model(vec_env, log_dir):
    """Create and configure the SAC model."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    new_logger = configure(log_dir, ["tensorboard", "stdout"])

    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=1e-4,        # Keep your current rate
        buffer_size=50_000,        # Increase this (big stability gain)
        batch_size=256,            # Increase this (stability)
        ent_coef='auto',
        target_entropy=-2.0,       # Fix entropy (stop the chaos)
        device=device,
        tensorboard_log=log_dir,
    )
    model.set_logger(new_logger)
    return model


def load_checkpoint(checkpoint_path, vec_env_stats_path, vec_env, log_dir):
    """Load model and vectorized environment from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Extract step count from checkpoint filename
    import re
    match = re.search(r'(\d+)_steps\.zip$', checkpoint_path)
    start_steps = int(match.group(1)) if match else 0
    
    # Load the model
    model = SAC.load(checkpoint_path)
    
    # Reconfigure the environment and logger
    model.set_env(vec_env)
    new_logger = configure(log_dir, ["tensorboard", "stdout"])
    model.set_logger(new_logger)
    
    # Load vectorized environment stats if available
    if vec_env_stats_path and os.path.exists(vec_env_stats_path):
        print(f"Loading VecNormalize stats from: {vec_env_stats_path}")
        vec_env = VecNormalize.load(vec_env_stats_path, vec_env)
        model.set_env(vec_env)
    else:
        print("VecNormalize stats file not found, using fresh normalization")
    
    print(f"Checkpoint loaded from step {start_steps}")
    return model, vec_env, start_steps


def create_callbacks(vec_env, save_freq, prefix, task):
    """Create training callbacks for model and environment checkpointing."""
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./outputs/checkpoints/",
        name_prefix=prefix,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    evaluate_callback = EvaluationVideoCallback(
        evaluate_function=evaluate, 
        eval_freq=save_freq * 3, 
        num_episodes=3,
        prefix=prefix,
        task=task,
    )

    # Combine callbacks into a single CallbackList
    combined_callback = CallbackList([checkpoint_callback, evaluate_callback])
    return combined_callback

class StageBasedTraining:
    def __init__(self, model, vec_env, callback=None, start_steps=0, num_envs=2):
        self.model = model
        self.vec_env = vec_env
        self.callback = callback
        self.start_steps = start_steps
        self.num_envs = num_envs

        # Define stage boundaries
        self.stage1_end = 40000 * num_envs
        self.stage2_end = 65000 * num_envs
        self.stage3_end = 85000 * num_envs

    def train(self):
        current_steps = self.start_steps

        # Stage 1: High exploration phase
        if current_steps < self.stage1_end:
            remaining_stage1 = self.stage1_end - current_steps
            print(f"Stage 1: Exploration phase (continuing from step {current_steps}, {remaining_stage1} steps remaining)")
            self.model.target_entropy = -2.0  # High exploration
            self.model.learning_rate = 1e-4   # Fast learning
            self.model.learn(remaining_stage1, callback=self.callback)
            current_steps = self.stage1_end
        else:
            print(f"Stage 1: Already completed (started from step {current_steps})")
        
        # Stage 2: Balanced phase 
        if current_steps < self.stage2_end:
            remaining_stage2 = self.stage2_end - current_steps
            print(f"Stage 2: Balanced phase (continuing from step {current_steps}, {remaining_stage2} steps remaining)")
            self.model.target_entropy = -3.0  # Low exploration
            self.model.learning_rate = 1e-4
            self.model.learn(remaining_stage2, callback=self.callback)
            current_steps = self.stage2_end
        else:
            print(f"Stage 2: Already completed (started from step {current_steps})")
        
        # Stage 3: Exploitation phase
        if current_steps < self.stage3_end:
            remaining_stage3 = self.stage3_end - current_steps
            print(f"Stage 3: Exploitation phase (continuing from step {current_steps}, {remaining_stage3} steps remaining)")
            self.model.target_entropy = -7.0  # Low exploration
            self.model.learning_rate = 5e-5   # Slow learning
            self.model.learn(remaining_stage3, callback=self.callback)
        else:
            print(f"Stage 3: Already completed (started from step {current_steps})")
            print("All training stages completed!")

def train_model(checkpoint_path, vec_env_stats_path, total_steps, save_freq, num_envs, prefix, task):
    """Main training function with optional checkpoint loading."""
    log_dir = "logs/sac_so100"  # will hold TB files
    
    # Create environment
    vec_env = create_environment(num_envs=num_envs, task=task)
    
    # Create or load model
    start_steps = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, vec_env, start_steps = load_checkpoint(checkpoint_path, vec_env_stats_path, vec_env, log_dir)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        model = create_model(vec_env, log_dir)
        print("Starting training from scratch")
    
    # Create callbacks
    combined_callback = create_callbacks(vec_env, save_freq, prefix, task)

    # Stage-based training with checkpoint awareness
    trainer = StageBasedTraining(model, vec_env, callback=combined_callback, start_steps=total_steps, num_envs=num_envs)
    trainer.train()

    # Save final model and environment stats
    model.save(prefix)
    vec_env.save(f"vec_normalize_stats_{prefix}.pkl")
    print("Training completed and model saved!")


def list_available_checkpoints(checkpoint_dir="./checkpoints/", prefix="sac_so100_get_cube_new"):
    """List available model checkpoints and their corresponding VecNormalize stats."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith(prefix) and file.endswith(".zip"):
            checkpoint_path = os.path.join(checkpoint_dir, file)
            
            # Look for corresponding VecNormalize stats
            import re
            match = re.search(r'(\d+)_steps\.zip$', file)
            if match:
                steps = match.group(1)
                vec_stats_file = f"vec_normalize_stats_{prefix}_{steps}.pkl"
                vec_stats_path = os.path.join(checkpoint_dir, vec_stats_file)
                vec_stats_exists = os.path.exists(vec_stats_path)
            else:
                vec_stats_path = None
                vec_stats_exists = False
            
            checkpoints.append({
                "checkpoint": checkpoint_path,
                "vec_stats": vec_stats_path if vec_stats_exists else None,
                "steps": steps if match else "unknown"
            })
    
    if checkpoints:
        print("Available checkpoints:")
        for cp in sorted(checkpoints, key=lambda x: int(x["steps"]) if x["steps"].isdigit() else 0):
            print(f"  Steps: {cp['steps']}")
            print(f"    Model: {cp['checkpoint']}")
            print(f"    VecNormalize: {cp['vec_stats'] if cp['vec_stats'] else 'Not found'}")
            print()
    else:
        print("No checkpoints found")
    
    return checkpoints


def main():
    """Main function with argument parsing for checkpoint loading."""
    parser = argparse.ArgumentParser(description="Train SAC model with optional checkpoint loading")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to model checkpoint to resume from (e.g., './checkpoints/sac_so100_pixels_agentpos_20000_steps.zip')"
    )
    parser.add_argument(
        "--vec-env-stats", 
        type=str, 
        default=None,
        help="Path to VecNormalize stats file (e.g., './checkpoints/vec_normalize_pixels_agentpos_stats_20000.pkl')"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=0,
        help="Start training steps (default: 0)"
    )
    parser.add_argument(
        "--num_envs", 
        type=int, 
        default=2,
        help="Total number of environments (default: 2)"
    )
    DEFAULT_TASK = "gym_so100/SO100TouchCube-v0"
    DEFAULT_PREFIX = "sac_so100_pixels_agentpos_new"
    parser.add_argument(
        "--prefix", 
        type=str, 
        default=DEFAULT_PREFIX,
        help="Prefix for the model name"
    )

    parser.add_argument(
        "--task", 
        type=str, 
        default=DEFAULT_TASK,
        help="Task for the model"
    )
    parser.add_argument(
        "--save-freq", 
        type=int, 
        default=1000,
        help="Frequency of saving checkpoints (default: 1000)"
    )
    parser.add_argument(
        "--list-checkpoints", 
        action="store_true",
        help="List available checkpoints and exit"
    )
    
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list_checkpoints:
        list_available_checkpoints(prefix=args.prefix)
        return
    
    # Auto-detect vec_env_stats path if not provided but checkpoint is given
    if args.checkpoint and not args.vec_env_stats:
        # Extract step number from checkpoint filename
        import re
        match = re.search(r'(\d+)_steps\.zip$', args.checkpoint)
        if match:
            steps = match.group(1)
            args.vec_env_stats = f"./checkpoints/vec_normalize_stats_{args.prefix}_{steps}.pkl"
            print(f"Auto-detected VecNormalize stats path: {args.vec_env_stats}")
    
    train_model(args.checkpoint, args.vec_env_stats, args.steps, args.save_freq, args.num_envs, args.prefix, args.task)


if __name__ == "__main__":
    main()

# Usage
# trainer = StageBasedTraining(model, env, callback=combined_callback)
# trainer.train()

# model.save("sac_so100_pixels_agentpos")
# vec_env.save("vec_normalize_stats_pixels_agentpos.pkl")

# 1. Train from scratch:
#    python train_sac.py
#
# 2. Resume from a specific checkpoint:
#    python train_sac.py --checkpoint ./checkpoints/sac_so100_pixels_agentpos_20000_steps.zip
#
# 3. Resume with custom VecNormalize stats:
#    python train_sac.py --checkpoint ./checkpoints/sac_so100_pixels_agentpos_20000_steps.zip --vec-env-stats ./checkpoints/vec_normalize_pixels_agentpos_stats_20000.pkl
#
# 4. List available checkpoints:
#    python -c "from train_sac import list_available_checkpoints; list_available_checkpoints()"
