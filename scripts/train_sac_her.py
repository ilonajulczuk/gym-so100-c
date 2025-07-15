from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from gymnasium.wrappers import RecordEpisodeStatistics
import os
import torch
import gym_so100
from gym_so100.env import SO100GoalEnv  # Import your goal environment
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import imageio
import numpy as np

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
        prefix="sac_so100_her",
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

def evaluate(model, num_episodes=10, deterministic=True, frames=None, task=None):
    """
    Evaluate using the original env.
    """
    if frames is None:
        frames = []
    
    
    eval_env = model.get_env()
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

    callback = create_callbacks(vec_env, save_freq=1000, prefix="sac_so100_her", task=None)
    model.learn(10000, callback=callback)