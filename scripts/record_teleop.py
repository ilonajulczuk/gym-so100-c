# record_teleop.py is used to record learning episodes interacting with the gym_so100 envs.
# The interaction is done via controller 8BitDO Pro 2 or a keyboard.

import gymnasium as gym
import numpy as np
import gym_so100
import cv2
import time
import pickle
import signal
import sys
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

import argparse

from gym_so100.teleop.gamepad_utils import GamepadControllerHID


# Key mapping for joint control - converted from the commented code above
# Format: key_code: (joint_index, delta_value)
DEFAULT_JOINT_DELTAS = {
    # Position controls (X, Y, Z axes)
    0: (1, -0.01),  # Up arrow - Arm joint (+)
    1: (1, 0.01),  # Down arrow - Arm joint (-)
    2: (0, 0.01),  # Left arrow - rotate around base (X axis - left)
    3: (0, -0.01),  # Right arrow - rotate around base (X axis - right)
    61: (2, 0.01),  # + key - Forearm (+)
    45: (2, -0.01),  # - key - Forearm (-)
    # Wrist angle (up down)
    113: (3, -0.01),  # Q key - up
    97: (3, 0.01),  # A key - down
    # Wrist rotation
    119: (4, 0.01),  # W
    115: (4, -0.01),  # S
    # Gripper controls
    53: (5, 0.05),  # 5 key - gripper open
    54: (5, -0.05),  # 6 key - gripper close
}


class KeyJointController:
    # Maintains the internal state of the joints and uses keyboard input to modify them.
    # Takes a mapping of key presses to joint deltas.
    def __init__(self, joint_deltas, dof=6):
        self.joint_deltas = joint_deltas
        self.joint_state = np.zeros(dof)

    def reset(self):
        """Reset the joint state to zero"""
        self.joint_state = np.zeros_like(self.joint_state)
        return self.joint_state

    def handle_keyboard(self, key):
        """Handle keyboard input for joint control"""
        print("Key pressed:", key)
        if key in self.joint_deltas:
            joint_index, delta = self.joint_deltas[key]
            self.joint_state[joint_index] += delta
            self.joint_state[joint_index] = np.clip(
                self.joint_state[joint_index], -1.0, 1.0
            )  # Assuming joint limits are [-1, 1]
            print(f"Updated joint state: {self.joint_state}")
            print("Current joint state:", self.joint_state)
            return True
        return False

    def get_joint_state(self):
        """Get the current joint state"""
        return self.joint_state.copy()


class GamepadJointController:

    def __init__(self, dof=6):
        self.gamepad_input = GamepadControllerHID()

        self.joint_state = np.zeros(dof)

    def start(self):
        return self.gamepad_input.start()

    def reset(self):
        """Reset the joint state to zero"""
        self.joint_state = np.zeros_like(self.joint_state)
        return self.joint_state

    def update(self):
        """Update the joint state based on gamepad input"""
        # Read gamepad input and update joint_state accordingly
        # This is a placeholder implementation
        self.gamepad_input.update()
        data = self.gamepad_input.get_all_data()

        def direction_to_delta(direction):
            if direction == "left":
                return -1
            elif direction == "right":
                return 1
            return 0

        self.joint_state[0] += -data["left_x"] * 0.03  # X axis
        self.joint_state[1] += data["left_y"] * 0.03  # Y axis
        self.joint_state[2] += data["right_x"] * 0.03  # Z axis
        self.joint_state[3] += data["right_y"] * 0.03  # Wrist angle
        self.joint_state[4] += (
            direction_to_delta(data["direction"]) * 0.01
        )  # Wrist rotation
        self.joint_state[5] += data["lt"] * 0.1  # Open Gripper
        self.joint_state[5] += -data["rt"] * 0.1  # Close Gripper

        self.joint_state = np.clip(self.joint_state, -1.0, 1.0)
        return self.joint_state

    def get_joint_state(self):
        """Get the current joint state"""
        return self.joint_state.copy()


def create_single_env(task):
    """Create a single environment for evaluation."""
    env = gym.make(
        task,
        obs_type="so100_pixels_agent_pos",
        observation_width=640,
        observation_height=480,
    )

    env = RecordEpisodeStatistics(env)
    return env


def create_environment(num_envs, task):
    """Create environment with macOS subprocess fixes."""
    vec_env = make_vec_env(
        create_single_env,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"task": task},
    )
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=255,
    )

    return vec_env


class TeleoperationRecorder:
    def __init__(
        self,
        controller_type,
        time_between_steps,
        time_between_episodes,
        auto_record,
        env_id="gym_so100/SO100CubeToBin-v0",
        num_episodes=10,
    ):
        self.controller_type = controller_type
        self.env = create_environment(1, env_id)

        self.time_between_steps = time_between_steps
        self.time_between_episodes = time_between_episodes
        self.time_to_next_episode = time_between_episodes
        self.auto_record = auto_record
        self.num_episodes = num_episodes
        self.episode_count = 0
        # Storage for expert demonstrations
        self.demonstrations = []
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "infos": [],
        }

        self.step = 0
        # Control state
        self.action = np.zeros(self.env.action_space.shape)
        self.recording = False
        self.paused = False
        self.running = True  # Flag to control main loop

        # Initialize joint controller if requested

        if controller_type == "keyboard":
            print("Using keyboard controller")
            self.joint_controller = KeyJointController(DEFAULT_JOINT_DELTAS)
        elif controller_type == "gamepad":
            print("Using gamepad controller")
            self.joint_controller = GamepadJointController()
            if not self.joint_controller.start():
                print(
                    "Failed to start gamepad controller, using keyboard control instead."
                )
                self.joint_controller = KeyJointController(DEFAULT_JOINT_DELTAS)
                self.controller_type = "keyboard"
        else:
            self.joint_controller = None

        # Display settings
        self.display_scale = 8  # Scale up the small observation for better visibility
        self.window_name = "Teleoperation - SO100 TouchCube"

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

        print("Teleoperation Controls:")
        print("R: Reset environment")
        print("SPACE: Start/Stop recording")
        print("P: Pause/Resume")
        print("ESC: Exit")
        print("S: Save demonstrations")
        print("Ctrl+C: Safe exit with cleanup")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal"""
        print("\nReceived Ctrl+C, shutting down gracefully...")
        self.running = False

    def handle_keyboard(self, key):
        """Handle keyboard input for teleoperation"""
        # Try joint controller first if available
        if (
            self.controller_type == "keyboard"
            and self.joint_controller.handle_keyboard(key)
        ):
            # Joint controller handled the key, update action from joint state
            self.action = self.joint_controller.get_joint_state()
            return None

        # Control commands
        elif key == ord("r"):  # Reset
            return "reset"
        elif key == ord(" "):  # Space - toggle recording
            return "toggle_recording"
        elif key == ord("p"):  # Pause
            return "pause"
        elif key == 27:  # ESC
            return "exit"
        elif key == ord("c"):  # Save
            return "save"

        return None

    def start_recording_episode(self):
        """Start recording a new episode"""
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "infos": [],
        }
        self.recording = True
        print("Started recording episode")

    def stop_recording_episode(self):
        """Stop recording and save the episode"""
        if self.recording and len(self.current_episode["observations"]) > 0:
            self.demonstrations.append(self.current_episode.copy())
            print(
                f"Saved episode with {len(self.current_episode['observations'])} steps"
            )
            print(f"Total episodes recorded: {len(self.demonstrations)}")
        self.time_to_next_episode = self.time_between_episodes
        self.recording = False

    def save_demonstrations(self, filename="expert_demonstrations.pkl"):
        """Save all recorded demonstrations to file"""
        if self.demonstrations:
            with open(filename, "wb") as f:
                pickle.dump(self.demonstrations, f)
            print(f"Saved {len(self.demonstrations)} demonstrations to {filename}")
        else:
            print("No demonstrations to save")

    def annotate_image(self, image):
        # Convert to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Scale up for better visibility
        display_image = cv2.resize(
            image_bgr,
            (
                image.shape[1] * self.display_scale,
                image.shape[0] * self.display_scale,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        # Add status text
        status_text = []
        if self.recording:
            status_text.append("RECORDING")
        if self.paused:
            status_text.append("PAUSED")

        if self.time_to_next_episode > 0:
            status_text.append(f"Next episode in {self.time_to_next_episode:.2f}s")

        status_str = " | ".join(status_text) if status_text else "READY"
        cv2.putText(
            display_image,
            status_str,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Show episode count
        cv2.putText(
            display_image,
            f"Episodes: {len(self.demonstrations)}, steps: {self.step}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return display_image

    def run(self):
        """Main teleoperation loop"""
        observation, info = self.env.reset()

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running and self.episode_count < self.num_episodes:
                if not self.paused and self.time_to_next_episode <= 0:
                    # Take action in environment
                    self.action = self.action.copy().squeeze().reshape(1, -1)

                    observation, reward, done, info = self.env.step(self.action)
                    terminated = done
                    truncated = False

                    self.step += 1
                    # Record data if recording
                    if self.recording:
                        self.current_episode["observations"].append(observation)
                        self.current_episode["actions"].append(self.action.copy())
                        self.current_episode["rewards"].append(reward)
                        self.current_episode["infos"].append(info)

                    # Handle episode termination
                    if terminated or truncated:
                        print(
                            f"Episode ended, truncated: {truncated}, terminated: {terminated}"
                        )
                        if self.recording:
                            self.stop_recording_episode()
                        observation, info = self.env.reset()
                        self.step = 0  # Reset step count
                        self.episode_count += 1
                        self.joint_controller.reset()  # Reset joint controller state

                # Render and display
                image = self.env.render()
                if image is not None:
                    display_image = self.annotate_image(image)
                    cv2.imshow(self.window_name, display_image)

                if self.controller_type == "gamepad":
                    # Update joint state from gamepad input
                    self.joint_controller.update()
                    self.action = self.joint_controller.get_joint_state()

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    command = self.handle_keyboard(key)

                    if command == "reset":
                        if self.recording:
                            self.stop_recording_episode()
                        observation, info = self.env.reset()
                        print("Environment reset")

                    elif command == "toggle_recording":
                        if self.recording:
                            self.stop_recording_episode()
                        else:
                            self.start_recording_episode()

                    elif command == "pause":
                        self.paused = not self.paused
                        print("Paused" if self.paused else "Resumed")
                        if self.auto_record and not self.paused:
                            print("Auto-recording resumed")
                            self.start_recording_episode()

                    elif command == "save":
                        self.save_demonstrations()

                    elif command == "exit":
                        break

                # Small delay to prevent excessive CPU usage
                time.sleep(self.time_between_steps)
                if self.time_to_next_episode > 0:
                    self.time_to_next_episode -= self.time_between_steps
                    if self.auto_record and self.time_to_next_episode < 0:
                        self.start_recording_episode()

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught, shutting down gracefully...")

        finally:
            # Clean up
            print("Performing cleanup...")
            if self.recording:
                self.stop_recording_episode()
                print("Stopped recording and saved current episode")
            cv2.destroyAllWindows()
            self.env.close()
            print("Cleanup completed")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Record expert demonstrations via teleoperation"
    )
    parser.add_argument(
        "--controller_type",
        type=str,
        default="keyboard",
        help="Type of controller to use (e.g., 'keyboard', 'gamepad')",
    )
    parser.add_argument(
        "--time_between_steps",
        type=float,
        default=0.01,
        help="Time to wait between steps (default: 0.01)",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )

    parser.add_argument(
        "--time_between_episodes",
        type=float,
        default=0.1,
        help="Time to wait between episodes (default: 0.1)",
    )

    parser.add_argument(
        "--auto_record",
        type=bool,
        default=False,
        help="Start auto recording episodes (default: False)",
    )

    args = parser.parse_args()

    try:
        teleop = TeleoperationRecorder(
            controller_type=args.controller_type,
            time_between_steps=args.time_between_steps,
            time_between_episodes=args.time_between_episodes,
            auto_record=args.auto_record,
            num_episodes=args.num_episodes,
        )
        teleop.run()

        # Save final demonstrations
        teleop.save_demonstrations("expert_demonstrations.pkl")
        print("Teleoperation session ended")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
