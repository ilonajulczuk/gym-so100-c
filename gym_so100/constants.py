from pathlib import Path
import numpy as np
### Simulation envs fixed constants
DT = 0.02  # 0.02 ms -> 1/0.2 = 50 hz
FPS = 50


SO100_JOINTS = [
    # absolute joint position
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_rotate",
    # normalized gripper position 0: close, 1: open
    "left_arm_gripper",
]

SO100_ACTIONS = [
    # position and quaternion for end effector
    "left_arm_waist",
    "left_arm_shoulder",
    "left_arm_elbow",
    "left_arm_forearm_roll",
    "left_arm_wrist_rotate",
    # normalized gripper position (0: close, 1: open)
    "left_arm_gripper",
]


bin_min = np.array([-0.25, 0.7, 0.01], dtype=np.float32)
bin_max = np.array([-0.14, 0.76, 0.05], dtype=np.float32)

SO100_START_ARM_POSE = [
    0.0,  # left_arm_waist
    -0.96,  # left_arm_shoulder
    1.16,  # left_arm_elbow
    0.0,  # left_arm_forearm_roll
    0.0,  # left_arm_wrist_rotate
    0.02239,  # left_arm_gripper
]

ASSETS_DIR = Path(__file__).parent.resolve() / "assets"  # note: absolute path


def unnormalize(num, min_val, max_val):
    """Scale action from [-1, 1] to [min_val, max_val] with clipping"""
    scaled = (num + 1) / 2 * (max_val - min_val) + min_val
    return np.clip(scaled, min_val, max_val)

def normalize_so100(action):
    """Normalize the action to [-1, 1] range"""
    action[0] = normalize(action[0], -1.92, 1.92)  # rotation around the waist
    action[1] = normalize(action[1], -3.32, 0.174)
    action[2] = normalize(action[2], -0.174, 3.14)  # elbow
    action[3] = normalize(action[3], -1.66, 1.66)  # wrist pitch
    action[4] = normalize(action[4], -2.79, 2.79)  # wrist roll
    action[5] = normalize(action[5], -0.174, 1.75)  # gripper position
    return action

def normalize(num, min_val, max_val):
    """Scale action from [min_val, max_val] to [-1, 1]"""
    if min_val == max_val:
        return 0.0  # Avoid division by zero
    scaled = (num - min_val) / (max_val - min_val) * 2 - 1
    return np.clip(scaled, -1, 1)


def unnormalize_so100(action):
    action[0] = unnormalize(action[0], -1.92, 1.92)  # rotation around the waist
    action[1] = unnormalize(action[1], -3.32, 0.174)
    action[2] = unnormalize(action[2], -0.174, 3.14)  # elbow
    action[3] = unnormalize(action[3], -1.66, 1.66)  # wrist pitch
    action[4] = unnormalize(action[4], -2.79, 2.79)  # wrist roll
    action[5] = unnormalize(action[5], -0.174, 1.75)  # gripper position
    # normalize the action to [-1, 1] range
    return action