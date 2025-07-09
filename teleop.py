
import mujoco
import mujoco.viewer
import numpy as np
import pyquaternion as pyq
import glfw

# from dm_control import mujoco
from gym_so100.constants import ASSETS_DIR
MOCAP_INDEX = 0

from gym_so100.constants import unnormalize_so100, SO100_START_ARM_POSE, normalize_so100

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def key_callback_data_joint_actions(pose, key, data):
    """
    Callback for key presses but with data passed in
    :param key: Key pressed
    :param data:  MjData object
    :return: None
    """

    print("joints", chr(key))
    if key == 263:  # Left arrow - rotate around base
        print("Left arrow pressed")
        pose[0] += 0.01
    elif key == 262:  # Right arrow
        pose[0] -= 0.01
    elif key == 265:  # Up arrow - rotate around shoulder
        pose[1] -= 0.01
    elif key == 264:  # Down arrow
        pose[1] += 0.01
    elif key == 61:  # +
        pose[2] += 0.01
    elif key == 45:  # -
        pose[2] -= 0.01
    elif key == glfw.KEY_V:
        pose[3] += 0.01
    elif key == glfw.KEY_B: 
        pose[3] -= 0.01
    elif key == glfw.KEY_G:  # - wrist rotate
        pose[4] += 0.01
    elif key == glfw.KEY_H:  # - wrist rotate
        pose[4] -= 0.01
    elif key == glfw.KEY_5:  # gripper
        pose[5] += 0.01
    elif key == glfw.KEY_6:  #
        pose[5] -= 0.01

    print("Updated pose:", pose)

    # data.qpos[:6] = pose
    env_action = unnormalize_so100(pose.copy())
    np.copyto(data.ctrl, env_action)
    
    # action = data.ctrl.copy()  # Get the current control signal
    # left_arm_action = action[:6]
    # env_action = unnormalize_so100(left_arm_action)
    

def main():
    # Load the mujoco model basic.xml
    xml_path = ASSETS_DIR / "so100_transfer_cube.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # physics = mujoco.Physics.from_xml_path(str(xml_path))
    pose = np.array(normalize_so100(SO100_START_ARM_POSE), dtype=np.float32)
    print("Initial pose:", pose)
    def key_callback(key):
        key_callback_data_joint_actions(pose, key, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
