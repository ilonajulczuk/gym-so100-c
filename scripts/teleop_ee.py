
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


import mujoco
import mujoco.viewer
import numpy as np
import pyquaternion as pyq

# from dm_control import mujoco
from gym_so100.constants import ASSETS_DIR
MOCAP_INDEX = 0

from gym_so100.constants import unnormalize_so100, SO100_START_ARM_POSE

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


def key_callback_data(key, data):
    """
    Callback for key presses but with data passed in
    :param key: Key pressed
    :param data:  MjData object
    :return: None
    """
    global MOCAP_INDEX
    print(chr(key))
    if key == 265:  # Up arrow - Y axis (+)
        data.mocap_pos[MOCAP_INDEX, 2] += 0.01
    elif key == 264:  # Down arrow - Y axis (-)
        data.mocap_pos[MOCAP_INDEX, 2] -= 0.01
    elif key == 263:  # Left arrow - Z axis (-)
        data.mocap_pos[MOCAP_INDEX, 0] -= 0.01
    elif key == 262:  # Right arrow - Z axis (+)
        data.mocap_pos[MOCAP_INDEX, 0] += 0.01
    elif key == 61:  # + - Y axis (+)
        data.mocap_pos[MOCAP_INDEX, 1] += 0.01
    elif key == 45:  # - - Y axis (-)
        data.mocap_pos[MOCAP_INDEX, 1] -= 0.01
    # Rotation around X-axis (Pitch)
    # Original: Insert (260), Home (261)
    # New: Q (81), A (65)
    elif key == 81:  # Q key (rotate +10 around X)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [1, 0, 0], 10)
    elif key == 65:  # A key (rotate -10 around X)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [1, 0, 0], -10)

    # Rotation around Y-axis (Yaw)
    # Original: Home (268), End (269) - note: 268 was also Home previously, good to change anyway.
    # New: W (87), S (83)
    elif key == 87:  # W key (rotate +10 around Y)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 1, 0], 10)
    elif key == 83:  # S key (rotate -10 around Y)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 1, 0], -10)

    # Rotation around Z-axis (Roll)
    # Original: Page Up (266), Page Down (267)
    # New: E (69), D (68)
    elif key == 69:  # E key (rotate +10 around Z)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 0, 1], 10)
    elif key == 68:  # D key (rotate -10 around Z)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 0, 1], -10)

    elif key == glfw.KEY_5:  # gripper open up
        data.ctrl[5] += 0.05
    elif key == glfw.KEY_6:  # gripper close down
        data.ctrl[5] -= 0.05
    else:
        print(key)


def main():
    # Load the mujoco model basic.xml
    xml_path = ASSETS_DIR / "so100_transfer_cube_ee.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    def key_callback(key):
        key_callback_data(key, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
