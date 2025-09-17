#!/usr/bin/env python3
"""
Demo of teleoperation using MuJoCo viewer.
This script allows you to control a robot arm in the SO100 environment using keyboard inputs.

It's using MOCAP to control the end effector position and orientation.

Does not record demonstrations, it's mostly to explore the mujoco scene.
"""

import mujoco
import mujoco.viewer
import numpy as np
import pyquaternion as pyq
import glfw

from gym_so100.constants import ASSETS_DIR

MOCAP_INDEX = 0


def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements


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
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [1, 0, 0], 10
        )
    elif key == 65:  # A key (rotate -10 around X)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [1, 0, 0], -10
        )

    # Rotation around Y-axis (Yaw)
    # Original: Home (268), End (269) - note: 268 was also Home previously, good to change anyway.
    # New: W (87), S (83)
    elif key == 87:  # W key (rotate +10 around Y)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 1, 0], 10
        )
    elif key == 83:  # S key (rotate -10 around Y)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 1, 0], -10
        )

    # Rotation around Z-axis (Roll)
    # Original: Page Up (266), Page Down (267)
    # New: E (69), D (68)
    elif key == 69:  # E key (rotate +10 around Z)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 0, 1], 10
        )
    elif key == 68:  # D key (rotate -10 around Z)
        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
            data.mocap_quat[MOCAP_INDEX], [0, 0, 1], -10
        )

    elif key == glfw.KEY_5:  # gripper open up
        data.ctrl[7] = -0.4

        finger1_id = 7  
    elif key == glfw.KEY_6:  # gripper close down
        data.ctrl[7] = 0.4
    else:
        print(key)


def main():
    xml_path = ASSETS_DIR / "pandas_transfer_cube_ee.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Add this to your code to check the model setup
    print("=== MODEL DIAGNOSTICS ===")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of joints: {model.nq}")
    print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")
    print(f"Joint names: {[model.joint(i).name for i in range(model.njnt)]}")

    # Check if finger joints exist
    try:
        finger1_id = model.joint("finger_joint1").id
        finger2_id = model.joint("finger_joint2").id
        print(f"Finger joint IDs: finger1={finger1_id}, finger2={finger2_id}")
    except:
        print("ERROR: Finger joints not found!")

    # Check actuator 7 (index 7)
    if model.nu > 7:
        print(f"Actuator 7 name: {model.actuator(7).name}")
        print(f"Actuator 7 controls joint: {model.actuator(7).trnid}")
    else:
        print("ERROR: Actuator 7 doesn't exist!")
        
    def key_callback(key):
        key_callback_data(key, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
