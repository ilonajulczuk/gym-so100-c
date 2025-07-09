import mujoco
import mujoco.viewer
import numpy as np
import pyquaternion as pyq
import time
import glfw # Still used for key codes in key_callback, but mainly for the viewer setup
from inputs import get_gamepad # Import the gamepad library

# from dm_control import mujoco
from gym_so100.constants import ASSETS_DIR
MOCAP_INDEX = 0

from gym_so100.constants import unnormalize_so100, SO100_START_ARM_POSE

# Global variables for MuJoCo model/data and teleoperation state
model = None
data = None
MOCAP_INDEX = 0
jaw_target_qpos = 0.0 # Global to hold the jaw's desired position

# --- Helper Function for Quaternion Rotation ---
def rotate_quaternion(quat_mj, axis, angle_degrees):
    """
    Rotate a MuJoCo quaternion (w,x,y,z) by an angle (degrees) around an axis.
    Returns a MuJoCo-formatted quaternion (w,x,y,z).
    """
    q_current = pyq.Quaternion(quat_mj[0], quat_mj[1], quat_mj[2], quat_mj[3])
    angle_rad = np.deg2rad(angle_degrees)
    axis_norm = np.array(axis) / np.linalg.norm(axis)
    q_rotation = pyq.Quaternion(axis=axis_norm, angle=angle_rad)
    q_new = q_rotation * q_current
    return np.array([q_new.w, q_new.x, q_new.y, q_new.z])


# --- Keyboard Callback (for non-gamepad specific actions like reset) ---
def key_callback_keyboard(key): # Renamed to avoid conflict with gamepad logic
    """
    Callback for keyboard presses, primarily for simulation-wide actions like reset.
    """
    global MOCAP_INDEX, jaw_target_qpos, model, data

    # --- Reset Simulation (still keyboard for convenience) ---
    if key == glfw.KEY_R: # R key to reset
        print("Resetting simulation...")
        mujoco.mj_resetData(model, data) # Resets physics state

        # --- CORRECTED MOCAP RESET LOGIC ---
        if model.nmocap > 0: # Check if any mocap bodies are defined in the model
            # Find the body ID corresponding to MOCAP_INDEX
            # Assuming MOCAP_INDEX directly corresponds to the mocap body's index
            # in the list of mocap bodies (which is usually the case if only one exists)

            # Reset mocap position/orientation to its initial value from the XML
            # These initial values are stored in model.body_pos and model.body_quat
            data.mocap_pos[MOCAP_INDEX] = [0, 0, 0.5]
            data.mocap_quat[MOCAP_INDEX] = [1, 0, 0, 0]  # Reset to identity quaternion
            
        else:
            print("No mocap bodies found in the model to reset.")
    # --- Unhandled Key Logging ---
    else:
        try:
            print(f"Unhandled keyboard key: '{chr(key)}' (code: {key})")
        except ValueError: # For non-printable keys
            print(f"Unhandled keyboard key code: {key}")


# --- Main Simulation Loop ---
def main():
    global model, data, jaw_target_qpos # Declare globals

    xml_path = ASSETS_DIR / "so100_transfer_cube_ee.xml" # Ensure this XML has mocap and weld
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Initialize jaw_target_qpos from the initial arm pose
    try:
        jaw_actuator_id = model.actuator("Jaw").id
        jaw_target_qpos = SO100_START_ARM_POSE[model.joint("Jaw").id]
        print(f"Initial Jaw target set to: {jaw_target_qpos:.3f}")
    except KeyError:
        print("Warning: 'Jaw' joint or actuator not found. Jaw control might not work.")
        jaw_target_qpos = 0.0 # Default if not found

    # Set initial arm qpos based on SO100_START_ARM_POSE
    num_arm_joints = model.nu
    if len(SO100_START_ARM_POSE) >= num_arm_joints:
        data.qpos[:num_arm_joints] = SO100_START_ARM_POSE[:num_arm_joints]
    else:
        print("Warning: SO100_START_ARM_POSE has fewer elements than robot joints/actuators. Using partial initialization.")

    # Step once to apply initial qpos and setup
    mujoco.mj_step(model, data)

    # Gamepad sensitivity (adjust these values as needed)
    TRANSLATION_SPEED = 0.005 # Meters per step for joystick deflection
    ROTATION_SPEED = 5.0      # Degrees per step for joystick deflection
    GRIPPER_SPEED = 0.005     # Radians per step for gripper triggers

    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback_keyboard) as viewer:
        print("\n--- MuJoCo Gamepad Controls (Mocap Arm + Direct Jaw) ---")
        print("Mocap Target Control (End-effector):")
        print("  Translate X/Z: Left Analog Stick (X, Y)")
        print("  Translate Y: Right Analog Stick (Y)")
        print("  Rotate Z (Roll): Right Analog Stick (X)")
        print("  Rotate X (Pitch): A / B Buttons")
        print("  Rotate Y (Yaw): X / Y Buttons") # Or adjust based on your gamepad
        print("Jaw Control:")
        print("  Open Jaw: Left Trigger (Analog)")
        print("  Close Jaw: Right Trigger (Analog)")
        print("General:")
        print("  Reset Simulation: R (Keyboard)")
        print("-------------------------------------------------------------\n")

        while viewer.is_running():
            step_start = time.time()

            # --- Gamepad Input Processing ---
            try:
                events = get_gamepad()
                for event in events:
                    # print(f"Event: Code={event.code}, State={event.state}") # Uncomment for debugging gamepad input

                    # Mocap Translation (Left Stick: LX, LY)
                    if event.code == 'ABS_X': # Left Stick X-axis
                        data.mocap_pos[MOCAP_INDEX, 0] += event.state * TRANSLATION_SPEED / 32768.0 # Normalize joystick range
                    elif event.code == 'ABS_Y': # Left Stick Y-axis (often inverted)
                        # We map Y-axis to MuJoCo Z-direction (vertical)
                        data.mocap_pos[MOCAP_INDEX, 2] -= event.state * TRANSLATION_SPEED / 32768.0 # Invert for intuitive control

                    # Mocap Translation Y (Right Stick: RY)
                    elif event.code == 'ABS_RY': # Right Stick Y-axis
                        # We map RY-axis to MuJoCo Y-direction (forward/backward)
                        data.mocap_pos[MOCAP_INDEX, 1] += event.state * TRANSLATION_SPEED / 32768.0

                    # Mocap Rotation Z (Roll) (Right Stick: RX)
                    elif event.code == 'ABS_RX': # Right Stick X-axis
                        if abs(event.state) > 5000: # Add a small deadzone
                            # Scale joystick input to rotation speed
                            data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(
                                data.mocap_quat[MOCAP_INDEX], [0, 0, 1], -event.state * ROTATION_SPEED / 32768.0
                            )

                    # Mocap Rotations X (Pitch) and Y (Yaw) (Face Buttons)
                    # Common Xbox/PS Button mapping:
                    # BTN_SOUTH (A/Cross): Pitch +X
                    # BTN_EAST (B/Circle): Pitch -X
                    # BTN_WEST (X/Square): Yaw +Y
                    # BTN_NORTH (Y/Triangle): Yaw -Y
                    elif event.code == 'BTN_SOUTH' and event.state == 1: # A button (Xbox) / Cross (PS)
                        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [1, 0, 0], ROTATION_SPEED)
                    elif event.code == 'BTN_EAST' and event.state == 1: # B button (Xbox) / Circle (PS)
                        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [1, 0, 0], -ROTATION_SPEED)
                    elif event.code == 'BTN_WEST' and event.state == 1: # X button (Xbox) / Square (PS)
                        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 1, 0], ROTATION_SPEED)
                    elif event.code == 'BTN_NORTH' and event.state == 1: # Y button (Xbox) / Triangle (PS)
                        data.mocap_quat[MOCAP_INDEX] = rotate_quaternion(data.mocap_quat[MOCAP_INDEX], [0, 1, 0], -ROTATION_SPEED)

                    # Jaw Control (Triggers)
                    # ABS_Z (Left Trigger), ABS_RZ (Right Trigger) for Xbox-style controllers
                    elif event.code == 'ABS_Z': # Left Trigger (Open Jaw)
                        if event.state > 0: # Trigger pressed
                            try:
                                jaw_joint_id = model.joint("Jaw").id
                                jaw_range = model.jnt_range[jaw_joint_id]
                                # Scale trigger value (0-255) to a small step
                                jaw_target_qpos = np.clip(jaw_target_qpos + GRIPPER_SPEED * (event.state / 255.0), jaw_range[0], jaw_range[1])
                            except KeyError: pass
                    elif event.code == 'ABS_RZ': # Right Trigger (Close Jaw)
                        if event.state > 0: # Trigger pressed
                            try:
                                jaw_joint_id = model.joint("Jaw").id
                                jaw_range = model.jnt_range[jaw_joint_id]
                                # Scale trigger value (0-255) to a small step
                                jaw_target_qpos = np.clip(jaw_target_qpos - GRIPPER_SPEED * (event.state / 255.0), jaw_range[0], jaw_range[1])
                            except KeyError: pass

            except EOFError: # No gamepad connected or gamepad disconnected
                # This error often occurs if get_gamepad() is called when no gamepad is found
                # print("No gamepad detected. Using keyboard for reset only.") # Uncomment for debugging
                pass # Continue without gamepad input
            except Exception as e:
                # Catch other potential errors from inputs library (e.g., specific driver issues)
                # print(f"Error reading gamepad: {e}") # Uncomment for debugging
                pass

            # --- Apply Direct Joint Control (for Jaw only) ---
            # The 'Jaw' actuator needs to be explicitly set.
            # The other arm joints are controlled by the IK weld constraint.
            try:
                jaw_actuator_id = model.actuator("Jaw").id
                data.ctrl[jaw_actuator_id] = jaw_target_qpos
            except KeyError:
                pass # If 'Jaw' actuator not found, just skip

            # --- Physics Step ---
            mujoco.mj_step(model, data)

            # --- Viewer Sync ---
            viewer.sync()

            # --- Maintain Real-Time ---
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()