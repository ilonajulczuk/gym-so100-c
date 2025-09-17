import collections

import numpy as np
from dm_control.suite import base

from gym_so100.constants import SO100_START_ARM_POSE, unnormalize_so100

BOX_POSE = [None]  # to be changed from outside

"""
Environment for simulated robot one arm manipulation, with joint position control
Action space:      [left_arm_qpos (5),             # absolute joint position
                    left_gripper_positions (1),    # absolute gripper position
                    

Observation space: {"qpos": Concat[ left_arm_qpos (5),         # absolute joint position
                                    left_gripper_position (1),  # absolute gripper position
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # absolute gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""


class SO100Task(base.Task):
    ARM_DOF = 5
    GRIPPER_DOF = 2  # Dunno why 2 ???

    def __init__(self, random=None, observation_width=640, observation_height=480):
        self.observation_width = observation_width
        self.observation_height = observation_height
        super().__init__(random=random)

    def before_step(self, action, physics):
        action = action.copy()
        left_arm_action = action[: self.ARM_DOF + 1]
        env_action = unnormalize_so100(left_arm_action)
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[: SO100Task.ARM_DOF + SO100Task.GRIPPER_DOF]
        left_arm_qpos = left_qpos_raw[: SO100Task.ARM_DOF]
        left_gripper_qpos = [left_qpos_raw[SO100Task.ARM_DOF]]
        return np.concatenate([left_arm_qpos, left_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[: SO100Task.ARM_DOF + SO100Task.GRIPPER_DOF]
        left_arm_qvel = left_qvel_raw[: SO100Task.ARM_DOF]
        left_gripper_qvel = [left_qvel_raw[SO100Task.ARM_DOF]]
        return np.concatenate([left_arm_qvel, left_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def _precompute_bin_aabb(self, physics):
        # call in reset(); store on self
        site_id = physics.model.site("bin_center").id
        center = physics.data.site_xpos[site_id].copy()
        self.bin_center = center
        hw = 0.06  # half-width in xy  (edit to match XML)
        h = 0.03  # inner height      (edit to match XML)
        self.bin_min = center + np.array([-hw, -hw, 0.0])
        self.bin_max = center + np.array([hw, hw, h])
        self.bin_center = center
        self.bin_radius = hw  # for bonus
        self.cube_half = 0.01  # edge/2  (match cube size)

    def _cube_inside_bin(self, cube_pos):
        lower = cube_pos - self.cube_half
        upper = cube_pos + self.cube_half
        return np.all(lower > self.bin_min) and np.all(upper < self.bin_max)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(
            height=self.observation_height,
            width=self.observation_width,
            camera_id="top",
        )
        obs["images"]["angle"] = physics.render(
            height=self.observation_height,
            width=self.observation_width,
            camera_id="angle",
        )
        obs["images"]["vis"] = physics.render(
            height=self.observation_height,
            width=self.observation_width,
            camera_id="front_close",
        )

        self._precompute_bin_aabb(physics)
        id_cube_site = physics.model.site("cube_site").id
        cube_pos = physics.data.site_xpos[id_cube_site]

        id_ee_site = physics.model.site("ee_site").id
        ee_pos = physics.data.site_xpos[id_ee_site]
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)
        obs["box_position"] = cube_pos.astype(np.float32)  # SO100 uses float32
        obs["bin_position"] = self.bin_center.astype(np.float32)  # SO100 uses float32
        obs["ee_position"] = ee_pos.astype(np.float32)  #
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class SO100TouchCubeTask(SO100Task):
    """Actions are normalized to [-1, 1] range. Observations are not, to be used with VecNormalize"""

    def __init__(self, random=None, observation_width=640, observation_height=480):
        super().__init__(
            random=random,
            observation_width=observation_width,
            observation_height=observation_height,
        )
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:6] = SO100_START_ARM_POSE
            np.copyto(physics.data.ctrl, SO100_START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[6:]
        return env_state

    def get_reward(self, physics):
        # -------- helpers --------

        self._precompute_bin_aabb(physics)
        id_cube_site = physics.model.site("cube_site").id
        cube_pos = physics.data.site_xpos[id_cube_site]

        id_ee_site = physics.model.site("ee_site").id
        ee_pos = physics.data.site_xpos[id_ee_site]
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)

        CUBE_GEOM = "red_box"
        TABLE_GEOM = "table"
        FIXED_FINGER_GEOMS = {f"fixed_jaw_pad_{i}" for i in range(1, 5)}
        MOVING_FINGER_GEOMS = {f"moving_jaw_pad_{i}" for i in range(1, 5)}
        FINGERTIP_GEOMS = FIXED_FINGER_GEOMS | MOVING_FINGER_GEOMS

        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_gripper = any(
            (g1 in FINGERTIP_GEOMS and g2 == CUBE_GEOM)
            or (g2 in FINGERTIP_GEOMS and g1 == CUBE_GEOM)
            for (g1, g2) in all_contact_pairs
        )

        touch_table = (CUBE_GEOM, TABLE_GEOM) in all_contact_pairs

        cube_over_bin = (self.bin_min[0] < cube_pos[0] < self.bin_max[0]) and (
            self.bin_min[1] < cube_pos[1] < self.bin_max[1]
        )

        inside_bin = self._cube_inside_bin(cube_pos)
        released = inside_bin and (not touch_gripper)

        reward = 0.0

        # Multi-stage distance rewards (smoother progression)
        if ee_cube_dist < 0.7:
            reward = max(reward, 0.1 * (1 - ee_cube_dist / 0.7))
        if ee_cube_dist < 0.5:
            reward = max(reward, 0.2 * (1 - ee_cube_dist / 0.5))
        if ee_cube_dist < 0.3:
            reward = max(reward, 0.5 * (1 - ee_cube_dist / 0.3))
        if ee_cube_dist < 0.1:  # NEW: bridge the gap
            reward = max(reward, 1.0 * (1 - ee_cube_dist / 0.1))
        if ee_cube_dist < 0.05:
            reward = max(reward, 2.0 * (1 - ee_cube_dist / 0.05))

        # Add contact bonus (already have the code!)
        if touch_gripper:
            reward += 1.0  # Big bonus for actually touching

        success = touch_gripper and ee_cube_dist < 0.05
        if success:
            print("SUCCESS!")
            return self.max_reward

        reward -= 0.2
        return reward


class SO100TouchCubeSparseTask(SO100Task):
    """Actions are normalized to [-1, 1] range. Observations are not, to be used with VecNormalize"""

    def __init__(self, random=None, observation_width=640, observation_height=480):
        super().__init__(
            random=random,
            observation_width=observation_width,
            observation_height=observation_height,
        )
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:6] = SO100_START_ARM_POSE
            np.copyto(physics.data.ctrl, SO100_START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[6:]
        return env_state

    def get_reward(self, physics):
        # -------- helpers --------

        self._precompute_bin_aabb(physics)
        id_cube_site = physics.model.site("cube_site").id
        cube_pos = physics.data.site_xpos[id_cube_site]

        id_ee_site = physics.model.site("ee_site").id
        ee_pos = physics.data.site_xpos[id_ee_site]
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)

        CUBE_GEOM = "red_box"
        FIXED_FINGER_GEOMS = {f"fixed_jaw_pad_{i}" for i in range(1, 5)}
        MOVING_FINGER_GEOMS = {f"moving_jaw_pad_{i}" for i in range(1, 5)}
        FINGERTIP_GEOMS = FIXED_FINGER_GEOMS | MOVING_FINGER_GEOMS

        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_gripper = any(
            (g1 in FINGERTIP_GEOMS and g2 == CUBE_GEOM)
            or (g2 in FINGERTIP_GEOMS and g1 == CUBE_GEOM)
            for (g1, g2) in all_contact_pairs
        )

        reward = 0.0
        success = touch_gripper and ee_cube_dist < 0.05
        if success:
            print("SUCCESS!")
            return self.max_reward

        reward -= 0.2
        return reward


class SO100CubeToBinTask(SO100Task):
    """Actions are normalized to [-1, 1] range. Observations are not, to be used with VecNormalize"""

    def __init__(self, random=None, observation_width=640, observation_height=480):
        super().__init__(
            random=random,
            observation_width=observation_width,
            observation_height=observation_height,
        )
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:6] = SO100_START_ARM_POSE
            np.copyto(physics.data.ctrl, SO100_START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[6:]
        return env_state

    def get_cube_position(self, physics):
        """Get the position of the cube in the environment."""
        id_cube_site = physics.model.site("cube_site").id
        cube_pos = physics.data.site_xpos[id_cube_site]
        return cube_pos.astype(np.float32)

    def get_reward(self, physics):
        # -------- helpers --------

        self._precompute_bin_aabb(physics)
        cube_pos = self.get_cube_position(physics)

        id_ee_site = physics.model.site("ee_site").id
        ee_pos = physics.data.site_xpos[id_ee_site]
        ee_cube_dist = np.linalg.norm(ee_pos - cube_pos)

        CUBE_GEOM = "red_box"
        TABLE_GEOM = "table"
        FIXED_FINGER_GEOMS = {f"fixed_jaw_pad_{i}" for i in range(1, 5)}
        MOVING_FINGER_GEOMS = {f"moving_jaw_pad_{i}" for i in range(1, 5)}
        FINGERTIP_GEOMS = FIXED_FINGER_GEOMS | MOVING_FINGER_GEOMS

        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_gripper = any(
            (g1 in FINGERTIP_GEOMS and g2 == CUBE_GEOM)
            or (g2 in FINGERTIP_GEOMS and g1 == CUBE_GEOM)
            for (g1, g2) in all_contact_pairs
        )

        touch_table = (CUBE_GEOM, TABLE_GEOM) in all_contact_pairs

        cube_over_bin = (self.bin_min[0] < cube_pos[0] < self.bin_max[0]) and (
            self.bin_min[1] < cube_pos[1] < self.bin_max[1]
        )

        inside_bin = self._cube_inside_bin(cube_pos)
        released = inside_bin and (not touch_gripper)

        reward = 0.0

        if touch_gripper:
            reward = 1.0
            print("Touched gripper!")
        if touch_gripper and not touch_table:  # lifted
            reward = 2
            print("Lifted!")
        if cube_over_bin:
            reward = 2.5
            print("Cube over bin!")
        if inside_bin:
            reward = 3
            print("Inside bin!")
        if released:
            reward = 4.0

        return reward
