#!/usr/bin/env python3
"""
Low Cost Robot Environment

Basic MuJoCo environment for robotic arm simulation.
Provides core functionality for robot control and observation.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class LowCostRobotEnv(gym.Env):
    """Basic MuJoCo environment for low-cost robot simulation."""

    def __init__(self, use_vision=True, camera_width=128, camera_height=128,
                 render_width=640, render_height=480, camera_name="robot_eye"):
        super().__init__()

        # Store parameters
        self.use_vision = use_vision
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.render_width = render_width
        self.render_height = render_height
        self.camera_name = camera_name

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent / "Simulation" / "ArmRobot" / "armrobot.xml"
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found at {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Set up action space (6 joints: Base_rotate, Shoulder, Shoulder_extension, Elbow, Rotate_Gripper, Open_gripper)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Set up observation space (will be overridden by subclasses)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float32)

        # Initialize viewer
        self.viewer = None

        print(f"✅ LowCostRobotEnv initialized with model: {model_path}")

    def _get_obs(self):
        """Get current observation (joint positions)."""
        return self.data.qpos.copy().astype(np.float32)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial joint positions slightly
        if seed is not None:
            np.random.seed(seed)

        # Small randomization of joint positions
        for i in range(self.model.njnt):
            if i < self.model.jnt_range.shape[0]:
                joint_range = self.model.jnt_range[i]
                if joint_range[0] < joint_range[1]:  # Valid range
                    center = (joint_range[0] + joint_range[1]) / 2
                    width = (joint_range[1] - joint_range[0]) * 0.1  # 10% of range
                    # Find the corresponding qpos index
                    qpos_addr = self.model.jnt_qposadr[i]
                    if qpos_addr >= 0 and qpos_addr < self.model.nq:
                        self.data.qpos[qpos_addr] = center + np.random.uniform(-width, width)

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        # Apply action to actuators
        action = np.clip(action, -1.0, 1.0)

        # Map actions to actuators (assuming 6 actuators)
        for i in range(min(len(action), self.model.nu)):
            self.data.ctrl[i] = action[i]

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation
        observation = self._get_obs()

        # Calculate reward (basic implementation - override in subclasses)
        reward = 0.0

        # Check if episode is done (basic implementation)
        terminated = False
        truncated = False

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        elif mode == "rgb_array":
            # Return RGB array for offscreen rendering
            if not hasattr(self, 'renderer'):
                self.renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """Close the environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None