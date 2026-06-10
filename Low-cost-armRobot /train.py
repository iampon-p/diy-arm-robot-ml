#!/usr/bin/env python3
"""
⭐ Multi-Robot Parallel Training Script

Trains a robotic arm using reinforcement learning with camera vision.
Supports a single robot with wrist-mounted camera (128x128 images) for pick-and-place tasks.
Features automatic checkpoints, real-time visualization, and TensorBoard monitoring.

Usage:
    python train.py                           # 100M steps with rendering
    python train.py --total_timesteps 10000000  # Custom timesteps
    python train.py --no-render                # Train headless (faster)
    python train.py --test_run --no-render     # Quick 1000-step test
"""

import os
import sys
import time
import random
import argparse
from typing import Dict, Optional
from pathlib import Path
import datetime
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Robot_Learning.robot_env import LowCostRobotEnv

# 📋 Set environment variables for compatibility
os.environ.update({
    'SYMPY_GROUND_TYPES': 'python',
    'USE_SYMENGINE': '0',
    'MPMATH_NOSAGE': '1',
    'MPMATH_NOGMPY': '1',
    'TORCH_COMPILE': '0',
    'PYTORCH_DISABLE_DYNAMO': '1',
    'TORCHDYNAMO_DISABLE': '1',
    'NUMPY_EXPERIMENTAL_ARRAY_FUNCTION': '0'
})

# 🛠️ Configure NumPy for stable output
np.set_printoptions(threshold=1000, edgeitems=3)

# 🧹 Remove gmpy2 if present to ensure pure Python
for module in ['gmpy2', 'gmpy']:
    sys.modules.pop(module, None)

print("✅ Compatibility patches applied")


class ProgressPercentCallback(BaseCallback):
    """📊 Displays training progress percentage in the terminal."""
    def __init__(self, total_timesteps: int, print_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.last_print = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        percent = (self.num_timesteps / self.total_timesteps) * 100
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        remaining_str = self._format_time((elapsed / (percent / 100)) - elapsed) if percent > 0 else "N/A"

        if self.num_timesteps - self.last_print >= self.print_freq or self.num_timesteps == self.total_timesteps:
            print(f"⏳ Progress: {percent:.2f}% ({self.num_timesteps:,}/{self.total_timesteps:,} steps) | "
                  f"Elapsed: {elapsed_str} | ETA: {remaining_str}")
            self.last_print = self.num_timesteps
        return True

    def _format_time(self, seconds: float) -> str:
        seconds = int(seconds)
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        return f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{s}s"


class ExplainedVarianceCallback(BaseCallback):
    """📈 Tracks and reports explained variance to monitor value function performance."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.explained_variances = []
        self.timesteps = []

    def _on_step(self) -> bool:
        return True
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value.get('train/explained_variance'):
            explained_var = self.model.logger.name_to_value['train/explained_variance']
            self.explained_variances.append(explained_var)
            self.timesteps.append(self.num_timesteps)

            if len(self.explained_variances) % 10 == 0 or len(self.explained_variances) <= 5:
                print(f"📊 Step {self.num_timesteps:,}: Explained Variance = {explained_var:.4f}")
                if explained_var > 0.8:
                    print("   🟢 Excellent value function performance!")
                elif explained_var > 0.5:
                    print("   🟡 Good value function learning.")
                elif explained_var > 0.0:
                    print("   🟠 Fair predictive power.")
                else:
                    print("   🔴 Poor value function performance.")
        return True

    def _on_training_end(self) -> None:
        if self.explained_variances:
            final_var = self.explained_variances[-1]
            avg_var = np.mean(self.explained_variances[-10:])
            print(f"\n📊 Explained Variance Summary:\n"
                  f"   Final: {final_var:.4f}\n"
                  f"   Recent Avg (last 10): {avg_var:.4f}\n"
                  f"   Total Measurements: {len(self.explained_variances)}")
            np.save("explained_variance_history.npy", {'timesteps': self.timesteps, 'explained_variances': self.explained_variances})
            print("   💾 Saved history to 'explained_variance_history.npy'")


class MultiRobotQuadCameraEnv(LowCostRobotEnv):
    """🤖 Environment for a single robot with camera vision for pick-and-place tasks."""
    def __init__(self, use_vision: bool = True, camera_width: int = 128, camera_height: int = 128,
                 render_width: int = 640, render_height: int = 480, num_robots: int = 1):
        self.num_robots = num_robots
        self.current_robot_id = 0
        self.camera_width, self.camera_height = camera_width, camera_height
        self.render_width, self.render_height = render_width, render_height
        self.use_vision = use_vision
        self.original_box_pos = np.array([0.0, 0.0, 0.035], dtype=np.float32)
        self.task_phase = "approach"
        self.episodes_completed = 0
        self.successful_cycles = 0

        super().__init__(use_vision=use_vision, camera_width=camera_width, camera_height=camera_height,
                         render_width=render_width, render_height=render_height, camera_name="robot_eye")

        if self.use_vision:
            try:
                from mujoco import GLContext
                self.gl_ctx = GLContext(self.camera_width, self.camera_height)
                self.gl_ctx.make_current()
                self.camera_renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
                print("✅ Camera renderer initialized!")
            except Exception as e:
                print(f"❌ Camera renderer failed: {e}. Vision disabled.")
                self.use_vision = False

        if self.use_vision:
            # Ensure observation dtypes are float32 to match Stable-Baselines3 expectations
            self.observation_space = spaces.Dict({
                "wrist_eye": spaces.Box(low=0.0, high=1.0, shape=(3, self.camera_height, self.camera_width), dtype=np.float32),
                "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(260,), dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Dict({
                "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(260,), dtype=np.float64)
            })

        self.robot_eye_id = -1
        self._setup_cameras()

    def _setup_cameras(self):
        """🎥 Initialize camera IDs for the robot."""
        camera_names = ["robot_eye", "top_view", "top_right_view"]
        camera_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in camera_names]
        self.robot_eye_id, self.top_view_id, self.top_right_view_id = camera_ids
        cameras_found = sum(1 for cid in camera_ids if cid != -1)
        if cameras_found == 0:
            print("❌ No cameras found! Disabling vision.")
            self.use_vision = False
        elif cameras_found < 3:
            print(f"⚠️ Only {cameras_found}/3 cameras found.")
        else:
            print("✅ All cameras initialized!")

    def _get_obs(self):
        """📸 Get observations with camera and joint positions."""
        qpos = np.nan_to_num(self.data.qpos.copy(), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        qpos = np.pad(qpos, (0, 260 - len(qpos)), mode='constant', constant_values=0).astype(np.float32)

        if not self.use_vision:
            return {"qpos": qpos}

        robot_eye_img = self._render_camera(self.robot_eye_id)
        if robot_eye_img is not None:
            # ensure channel-first and float32
            robot_eye_img = np.transpose(robot_eye_img, (2, 0, 1)).astype(np.float32) / 255.0
        else:
            robot_eye_img = np.zeros((3, self.camera_height, self.camera_width), dtype=np.float32)

        return {"wrist_eye": robot_eye_img, "qpos": qpos}

    def _render_camera(self, camera_id: int) -> Optional[np.ndarray]:
        """🎥 Render camera view, handling missing cameras gracefully."""
        if camera_id == -1 or not hasattr(self, 'camera_renderer'):
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        self.camera_renderer.update_scene(self.data, camera=camera_id)
        return np.flipud(self.camera_renderer.render())

    def _get_gripper_position(self) -> np.ndarray:
        """🤏 Get gripper position for the current robot."""
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Open_gripper")
        return self.data.xpos[gripper_id].copy() if gripper_id != -1 else np.zeros(3)

    def _get_box_position(self) -> np.ndarray:
        """📍 Get box position for the current robot."""
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        return self.data.xpos[box_id].copy() if box_id != -1 else self.original_box_pos.copy()

    def _is_box_grasped(self) -> bool:
        """✅ Check if the box is grasped by the gripper."""
        box_pos = self._get_box_position()
        gripper_pos = self._get_gripper_position()
        dist = np.linalg.norm(box_pos - gripper_pos)
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Open_gripper")
        gripper_closed = self.data.qpos[gripper_joint_id] < -0.2 if gripper_joint_id != -1 else False
        box_moved = np.linalg.norm(box_pos - self.original_box_pos) > 0.015
        return dist < 0.05 and gripper_closed and box_moved

    def _get_reward(self) -> float:
        """💰 Calculate reward for pick-and-place task."""
        box_pos = self._get_box_position()
        gripper_pos = self._get_gripper_position()
        drop_zone_pos = np.array([0.0, 0.8, 0.035], dtype=np.float32)  # Fixed drop zone
        dist_to_box = np.linalg.norm(gripper_pos - box_pos)
        reward = -dist_to_box * 2.0

        # Proximity bonuses
        if dist_to_box < 0.15:
            reward += 5.0
        if dist_to_box < 0.10:
            reward += 10.0
        if dist_to_box < 0.08:
            reward += 20.0
            print(f"📍 Close to box! Dist={dist_to_box:.3f}")

        # Movement incentives
        if hasattr(self, '_prev_gripper_pos'):
            movement = np.linalg.norm(gripper_pos - self._prev_gripper_pos)
            reward += movement * 0.5
            if dist_to_box < np.linalg.norm(self._prev_gripper_pos - box_pos):
                reward += 2.0
        self._prev_gripper_pos = gripper_pos.copy()

        # Gripper control
        gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Open_gripper")
        if gripper_joint_id != -1 and dist_to_box < 0.15:
            gripper_pos_val = self.data.qpos[gripper_joint_id]
            reward += max(0, (-0.1 - gripper_pos_val) * 2.0)

        # Success rewards
        if self._is_box_grasped():
            reward += 10.0
            print("🎯 Box grasped! +10 reward")
            dist_to_drop = np.linalg.norm(box_pos - drop_zone_pos)
            reward -= dist_to_drop * 1.0
            if dist_to_drop < 0.08:
                reward += 20.0
                print("🏆 Task completed! Box delivered!")
                self.successful_cycles += 1

        # Penalties
        if hasattr(self, '_was_grasped') and self._was_grasped and not self._is_box_grasped():
            reward -= 5.0
        self._was_grasped = self._is_box_grasped()
        self._step_count = getattr(self, '_step_count', 0) + 1
        reward -= 0.01  # Time penalty

        return np.clip(reward, -500.0, 500.0)

    def reset(self, **kwargs):
        """🔄 Reset environment for a new episode."""
        self.current_robot_id = 0
        if self.use_vision:
            self._setup_cameras()
        obs, info = super().reset(**kwargs)
        self.task_phase = "approach"
        self._prev_gripper_pos = self._get_gripper_position()
        self._step_count = 0
        self._was_grasped = False
        return obs, info

    def step(self, action):
        """🚀 Execute one step in the environment."""
        action = np.asarray(action, dtype=np.float32)
        actuator_names = ["Base_rotate", "Shoulder", "Shoulder_extension", "Elbow", "Rotate_Gripper", "Open_gripper"]
        joint_ranges = [(-3.14158, 3.14158), (-1.47958, 0.650), (-1.39158, 1.66158), (-1.554158, 1.77158),
                        (-3.14158, 3.14158), (-1.55, 0.035)]

        for i, name in enumerate(actuator_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id != -1 and i < len(action):
                joint_min, joint_max = joint_ranges[i]
                self.data.ctrl[actuator_id] = joint_min + (action[i] + 1.0) * (joint_max - joint_min) / 2.0

        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._get_reward()
        return obs, reward, False, False, {
            'robot_id': self.current_robot_id,
            'gripper_pos': self._get_gripper_position(),
            'box_pos': self._get_box_position(),
            'box_grasped': self._is_box_grasped()
        }


class RenderCallback(BaseCallback):
    """🎬 Renders the environment during training for real-time visualization."""
    def __init__(self, render_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self._step_count = 0
        self._viewer_initialized = False

    def _on_training_start(self):
        env = self.training_env.envs[0].env
        while hasattr(env, 'env'):
            env = env.env
        try:
            env.render(mode="human")
            if hasattr(env, 'viewer') and env.viewer is not None:
                self._viewer_initialized = True
                print("✅ MuJoCo viewer initialized!")
        except Exception as e:
            print(f"❌ Viewer initialization failed: {e}")
            self._viewer_initialized = False

    def _on_step(self) -> bool:
        self._step_count += 1
        if self._viewer_initialized and self._step_count % self.render_freq == 0:
            env = self.training_env.envs[0].env
            while hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.sync()
        return True


class CheckpointCallback(BaseCallback):
    """💾 Saves model checkpoints periodically."""
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "checkpoint", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self.save_path / f"{self.name_prefix}_{self.n_calls}_steps.zip"
            self.model.save(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        return True


class TrainingCallback(BaseCallback):
    """📈 Logs training metrics like rewards and episode lengths."""
    def _on_step(self) -> bool:
        # Periodically report episode stats
        if self.n_calls % 1000 == 0 and self.model.ep_info_buffer:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
            print(f"📊 Step {self.n_calls:,}: Avg Reward: {np.mean(ep_rewards):.2f} | "
                  f"Avg Length: {np.mean(ep_lengths):.1f} | Episodes: {len(ep_rewards)}")

        # Check for NaNs/Infs in policy parameters occasionally
        if self.n_calls % 500 == 0:
            try:
                params = []
                for p in self.model.policy.parameters():
                    if p is None:
                        continue
                    a = p.detach().cpu().numpy()
                    if not np.isfinite(a).all():
                        print(f"❌ NaN/Inf detected in policy parameters at step {self.n_calls}!")
                        # Save the model for debugging
                        path = Path('models') / f"nan_detected_{self.n_calls}.zip"
                        self.model.save(path)
                        print(f"💾 Saved model snapshot to {path}")
                        return False  # stop training
                    params.append(np.mean(a))
                if params:
                    print(f"🔬 Param mean (sample): {np.mean(params):.6f}")
            except Exception as e:
                print(f"⚠️ Param check failed: {e}")

        return True


def parse_args():
    """⚙️ Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a robotic arm with PPO in MuJoCo")
    parser.add_argument("--total_timesteps", type=int, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    parser.add_argument("--no-vision", action="store_true", help="Disable camera inputs")
    parser.add_argument("--test_run", action="store_true", help="Run a quick 1000-step test")
    return parser.parse_args()


def main():
    """🚀 Main training function."""
    args = parse_args()
    if args.test_run:
        args.total_timesteps = 1000
        print("🧪 Test run mode: 1000 timesteps")

    print("🤖 Single-Robot PPO Training\n" + "=" * 60)
    print(f"📊 Total timesteps: {args.total_timesteps:,}")
    print(f"🎮 Rendering: {'✅ Enabled' if args.render else '❌ Disabled'}")
    print(f"📹 Vision: {'✅ Enabled' if not args.no_vision else '❌ Disabled'}")

    # 🏗️ Initialize environment
    # Create a fresh env factory so that DummyVecEnv constructs correct env instances
    def make_env():
        e = MultiRobotQuadCameraEnv(use_vision=not args.no_vision, num_robots=1)
        return Monitor(e)

    env = DummyVecEnv([make_env])

    # Small wrapper to catch and sanitize NaN/Inf in observations for vectorized envs
    # Implemented as a simple class with the VecEnv API
    class VecObsSanitize:
        def __init__(self, vec_env):
            self.venv = vec_env

        def reset(self):
            obs = self.venv.reset()
            return self._sanitize_obs(obs)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.venv.step(action)
            return self._sanitize_obs(obs), reward, terminated, truncated, info

        def _sanitize_obs(self, obs):
            try:
                if isinstance(obs, dict) or hasattr(obs, 'items'):
                    new = {}
                    for k, v in obs.items():
                        a = np.array(v)
                        if not np.isfinite(a).all():
                            a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
                        new[k] = a.astype(np.float32)
                    return new
                else:
                    a = np.array(obs)
                    if not np.isfinite(a).all():
                        a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
                    return a.astype(np.float32)
            except Exception:
                return obs

        def __getattr__(self, name):
            """Pass attribute calls to the wrapped environment."""
            return getattr(self.venv, name)

    # Wrap then normalize
    env = VecObsSanitize(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 🧠 Initialize PPO model
    # Use more conservative hyperparameters for stability (lower LR, fewer steps/epochs)
    model = PPO(
        "MultiInputPolicy", env, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.8, max_grad_norm=0.5,
        verbose=1, tensorboard_log="./multi_robot_tensorboard/" if 'tensorboard' in sys.modules else None
    )

    # 💾 Setup callbacks
    callbacks = [
        CheckpointCallback(save_freq=100_000, save_path="models/checkpoints", name_prefix="multi_robot_ppo"),
        TrainingCallback(),
        ExplainedVarianceCallback(),
        ProgressPercentCallback(total_timesteps=args.total_timesteps)
    ]
    if args.render:
        callbacks.append(RenderCallback())

    # 🚀 Train model
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        print("🎉 Training completed!")
    except KeyboardInterrupt:
        print("⚠️ Training interrupted. Saving progress...")
    except Exception as e:
        print(f"❌ Training error: {e}. Saving progress...")

    # 💾 Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    final_path = models_dir / "ml-train.zip"
    model.save(final_path)
    print(f"💾 Model saved: {final_path}")
    backup_path = models_dir / f"ml-train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(backup_path)
    print(f"📦 Backup saved: {backup_path}")

    # 🧪 Test model
    print("🧪 Testing trained model...")
    test_obs = env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(test_obs, deterministic=True)
        test_obs, reward, done, _ = env.step(action)
        total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        if done.any() if isinstance(done, np.ndarray) else done:
            test_obs = env.reset()
    print(f"✅ Test completed - Average reward: {total_reward/100:.3f}")

    env.close()
    print("🏆 Training complete!")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()