#!/usr/bin/env python3
"""
Real-time AI Robot Testing Script

This script loads a trained PPO model and lets the AI control a single robot arm in real-time.
Unlike the hardcoded demo, this uses the trained neural network to make decisions
based on camera input and robot state.

Usage:
    mjpython test_model.py                    # Shows MuJoCo GUI viewer (default)
    mjpython test_model.py --test-cameras     # Test all 4 camera views
    mjpython test_model.py --opencv-viewer    # OpenCV GUI with AI testing
    mjpython test_model.py --headless         # Text-only output
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import shutil
import os
import sys
import os

# Add current directory to path first
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Apply compatibility patches FIRST before any other imports
try:
    import gmpy2
    if not hasattr(gmpy2, 'version'):
        gmpy2.version = '2.1.2'
        print("✅ Applied gmpy2 compatibility patch")
except ImportError:
    pass

# Handle mjpython import issues on macOS
try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    from stable_baselines3 import PPO
    # Import from the correct training script
    sys.path.append(str(PROJECT_ROOT.parent))  # Add parent directory to path
    from train import MultiRobotQuadCameraEnv as QuadCameraRobotEnv
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Solution: Make sure you're running with 'mjpython' and all packages are installed")
    print("   Try: conda install -c conda-forge stable-baselines3")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./models/ml-train.zip"),
        help="Path to the trained PPO model",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to run (default: infinite)",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=3000,
        help="Maximum steps per episode (default: 3000 = ~60 seconds)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video recordings of AI performance",
    )
    parser.add_argument(
        "--save-camera-images",
        action="store_true",
        help="Save camera images during AI execution",
    )
    parser.add_argument(
        "--visualize-ai-vision",
        action="store_true",
        help="Save combined visualizations showing what the AI sees from both cameras",
    )
    parser.add_argument(
        "--show-ai-thinking",
        action="store_true",
        help="Display AI action values and decision process",
    )
    parser.add_argument(
        "--render-cameras",
        action="store_true",
        help="Render camera views in separate windows",
    )
    parser.add_argument(
        "--test-all-models",
        action="store_true",
        help="Test and compare all models in the models folder (single robot only)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Ultra-fast simulation mode (100x speed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI viewer (text-only output)",
    )
    parser.add_argument(
        "--test-cameras",
        action="store_true",
        help="Test camera views only (shows camera feeds for a few seconds)",
    )
    parser.add_argument(
        "--opencv-viewer",
        action="store_true",
        default=False,
        help="Use OpenCV windows instead of MuJoCo viewer (fallback for terminals)",
    )
    return parser.parse_args()


def save_ai_camera_images(env: QuadCameraRobotEnv, episode: int, step: int) -> None:
    """Save camera images from AI's perspective"""
    try:
        import cv2
        
        if env.use_vision:
            robot_eye_img = env._get_camera_image(env.robot_eye_id, "robot_eye")
            robot_eye_img = env._get_camera_image(env.robot_eye_id, "robot_eye")
            
            if robot_eye_img is not None:
                robot_eye_bgr = cv2.cvtColor(robot_eye_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"ai_robot_eye_ep{episode:03d}_step{step:04d}.png", robot_eye_bgr)
            
            if robot_eye_img is not None:
                robot_eye_bgr = cv2.cvtColor(robot_eye_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"ai_robot_eye_ep{episode:03d}_step{step:04d}.png", robot_eye_bgr)
                
    except ImportError:
        print("⚠️ opencv-python not installed. Cannot save images.")
    except Exception as e:
        print(f"⚠️ Failed to save AI camera images: {e}")


def visualize_ai_vision(env: QuadCameraRobotEnv, episode: int, step: int, action: np.ndarray) -> None:
    """Create a combined visualization showing what the AI sees from both cameras + additional scene cameras"""
    try:
        import cv2
        import mujoco
        
        if not env.use_vision:
            return
        
        # Flatten action array if it's wrapped (from VecEnv)
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.flatten()
        elif isinstance(action, np.ndarray) and len(action) > 0:
            action = action.flatten()
            
        robot_eye_img = env._get_camera_image(env.robot_eye_id, "robot_eye")
        
        # Try to get additional cameras from scene
        top_view_img = None
        top_right_img = None
        
        try:
            top_view_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view")
            if top_view_id != -1:
                top_view_img = env._get_camera_image(top_view_id, "top_view")
        except:
            pass
            
        try:
            top_right_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_right_view")
            if top_right_id != -1:
                top_right_img = env._get_camera_image(top_right_id, "top_right_view")
        except:
            pass
        
        # Check if we have all three cameras
        if all(img is not None for img in [robot_eye_img, top_view_img, top_right_img]):
            # TRI CAMERA MODE - All 3 cameras available
            size = (150, 150)
            robot_resized = cv2.resize(robot_eye_img, size, interpolation=cv2.INTER_NEAREST)
            top_resized = cv2.resize(top_view_img, size, interpolation=cv2.INTER_NEAREST)
            top_right_resized = cv2.resize(top_right_img, size, interpolation=cv2.INTER_NEAREST)
            
            # Create 2x2 grid with robot_eye in top-left, top_view in top-right, top_right in bottom-left
            top_row = np.hstack([robot_resized, top_resized])
            bottom_row = np.hstack([top_right_resized, np.zeros_like(top_right_resized)])  # Empty bottom-right
            combined = np.vstack([top_row, bottom_row])
            
            # Convert to BGR for OpenCV
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            
            # Add labels for tri camera view
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_bgr, "Robot Eye", (5, 15), font, 0.4, (0, 255, 0), 1)
            cv2.putText(combined_bgr, "Top View", (155, 15), font, 0.4, (0, 255, 0), 1)
            cv2.putText(combined_bgr, "Top Right", (5, 165), font, 0.4, (0, 255, 0), 1)
            cv2.putText(combined_bgr, f"TRI CAM | Ep {episode} Step {step}", (50, 290), font, 0.35, (255, 255, 255), 1)
            # Format action safely
            action_str = f"Action: [{float(action[0]):.1f},{float(action[1]):.1f},{float(action[2]):.1f}...]"
            cv2.putText(combined_bgr, action_str, (10, 310), font, 0.3, (255, 255, 0), 1)
            
            # Save tri camera visualization
            cv2.imwrite(f"ai_vision_ep{episode:03d}_step{step:04d}_TRI.png", combined_bgr)
            print(f"      📸 Saved TRI camera vision at step {step}")
            
        elif robot_eye_img is not None:
            # Single camera mode - just show robot_eye
            robot_eye_large = cv2.resize(robot_eye_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            # Convert to BGR for OpenCV
            combined_bgr = cv2.cvtColor(robot_eye_large, cv2.COLOR_RGB2BGR)
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_bgr, "Robot Eye (Mounted)", (10, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(combined_bgr, f"Episode {episode} | Step {step}", (10, 245), font, 0.4, (255, 255, 255), 1)
            
            # Add action info (safely format)
            action_text = f"Action: [{float(action[0]):.2f}, {float(action[1]):.2f}, {float(action[2]):.2f}]"
            cv2.putText(combined_bgr, action_text, (10, 270), font, 0.35, (255, 255, 0), 1)
            
            # Save single camera visualization
            cv2.imwrite(f"ai_vision_ep{episode:03d}_step{step:04d}.png", combined_bgr)
            
            print(f"      📸 Saved robot eye vision at step {step}")
                
    except ImportError:
        pass  # OpenCV not available
    except Exception as e:
        print(f"⚠️ Failed to visualize AI vision: {e}")


def display_ai_thinking(action: np.ndarray, observation: dict, step: int, base_env=None) -> None:
    """Display what the AI is thinking and doing"""
    print(f"\n🧠 AI Step {step}:")
    print(f"   Action: {action}")

    if isinstance(observation, dict):
        if 'qpos' in observation:
            # Show joint positions from qpos (first 7 joints are positions)
            qpos = observation['qpos']
            if hasattr(qpos, 'shape') and len(qpos.shape) > 0:
                joint_positions = qpos[0][:7] if qpos.shape[0] > 0 else qpos[:7]
                print(f"   Joint positions: {joint_positions}")

        if 'robot_eye' in observation:
            robot_shape = observation['robot_eye'].shape
            print(f"   Vision: robot_eye{robot_shape}")

    # Always show ASCII visualization if base_env is provided
    if base_env:
        create_ascii_visualization(base_env, step)
    """Display what the AI is thinking and doing"""
    print(f"\n🧠 AI Step {step}:")
    print(f"   Action: {action}")

    if isinstance(observation, dict):
        if 'qpos' in observation:
            # Show joint positions from qpos (first 7 joints are positions)
            qpos = observation['qpos']
            if hasattr(qpos, 'shape') and len(qpos.shape) > 0:
                joint_positions = qpos[0][:7] if qpos.shape[0] > 0 else qpos[:7]
                print(f"   Joint positions: {joint_positions}")

        if 'robot_eye' in observation and 'robot_eye' in observation:
            robot_shape = observation['robot_eye'].shape
            wrist_shape = observation['robot_eye'].shape
            print(f"   Vision: robot_eye{robot_shape}, robot_eye{wrist_shape}")

    # Always show ASCII visualization if base_env is provided
    if base_env:
        create_ascii_visualization(base_env, step)
def create_ascii_visualization(base_env, step: int) -> None:
    """Create a simple ASCII visualization of the robot and box positions"""
    try:
        # Get positions
        gripper_pos = base_env._get_gripper_position()
        box_pos = base_env._get_box_position()

        # Try to get drop zone position, use default if not available
        try:
            if hasattr(base_env, 'current_robot_id'):
                drop_zone_pos = base_env._get_drop_zone_pos(base_env.current_robot_id)
            else:
                drop_zone_pos = base_env._get_drop_zone_pos(0)  # Default to robot 0
        except:
            # Default drop zone position if method fails
            drop_zone_pos = np.array([0.15, 0.05, 0.05])

        # Calculate distance
        dist_to_box = np.linalg.norm(gripper_pos - box_pos)
        dist_to_drop = np.linalg.norm(box_pos - drop_zone_pos)

        # Check if box is grasped
        grasped = base_env._is_box_grasped()

        # Create simple 2D visualization (top-down view)
        print(f"\n📍 Step {step} - Robot Status:")
        print(f"   🤖 Gripper: ({gripper_pos[0]:.2f}, {gripper_pos[1]:.2f}, {gripper_pos[2]:.2f})")
        print(f"   📦 Box: ({box_pos[0]:.2f}, {box_pos[1]:.2f}, {box_pos[2]:.2f})")
        print(f"   🎯 Drop Zone: ({drop_zone_pos[0]:.2f}, {drop_zone_pos[1]:.2f}, {drop_zone_pos[2]:.2f})")
        print(f"   📏 Distance to box: {dist_to_box:.3f}m")
        print(f"   📏 Distance to drop: {dist_to_drop:.3f}m")
        print(f"   ✋ Grasped: {'YES' if grasped else 'NO'}")

        # Simple ASCII art
        print("   [Scene Top View]")
        print("        🎯")
        print("         |")
        if dist_to_box < 0.1:  # Close enough
            if grasped:
                print("   🤖✋📦  (Robot holding box!)")
            else:
                print("   🤖📦  (Robot near box!)")
        else:
            print("   🤖    📦")

    except Exception as e:
        print(f"   ⚠️ Visualization error: {e}")


def test_camera_views(base_env) -> None:
    """Test and display all 4 camera views for a few seconds"""
    try:
        import cv2
        import time
        import mujoco

        print("📹 Testing all camera views...")
        print("   Available cameras: robot_eye, top_view, top_right_view")

        # Reset environment to get initial state
        base_env.reset()

        # Test for 10 seconds or until user presses 'q'
        start_time = time.time()
        frames_shown = 0

        while time.time() - start_time < 10:  # 10 seconds
            # Test all cameras
            cameras = [
                ('robot_eye', mujoco.mj_name2id(base_env.model, mujoco.mjtObj.mjOBJ_CAMERA, 'robot_eye')),
                ('top_view', mujoco.mj_name2id(base_env.model, mujoco.mjtObj.mjOBJ_CAMERA, 'top_view')),
                ('top_right_view', mujoco.mj_name2id(base_env.model, mujoco.mjtObj.mjOBJ_CAMERA, 'top_right_view')),
            ]

            # Create status info
            status_img = np.zeros((250, 500, 3), dtype=np.uint8)
            status_img.fill(50)

            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            cv2.putText(status_img, "Quad Camera Test Mode", (10, y_offset), font, 0.8, (255, 255, 255), 2)
            y_offset += 35

            # Display camera status
            for cam_name, cam_id in cameras:
                status = "✅" if cam_id != -1 else "❌"
                cv2.putText(status_img, f"{status} {cam_name}: {'Available' if cam_id != -1 else 'Not found'}",
                           (10, y_offset), font, 0.5, (0, 255, 0) if cam_id != -1 else (0, 0, 255), 1)
                y_offset += 25

            y_offset += 10
            cv2.putText(status_img, f"Frames shown: {frames_shown}", (10, y_offset), font, 0.5, (0, 255, 0), 1)
            y_offset += 25
            cv2.putText(status_img, "Press 'q' to exit", (10, y_offset), font, 0.5, (0, 255, 255), 1)
            y_offset += 25
            cv2.putText(status_img, f"Time: {time.time() - start_time:.1f}s / 10.0s", (10, y_offset), font, 0.5, (255, 255, 0), 1)

            # Display status window
            cv2.imshow("Quad Camera Test Status", status_img)

            # Display camera windows with positioning to avoid overlap
            window_positions = {
                'robot_eye': (50, 50),
                'top_view': (350, 50), 
                'top_right_view': (650, 50),
                'robot_eye': (950, 50)
            }
            
            for cam_name, cam_id in cameras:
                if cam_id != -1:
                    cam_img = base_env._render_camera(cam_id)
                    if cam_img is not None:
                        cam_large = cv2.resize(cam_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                        cam_bgr = cv2.cvtColor(cam_large, cv2.COLOR_RGB2BGR)
                        cv2.putText(cam_bgr, cam_name, (10, 20), font, 0.5, (0, 255, 0), 1)
                        
                        # Position windows to avoid overlap
                        if cam_name in window_positions:
                            cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
                            cv2.moveWindow(cam_name, *window_positions[cam_name])
                        
                        cv2.imshow(cam_name, cam_bgr)
                        print(f"📺 Displaying window: {cam_name} at position {window_positions.get(cam_name, 'default')}")
                    else:
                        print(f"⚠️ No image for camera: {cam_name}")

            frames_shown += 1

            # Check for 'q' key press
            key = cv2.waitKey(100) & 0xFF  # Wait 100ms
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print(f"✅ Quad camera test completed! Showed {frames_shown} frames over {time.time() - start_time:.1f} seconds")
        print("   Cameras tested: robot_eye, top_view, top_right_view")

    except ImportError:
        print("❌ OpenCV not available for camera testing")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        try:
            cv2.destroyAllWindows()
        except:
            pass


def create_opencv_viewer(base_env, episode: int, step: int, action: np.ndarray, reward: float) -> bool:
    """Create a single comprehensive GUI viewer using OpenCV window with all cameras and status"""
    try:
        import cv2

        # Get all camera images
        robot_eye_img = base_env._render_camera(base_env.robot_eye_id)
        top_view_img = base_env._render_camera(base_env.top_view_id)
        top_right_view_img = base_env._render_camera(base_env.top_right_view_id)

        # Get position information
        gripper_pos = base_env._get_gripper_position()
        box_pos = base_env._get_box_position()
        dist_to_box = np.linalg.norm(gripper_pos - box_pos)
        grasped = base_env._is_box_grasped()

        # Convert positions to lists for safe formatting
        gripper_list = gripper_pos.tolist() if hasattr(gripper_pos, 'tolist') else gripper_pos
        box_list = box_pos.tolist() if hasattr(box_pos, 'tolist') else box_pos

        # Camera dimensions - use higher resolution for better quality
        cam_width, cam_height = 240, 240

        # Create a large combined window (2x2 grid: 3 cameras + status)
        window_width = cam_width * 2
        window_height = cam_height * 2
        combined_img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        combined_img.fill(20)  # Dark background

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Top-left: Robot Eye Camera
        if robot_eye_img is not None:
            robot_large = cv2.resize(robot_eye_img, (cam_width, cam_height), interpolation=cv2.INTER_LINEAR)
            robot_bgr = cv2.cvtColor(robot_large, cv2.COLOR_RGB2BGR)
            cv2.putText(robot_bgr, "Robot Eye", (10, 20), font, 0.5, (0, 255, 0), 1)
            combined_img[0:cam_height, 0:cam_width] = robot_bgr
        else:
            # Placeholder if camera not available
            placeholder = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            placeholder.fill(100)
            cv2.putText(placeholder, "Robot Eye", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(placeholder, "N/A", (cam_width//2-20, cam_height//2), font, 1.0, (255, 255, 255), 2)
            combined_img[0:cam_height, 0:cam_width] = placeholder

        # Top-right: Top View Camera
        if top_view_img is not None:
            top_large = cv2.resize(top_view_img, (cam_width, cam_height), interpolation=cv2.INTER_LINEAR)
            top_bgr = cv2.cvtColor(top_large, cv2.COLOR_RGB2BGR)
            cv2.putText(top_bgr, "Top View", (10, 20), font, 0.5, (0, 255, 0), 1)
            combined_img[0:cam_height, cam_width:window_width] = top_bgr
        else:
            # Placeholder if camera not available
            placeholder = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            placeholder.fill(100)
            cv2.putText(placeholder, "Top View", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(placeholder, "N/A", (cam_width//2-20, cam_height//2), font, 1.0, (255, 255, 255), 2)
            combined_img[0:cam_height, cam_width:window_width] = placeholder

        # Bottom-left: Top Right View Camera
        if top_right_view_img is not None:
            top_right_large = cv2.resize(top_right_view_img, (cam_width, cam_height), interpolation=cv2.INTER_LINEAR)
            top_right_bgr = cv2.cvtColor(top_right_large, cv2.COLOR_RGB2BGR)
            cv2.putText(top_right_bgr, "Top Right", (10, 20), font, 0.5, (0, 255, 0), 1)
            combined_img[cam_height:window_height, 0:cam_width] = top_right_bgr
        else:
            # Placeholder if camera not available
            placeholder = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            placeholder.fill(100)
            cv2.putText(placeholder, "Top Right", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(placeholder, "N/A", (cam_width//2-20, cam_height//2), font, 1.0, (255, 255, 255), 2)
            combined_img[cam_height:window_height, 0:cam_width] = placeholder

        # Bottom-right: Status information
        status_img = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
        status_img.fill(50)  # Dark gray background

        # Add text information
        y_offset = 25
        cv2.putText(status_img, f"Ep{episode} St{step}", (10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(status_img, f"Reward:{float(reward):.1f}", (10, y_offset), font, 0.4, (0, 255, 0), 1)
        y_offset += 20
        cv2.putText(status_img, f"G:({gripper_list[0]:.2f},{gripper_list[1]:.2f},{gripper_list[2]:.2f})", (10, y_offset), font, 0.35, (255, 255, 0), 1)
        y_offset += 18
        cv2.putText(status_img, f"B:({box_list[0]:.2f},{box_list[1]:.2f},{box_list[2]:.2f})", (10, y_offset), font, 0.35, (255, 255, 0), 1)
        y_offset += 18
        cv2.putText(status_img, f"Dist:{float(dist_to_box):.3f}m", (10, y_offset), font, 0.4, (255, 255, 0), 1)
        y_offset += 20
        cv2.putText(status_img, f"Grasp:{'YES' if grasped else 'NO'}", (10, y_offset), font, 0.4, (0, 255, 0) if grasped else (0, 0, 255), 1)
        y_offset += 22

        # Action values (compact format)
        try:
            if hasattr(action, 'flatten'):
                action_vals = action.flatten()[:3]  # Take first 3 values
            else:
                action_vals = action[:3] if len(action) >= 3 else action
            cv2.putText(status_img, f"A:[{float(action_vals[0]):.2f},{float(action_vals[1]):.2f},{float(action_vals[2]):.2f}]", (10, y_offset), font, 0.35, (255, 0, 255), 1)
        except Exception as e:
            cv2.putText(status_img, "Action:Error", (10, y_offset), font, 0.35, (255, 0, 255), 1)

        # Add status panel title
        cv2.putText(status_img, "STATUS", (10, 15), font, 0.6, (255, 255, 255), 2)

        combined_img[cam_height:window_height, cam_width:window_width] = status_img

        # Display the combined window
        cv2.imshow("Robot AI Control Panel - Camera & Status", combined_img)

        # Handle keyboard input (press 'q' to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False

        return True

    except ImportError:
        print("⚠️ OpenCV not available for GUI viewer")
        return False
    except Exception as e:
        print(f"⚠️ OpenCV viewer error: {e}")
        return False
    """Create a simple ASCII visualization of the robot and box positions"""
    try:
        # Get positions
        gripper_pos = base_env._get_gripper_position()
        box_pos = base_env._get_box_position()
        
        # Try to get drop zone position, use default if not available
        try:
            if hasattr(base_env, 'current_robot_id'):
                drop_zone_pos = base_env._get_drop_zone_pos(base_env.current_robot_id)
            else:
                drop_zone_pos = base_env._get_drop_zone_pos(0)  # Default to robot 0
        except:
            # Default drop zone position if method fails
            drop_zone_pos = np.array([0.15, 0.05, 0.05])

        # Calculate distance
        dist_to_box = np.linalg.norm(gripper_pos - box_pos)
        dist_to_drop = np.linalg.norm(box_pos - drop_zone_pos)

        # Check if box is grasped
        grasped = base_env._is_box_grasped()

        # Create simple 2D visualization (top-down view)
        print(f"\n📍 Step {step} - Robot Status:")
        print(f"   🤖 Gripper: ({gripper_pos[0]:.2f}, {gripper_pos[1]:.2f}, {gripper_pos[2]:.2f})")
        print(f"   📦 Box: ({box_pos[0]:.2f}, {box_pos[1]:.2f}, {box_pos[2]:.2f})")
        print(f"   🎯 Drop Zone: ({drop_zone_pos[0]:.2f}, {drop_zone_pos[1]:.2f}, {drop_zone_pos[2]:.2f})")
        print(f"   📏 Distance to box: {dist_to_box:.3f}m")
        print(f"   📏 Distance to drop: {dist_to_drop:.3f}m")
        print(f"   ✋ Grasped: {'YES' if grasped else 'NO'}")

        # Simple ASCII art
        print("   [Scene Top View]")
        print("        🎯")
        print("         |")
        if dist_to_box < 0.1:  # Close enough
            if grasped:
                print("   🤖✋📦  (Robot holding box!)")
            else:
                print("   🤖📦  (Robot near box!)")
        else:
            print("   🤖    📦")

    except Exception as e:
        print(f"   ⚠️ Visualization error: {e}")


def display_episode_stats(episode: int, steps: int, total_reward: float, success: bool) -> None:
    """Display episode statistics"""
    # Convert numpy array to scalar if needed
    if hasattr(total_reward, 'item'):
        total_reward = total_reward.item()
    
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\n📊 Episode {episode} Complete:")
    print(f"   {status}")
    print(f"   Steps: {steps}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Average Reward/Step: {total_reward/max(1, steps):.3f}")


def check_success_condition(env: QuadCameraRobotEnv) -> bool:
    """Check if the AI successfully completed the pick-and-place task"""
    try:
        # Get box position
        box_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
        if box_joint_id != -1:
            box_qpos_addr = env.model.jnt_qposadr[box_joint_id]
            box_pos = env.data.qpos[box_qpos_addr:box_qpos_addr + 3]
            
            # Check if box is in drop zone (approximately)
            drop_zone_pos = np.array([0.15, 0.05, 0.05])
            distance_to_drop = np.linalg.norm(box_pos - drop_zone_pos)
            return distance_to_drop < 0.08  # Within 8cm of drop zone
    except:
        pass
    return False


def run_headless_testing(
    env,
    model,
    base_env,
    args
) -> None:
    """Run AI testing without the MuJoCo viewer (headless mode)"""
    
    print("🤖 Running AI testing in HEADLESS mode...")
    print("=" * 50)
    
    episode = 0
    successful_episodes = 0
    total_episodes = args.max_episodes or 5  # Default to 5 episodes if not specified
    total_all_rewards = 0.0  # Accumulate rewards across all episodes
    
    max_episodes = args.max_episodes
    episode_length = args.episode_length
    show_thinking = args.show_ai_thinking
    fast_mode = args.fast_mode
    opencv_viewer = getattr(args, 'opencv_viewer', False)
    
    for episode in range(1, total_episodes + 1):
        if max_episodes and episode > max_episodes:
            break
            
        print(f"\n🎯 AI Episode {episode} Starting...")

        # Reset environment (VecEnv API returns just obs)
        obs = env.reset()
        total_reward = 0.0
        step = 0
        action = np.zeros(7)  # Initialize action array

        for step in range(episode_length):
            # Check if model expects vision or not
            model_expects_vision = 'robot_eye' in model.observation_space.spaces

            # Debug: Check observation structure
            if step == 0:
                print(f"🔍 Observation type: {type(obs)}")
                print(f"🔍 Observation shape: {obs.shape if hasattr(obs, 'shape') else 'no shape'}")
                if isinstance(obs, dict):
                    print(f"🔍 Dict keys: {list(obs.keys())}")
                    if 'robot_eye' in obs:
                        print(f"🔍 robot_eye shape: {obs['robot_eye'].shape}")
                elif hasattr(obs, '__len__') and len(obs) > 0:
                    print(f"🔍 First element type: {type(obs[0])}")

            if not model_expects_vision:
                if isinstance(obs, dict) and 'robot_eye' in obs:
                    # Single env observation (DummyVecEnv returns dict directly)
                    obs = {'qpos': obs['qpos']}
                    print(f"   🔧 Filtered observation to match model (removed robot_eye)")
                # Note: No need for VecEnv array handling since DummyVecEnv returns dict directly

            # AI makes decision based on observation
            action, _states = model.predict(obs, deterministic=True)            # Display AI thinking process occasionally
            if show_thinking and step % 50 == 0:  # Every 50 steps
                display_ai_thinking(action, obs, step, base_env)

            # Execute AI action (VecEnv API returns obs, reward, done, info)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Update OpenCV viewer if enabled
            if opencv_viewer and step % 5 == 0:  # Update every 5 steps for smooth display
                viewer_continue = create_opencv_viewer(base_env, episode, step, action, total_reward)
                if not viewer_continue:
                    print("\n⏹️ OpenCV viewer closed by user")
                    return

            # Debug reward breakdown every 500 steps
            if step % 500 == 0 and hasattr(base_env, '_get_reward'):
                try:
                    current_reward = base_env._get_reward()
                    gripper_pos = base_env._get_gripper_position()
                    box_pos = base_env._get_box_position()
                    dist = np.linalg.norm(gripper_pos - box_pos)
                    grasped = base_env._is_box_grasped()
                    print(f"   Step {step}: Reward={current_reward:.1f}, Dist={dist:.3f}, Grasped={grasped}")
                except Exception as e:
                    pass

            # Check if episode is done
            if done:
                break
        
        # Check if AI succeeded
        success = check_success_condition(base_env)
        if success:
            successful_episodes += 1
            
        # Display episode results
        display_episode_stats(episode, step + 1, total_reward, success)
        total_all_rewards += total_reward  # Accumulate total rewards
        
        # If using OpenCV viewer, keep windows open after episode ends
        if opencv_viewer:
            print(f"\n🎥 Episode {episode} complete! Windows will stay open.")
            print("   Press 'q' in any window to continue to next episode or exit")
            
            # Keep windows open and update them until user presses 'q'
            try:
                import cv2
                while True:
                    # Update the viewer with final state
                    viewer_continue = create_opencv_viewer(base_env, episode, step + 1, action, total_reward)
                    if not viewer_continue:
                        break
                    # Wait a bit longer for user input
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
            except Exception as e:
                print(f"⚠️ Error keeping viewer open: {e}")
    
    # Final summary
    success_rate = (successful_episodes / episode) * 100 if episode > 0 else 0
    print(f"\n🏆 TESTING COMPLETE!")
    print(f"   Episodes run: {episode}")
    print(f"   Success rate: {success_rate:.1f}% ({successful_episodes}/{episode})")
    if episode > 0:
        avg_reward = total_all_rewards / episode
        if hasattr(avg_reward, 'item'):
            avg_reward = avg_reward.item()
        print(f"   Average reward per episode: {avg_reward:.2f}")

    # Clean up OpenCV windows
    if opencv_viewer:
        try:
            import cv2
            cv2.destroyAllWindows()
            print("✅ OpenCV windows closed")
        except:
            pass


def run_ai_testing(
    model: PPO,
    env: QuadCameraRobotEnv,
    max_episodes: int = None,
    episode_length: int = 1000,
    save_images: bool = False,
    visualize_vision: bool = False,
    show_thinking: bool = False,
    fast_mode: bool = False,
) -> None:
    """Run the AI model in real-time testing"""
    
    episode = 0
    total_episodes = 0
    successful_episodes = 0
    
    print("🤖 Starting AI REAL-TIME TESTING...")
    print("=" * 50)
    
    # Get the underlying environment (Monitor wrapper)
    try:
        # Robust DummyVecEnv/Monitor unwrapping with debug prints
        base_env = env
        print(f"[DEBUG] Initial env type: {type(base_env)}")
        if hasattr(base_env, 'envs'):
            base_env = base_env.envs[0]
            print(f"[DEBUG] After envs[0]: {type(base_env)}")
        if hasattr(base_env, 'env'):
            base_env = base_env.env
            print(f"[DEBUG] After .env: {type(base_env)}")
        # Print all attributes for inspection
        print(f"[DEBUG] Final base_env type: {type(base_env)}")
        print(f"[DEBUG] base_env dir: {dir(base_env)}")
        if not hasattr(base_env, 'model'):
            print("❌ Underlying environment does not have 'model'.")
            raise AttributeError("Underlying environment does not have 'model' attribute. Check environment creation.")
        mujoco_model = base_env.model
        mujoco_data = base_env.data
        print(f"✅ Unwrapped environment: {type(base_env).__name__}")
        
        # Debug observation spaces
        print(f"🔍 Model observation space: {model.observation_space}")
        print(f"🔍 Model action space: {model.action_space}")
        print(f"🔍 Environment observation space: {env.observation_space}")
        print(f"🔍 Base env observation space: {base_env.observation_space}")
        print(f"🔍 Base env model.nq: {base_env.model.nq}")
        print(f"🔍 Base env model.njnt: {base_env.model.njnt}")
        print(f"🔍 Base env model.nqvel: {getattr(base_env.model, 'nqvel', 'N/A')}")
        
        # Check what the model expects vs what we provide
        obs_sample = env.reset()  # VecEnv returns just obs
        print(f"🔍 Sample observation shape: {obs_sample.shape if hasattr(obs_sample, 'shape') else type(obs_sample)}")
        if isinstance(obs_sample, dict):
            for key, value in obs_sample.items():
                print(f"   {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        elif hasattr(obs_sample, '__len__') and len(obs_sample) > 0:
            # VecEnv returns array of observations
            obs_dict = obs_sample[0]
            print(f"🔍 VecEnv observation (first env): {type(obs_dict)}")
            if isinstance(obs_dict, dict):
                for key, value in obs_dict.items():
                    print(f"   {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        
    except Exception as e:
        print(f"❌ Could not access underlying environment: {e}")
        return
    
    try:
        with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
            
            while viewer.is_running():
                if max_episodes and episode >= max_episodes:
                    print(f"✅ Completed {max_episodes} episodes. Exiting.")
                    break
                
                episode += 1
                print(f"\n🎯 AI Episode {episode} Starting...")
                
                # Reset environment (VecEnv.reset() returns only obs, not (obs, info))
                obs = env.reset()
                total_reward = 0.0
                step = 0
                
                for step in range(episode_length):
                    if not viewer.is_running():
                        break
                    
                    # AI makes decision based on observation
                    action, _states = model.predict(obs, deterministic=True)
                    
                    # Display AI thinking process
                    if step % 50 == 0:  # Every 50 steps
                        print(f"🔍 DEBUG: step={step}, show_thinking={show_thinking}")
                        if show_thinking:
                            print(f"🔍 DEBUG: Calling display_ai_thinking at step {step}")
                            display_ai_thinking(action, obs, step, base_env)
                    
                    # Execute AI action (VecEnv returns 4 values: obs, reward, done, info)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                    done = done[0] if isinstance(done, np.ndarray) else done
                    
                    # Update viewer
                    viewer.sync()
                    
                    # Save camera images if requested
                    if save_images and step % 100 == 0:  # Every 100 steps
                        save_ai_camera_images(base_env, episode, step)
                    
                    # Visualize AI vision if requested
                    if visualize_vision and step % 100 == 0:  # Every 100 steps
                        visualize_ai_vision(base_env, episode, step, action)
                    
                    # Adaptive simulation speed
                    if fast_mode:
                        time.sleep(0.0002)  # Ultra-fast: 5000 FPS (100x faster)
                    else:
                        time.sleep(0.02)   # Normal: 50 FPS
                    
                    # Check if episode is done
                    if done:
                        break
                
                # Check if AI succeeded
                success = check_success_condition(base_env)
                if success:
                    successful_episodes += 1
                    
                total_episodes += 1
                
                # Display episode results
                display_episode_stats(episode, step + 1, total_reward, success)
                
                # Adaptive pause between episodes
                if fast_mode:
                    time.sleep(0.01)  # Ultra-fast mode
                else:
                    time.sleep(1.0)   # Normal mode
            
            # Final statistics
            print("\n" + "=" * 50)
            print("🏆 AI TESTING COMPLETE!")
            print(f"Total Episodes: {total_episodes}")
            print(f"Successful Episodes: {successful_episodes}")
            if total_episodes > 0:
                success_rate = (successful_episodes / total_episodes) * 100
                print(f"Success Rate: {success_rate:.1f}%")
            print("=" * 50)
                
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        error_msg = str(e)
        if "mjpython" in error_msg.lower() or "launch_passive" in error_msg.lower():
            # Re-raise mjpython errors so they can be caught by the main function
            # and fall back to headless mode
            raise e
        else:
            print(f"\n❌ Testing error: {e}")


def run_simple_robot_demo(env, args):
    """Run a simple robot demonstration when no models can be loaded"""
    
    class SimplePolicy:
        def __init__(self):
            self.step_count = 0
            
        def predict(self, observation, deterministic=True):
            self.step_count += 1
            t = self.step_count * 0.02
            
            # Create smooth, coordinated movements
            action = np.array([
                0.3 * np.sin(t * 0.5),        # Joint 1
                0.2 * np.cos(t * 0.7) - 0.1,  # Joint 2  
                0.15 * np.sin(t * 0.9),       # Joint 3
                0.1 * np.cos(t * 1.1),        # Joint 4
                0.08 * np.sin(t * 1.3),       # Joint 5
                0.05 * np.cos(t * 0.6),       # Joint 6
                0.3 * np.sin(t * 0.3)         # Joint 7: Gripper
            ], dtype=np.float32)
            
            return action, None
    
    print("🤖 FALLBACK: Simple Robot Demonstration")
    print("=" * 40)
    print("Since no trained models could be loaded, running a simple demonstration")
    print("that shows the robot moving smoothly with all cameras working.")
    
    policy = SimplePolicy()
    
    # Use the existing run_ai_testing function but with our simple policy
    run_ai_testing(
        model=policy,  # Our simple policy acts like a model
        env=env,
        max_episodes=args.max_episodes or 2,  # Default to 2 episodes
        episode_length=args.episode_length,
        save_images=args.save_camera_images,
        visualize_vision=args.visualize_ai_vision,
        show_thinking=args.show_ai_thinking,
        fast_mode=args.fast_mode,
    )


def test_all_models(args):
    """Test and compare all models in the models folder"""
    print("🧪 TESTING ALL MODELS IN MODELS FOLDER")
    print("=" * 60)

    # Apply gmpy2 patch for compatibility
    try:
        import gmpy2
        if not hasattr(gmpy2, 'version'):
            gmpy2.version = '2.1.2'
            print("✅ Applied gmpy2 compatibility patch")
    except ImportError:
        pass

    models_dir = Path("../models")
    if not models_dir.exists():
        print("❌ Models folder not found!")
        return

    all_models = list(models_dir.glob("*.zip"))
    if not all_models:
        print("❌ No models found in models folder!")
        return

    print(f"🔍 Found {len(all_models)} models to test:")
    for model in all_models:
        print(f"   - {model.name}")

    results = []

    # Try different environment configurations (single robot only)
    env_configs = [
        {"num_robots": 1, "name": "Single Robot"},
    ]

    for model_path in all_models:
        print(f"\n🤖 Testing: {model_path.name}")
        print("-" * 40)

        model_loaded = False
        best_result = None

        # Try different environment configurations
        for config in env_configs:
            if model_loaded:
                break

            print(f"   🔄 Trying {config['name']} configuration...")

            try:
                # Create environment for this model
                from stable_baselines3.common.env_util import make_vec_env
                from stable_baselines3.common.vec_env import VecTransposeImage

                env = make_vec_env(
                    lambda: QuadCameraRobotEnv(
                        use_vision=True,
                        camera_width=128,
                        camera_height=128,
                        num_robots=config["num_robots"]
                    ),
                    n_envs=1
                )
                env = VecTransposeImage(env)

                # Try to load model without environment first to inspect it
                try:
                    temp_model = PPO.load(str(model_path), device="cpu")
                    print(f"   📊 Model expects action space: {temp_model.action_space}")
                    print(f"   📊 Model expects obs space: {temp_model.observation_space}")
                except:
                    pass

                # Try to load model with environment
                model = PPO.load(
                    str(model_path),
                    env=env,
                    custom_objects={
                        "observation_space": env.observation_space,
                        "action_space": env.action_space,
                        "_last_obs": None,
                        "_last_episode_starts": None,
                    }
                )

                print(f"✅ Model loaded successfully with {config['name']} config!")
                model_loaded = True

                # Run test episode
                test_episodes = 2  # Reduced for faster testing
                total_rewards = []

                for episode in range(test_episodes):
                    obs = env.reset()
                    episode_reward = 0
                    steps = 0
                    max_steps = 500  # Shorter test episodes

                    for step in range(max_steps):
                        action, _states = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                        episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                        steps += 1

                        if done:
                            break

                    total_rewards.append(episode_reward)
                    print(f"   Episode {episode + 1}: {steps} steps, reward: {episode_reward:.3f}")

                avg_reward = sum(total_rewards) / len(total_rewards)
                max_reward = max(total_rewards)

                result = {
                    'model': model_path.name,
                    'status': 'SUCCESS',
                    'config': config['name'],
                    'avg_reward': avg_reward,
                    'max_reward': max_reward,
                    'episodes': test_episodes,
                    'action_space': str(model.action_space),
                    'obs_space': str(model.observation_space)
                }

                if best_result is None or avg_reward > best_result['avg_reward']:
                    best_result = result

                print(f"✅ {config['name']}: Avg reward: {avg_reward:.3f}, Max: {max_reward:.3f}")

                env.close()

            except Exception as e:
                print(f"   ❌ {config['name']} failed: {str(e)[:100]}...")
                continue

        if best_result:
            results.append(best_result)
        else:
            results.append({
                'model': model_path.name,
                'status': 'FAILED',
                'error': 'All configurations failed'
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    successful_models = [r for r in results if r['status'] == 'SUCCESS']
    failed_models = [r for r in results if r['status'] == 'FAILED']
    
    print(f"✅ Successful: {len(successful_models)}/{len(results)} models")
    print(f"❌ Failed: {len(failed_models)}/{len(results)} models")
    
    if successful_models:
        print("\n🏆 TOP PERFORMING MODELS:")
        successful_models.sort(key=lambda x: x['avg_reward'], reverse=True)
        for i, result in enumerate(successful_models[:5]):
            print(f"   {i+1}. {result['model']}")
            print(f"      Avg: {result['avg_reward']:.3f}, Max: {result['max_reward']:.3f}")
    
    if failed_models:
        print("\n❌ FAILED MODELS:")
        for result in failed_models:
            print(f"   - {result['model']}: {result['error']}")
    
    return results


def create_compatible_environment():
    """Create environment for single robot testing (Dict observation space)"""
    try:
        # Single robot environment with Dict observations (MultiInputPolicy)
        # Use DummyVecEnv to match training setup
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = QuadCameraRobotEnv(
            use_vision=True,  # Enable vision to match trained model
            camera_width=128,
            camera_height=128,
            num_robots=1  # Single robot only
        )
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])  # Wrap in VecEnv to match training

        print("✅ Created single robot environment with Dict observation space (VecEnv)")
        return env, "single_robot_dict_vec"

    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return None, "failed"
def load_model_with_fallbacks(model_path, env, env_type):
    """Try multiple strategies to load the model with the compatible environment"""
    
    # Apply compatibility patches
    try:
        import gmpy2
        if not hasattr(gmpy2, 'version'):
            gmpy2.version = '2.1.2'
    except ImportError:
        pass
    
    # Strategy 1: Load with explicit custom objects to handle environment verification
    try:
        print(f"   🔄 Strategy 1: Loading with explicit custom objects...")
        
        custom_objects = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "lr_schedule": lambda x: 3e-4,  # Provide default learning rate
            "clip_range": lambda x: 0.2,   # Provide default clip range
            "_last_obs": None,
            "_last_episode_starts": None,
        }
        
        # Try loading with exact_match=False to allow mismatched keys
        model = PPO.load(str(model_path), env=env, custom_objects=custom_objects, device="auto")
        print(f"   ✅ Strategy 1 successful!")
        return model
        
    except Exception as e:
        print(f"   ❌ Strategy 1 failed: {str(e)[:100]}...")
    
    # Strategy 2: Load with environment but prevent VecEnv wrapping
    try:
        print(f"   🔄 Strategy 2: Loading with environment (no VecEnv)...")
        
        # Temporarily modify the env to prevent PPO.load from wrapping it
        original_class = env.__class__
        
        # Create a custom class that prevents wrapping
        class NoWrapEnv(original_class):
            def __init__(self, wrapped_env):
                self.__dict__.update(wrapped_env.__dict__)
            
            @property
            def observation_space(self):
                return self.__dict__['observation_space']
            
            @property  
            def action_space(self):
                return self.__dict__['action_space']
        
        no_wrap_env = NoWrapEnv(env)
        model = PPO.load(str(model_path), env=no_wrap_env, device="auto")
        print(f"   ✅ Strategy 2 successful!")
        return model
        
    except Exception as e:
        print(f"   ❌ Strategy 2 failed: {str(e)[:100]}...")
    
    # Strategy 3: Load without environment and manually set observation space
    try:
        print(f"   🔄 Strategy 3: Loading without environment...")
        
        # Load model without environment
        model = PPO.load(str(model_path), device="auto")
        
        # Manually set the observation space to match current environment
        model.observation_space = env.observation_space
        model.action_space = env.action_space
        
        # Also update the policy's observation space
        if hasattr(model, 'policy') and hasattr(model.policy, 'observation_space'):
            model.policy.observation_space = env.observation_space
        
        print(f"   ✅ Strategy 3 successful!")
        return model
        
    except Exception as e:
        print(f"   ❌ Strategy 3 failed: {str(e)[:100]}...")
        return None


def validate_model_parameters(model) -> bool:
    """Check if model parameters contain NaN or Inf values."""
    import torch
    try:
        # For PPO models, check policy parameters
        if hasattr(model, 'policy'):
            params = list(model.policy.parameters())
        else:
            params = list(model.parameters())
        
        for param in params:
            if param is None:
                continue
            if not torch.isfinite(param).all():
                return False
        return True
    except Exception as e:
        print(f"   ⚠️ Could not validate parameters: {e}")
        return False


def main() -> None:
    args = parse_args()

    # If user wants the native MuJoCo GUI (not OpenCV/headless) try to
    # re-exec the script under `mjpython` on macOS so `mujoco.viewer.launch_passive`
    # can open a native window. Guard with MJPYTHON_REEXEC to avoid loops.
    try:
        wants_native_viewer = not args.headless and not args.opencv_viewer
        already_reexec = os.environ.get('MJPYTHON_REEXEC') == '1'
        exe_name = os.path.basename(sys.executable).lower()
        if wants_native_viewer and not already_reexec and 'mjpython' not in exe_name:
            mjp = shutil.which('mjpython')
            if mjp:
                print("🔁 Re-launching script under 'mjpython' to enable MuJoCo native viewer...")
                os.environ['MJPYTHON_REEXEC'] = '1'
                os.execv(mjp, [mjp] + sys.argv)
            else:
                # No mjpython found: ensure a GUI-capable GL backend is selected
                os.environ.setdefault('MUJOCO_GL', 'glfw')
                print("ℹ️ 'mjpython' not found in PATH. Setting MUJOCO_GL=glfw and continuing.\n   If the MuJoCo viewer still fails, run this script with 'mjpython' from a GUI terminal.")
    except Exception as _e:
        # Non-fatal; continue and let the existing fallback logic handle viewer
        print(f"⚠️ Could not auto-launch mjpython: {_e}")
    
    # Check for test-all-models mode
    if args.test_all_models:
        test_all_models(args)
        return
    
    print("🧠 LOADING TRAINED AI MODEL...")
    print("=" * 50)
    
    # Apply global compatibility patches
    try:
        import gmpy2
        if not hasattr(gmpy2, 'version'):
            gmpy2.version = '2.1.2'
            print("✅ Applied gmpy2 compatibility patch")
    except ImportError:
        pass
    
    # Discover all model files in models folder and prioritize newest models
    models_dir = Path("./models")
    if not models_dir.exists():
        print(f"❌ Models folder not found at: {models_dir.resolve()}")
        if not args.model_path.exists():
            print(f"❌ Model file not found: {args.model_path}")
            print("💡 Train a model first using: python train.py")
            return
        model_paths_to_try = [args.model_path]
    else:
        all_models = list(models_dir.glob("*.zip"))
        if not all_models:
            print(f"❌ No models found in {models_dir.resolve()}")
            if not args.model_path.exists():
                print(f"❌ Model file not found: {args.model_path}")
                print("💡 Train a model first using: python train.py")
                return
            model_paths_to_try = [args.model_path]
        else:
            # Sort models by modification time (newest first)
            all_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            print(f"🔍 Found {len(all_models)} models in {models_dir.resolve()}")
            print(f"🎯 Using newest model: {all_models[0].name}")
            # If a model is specified, use it first, otherwise use the newest
            if args.model_path and args.model_path.exists() and args.model_path.name != "ml-train.zip":
                model_paths_to_try = [args.model_path] + [m for m in all_models if m != args.model_path]
            else:
                model_paths_to_try = all_models
    
    # Try to create compatible environment
    env, env_type = create_compatible_environment()
    if env is None:
        print("❌ Could not create compatible environment!")
        return
    
    print(f"✅ Created {env_type} environment")
    
    model = None
    successful_path = None
    
    for model_path in model_paths_to_try:
        if not model_path.exists():
            continue
            
        print(f"\n🤖 Trying to load: {model_path.name}")
        
        model = load_model_with_fallbacks(model_path, env, env_type)
        if model is not None:
            successful_path = model_path
            print(f"✅ AI model loaded successfully from: {model_path.name}")
            
            # Validate model parameters for NaNs/Infs
            if not validate_model_parameters(model):
                print(f"❌ Model {model_path.name} contains NaN or Inf values in parameters!")
                print("   This model was corrupted during training. Skipping...")
                model = None
                continue
            
            break
        else:
            print(f"❌ All loading strategies failed for: {model_path.name}")
    
    if model is None:
        print("\n❌ Could not load any compatible model!")
        print("💡 Available options:")
        print("   1. Train a new model: mjpython train_ml.py")
        print("   2. Use simple demonstration: mjpython simple_demo.py")
        print("   3. Check if newer model files exist in models/ folder")
        
        # Offer to run simple demonstration instead
        print("\n🤖 Would you like to run a simple robot demonstration instead?")
        print("   This will show the robot working without needing trained models.")
        
        # Auto-run simple demo for now
        print("🎬 Starting simple robot demonstration...")
        run_simple_robot_demo(env, args)
        return
    
    print("✅ Testing environment created!")
    print("📹 Single robot with wrist camera enabled")
    
    print("\n🎮 REAL-TIME AI CONTROL:")
    print("• AI will use camera vision to make decisions")
    print("• Single robot arm controlled by neural network")
    print("• OpenCV control panel shows all camera feeds and status")
    print("• Use --headless for text-only output")
    print("• Use --opencv-viewer if MuJoCo GUI doesn't work")
    
    if args.headless:
        print("• Running in headless mode (no GUI)")
    elif args.opencv_viewer:
        print("• Using OpenCV control panel for GUI display")
    else:
        print("• Launching MuJoCo GUI viewer")
    
    if args.max_episodes:
        print(f"• Will run {args.max_episodes} episodes maximum")
    else:
        print("• Will run until viewer is closed")
    
    if args.save_camera_images:
        print("• Camera images will be saved during execution")
    
    if args.visualize_ai_vision:
        print("• AI vision will be visualized (combined camera views)")
    
    if args.show_ai_thinking:
        print("• AI decision process will be displayed")
    
    # Get the underlying environment for both viewer and headless modes
    try:
        # Robust DummyVecEnv/Monitor unwrapping with debug prints
        base_env = env
        print(f"[DEBUG] Initial env type: {type(base_env)}")
        if hasattr(base_env, 'envs'):
            base_env = base_env.envs[0]
            print(f"[DEBUG] After envs[0]: {type(base_env)}")
        if hasattr(base_env, 'env'):
            base_env = base_env.env
            print(f"[DEBUG] After .env: {type(base_env)}")
        # Print all attributes for inspection
        print(f"[DEBUG] Final base_env type: {type(base_env)}")
        print(f"[DEBUG] base_env dir: {dir(base_env)}")
        if not hasattr(base_env, 'model'):
            print("❌ Underlying environment does not have 'model'.")
            raise AttributeError("Underlying environment does not have 'model' attribute. Check environment creation.")
        mujoco_model = base_env.model
        mujoco_data = base_env.data
        print(f"✅ Unwrapped environment: {type(base_env).__name__}")
    except Exception as e:
        print(f"❌ Could not access underlying environment: {e}")
        base_env = None
        mujoco_model = None
        mujoco_data = None
    
    # Check which mode to run
    if args.headless:
        print("🎯 Running AI testing in HEADLESS mode")
        print("🤖 AI will run episodes and report results")
        
        # Run headless testing
        run_headless_testing(env, model, base_env, args)
        return
    elif args.test_cameras:
        print("🎯 Testing camera views only")
        print("📹 Camera windows will open for 10 seconds")
        print("   Press 'q' in any window to close early")
        
        # Test cameras only
        test_camera_views(base_env)
        return
    elif args.opencv_viewer:
        print("🎯 Running AI testing with OpenCV GUI")
        print("🎥 OpenCV control panel will show all camera feeds and status")
        print("   Press 'q' in any window to quit")
        
        # Run with OpenCV viewer
        run_headless_testing(env, model, base_env, args)
        return
    else:
        print("🎯 Running AI testing with MuJoCo GUI viewer")
        print("🖥️  MuJoCo simulation window will open")
    
    try:
        # Run AI testing with viewer
        run_ai_testing(
            model=model,
            env=env,
            max_episodes=args.max_episodes,
            episode_length=args.episode_length,
            save_images=args.save_camera_images,
            visualize_vision=args.visualize_ai_vision,
            show_thinking=args.show_ai_thinking,
            fast_mode=args.fast_mode,
        )
    except Exception as e:
        error_msg = str(e)
        if "mjpython" in error_msg.lower() or "launch_passive" in error_msg.lower():
            import platform
            is_macos = platform.system() == "Darwin"

            print(f"\n❌ MuJoCo GUI viewer failed to launch")
            print(f"   Error: {error_msg}")

            if is_macos:
                print(f"\n💡 macOS Solutions:")
                print(f"   1. Make sure you're using 'mjpython' (you are!)")
                print(f"   2. Run from a graphical terminal app (Terminal.app, iTerm2, etc.)")
                print(f"   3. Ensure XQuartz is installed if needed: brew install xquartz")
                print(f"   4. Try running from VS Code integrated terminal")
                print(f"   5. Alternative: python3 test_model.py --opencv-viewer")
                print(f"   6. Alternative: python3 test_model.py --headless")
            else:
                print(f"\n💡 Solutions:")
                print(f"   1. Run in a graphical environment (not terminal)")
                print(f"   2. Use: python3 test_model.py --opencv-viewer")
                print(f"   3. Use: python3 test_model.py --headless")

            print(f"\n🔄 Try: python3 test_model.py --opencv-viewer")
            sys.exit(1)
        else:
            print(f"\n❌ Testing error: {e}")
            raise
    finally:
        env.close()


if __name__ == "__main__":
    main()