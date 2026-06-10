# DIY Arm Robot — ML Control

Low-cost 3D-printed robotic arm controlled via a Python reinforcement learning pipeline (Jan 2025).

## About

When you can't afford a ready-made robot arm, you build one.

Used the UTCC 3D printer lab to print each component from open-source part designs, assembled the arm by hand, then built an RL training pipeline to control it using camera vision. Working through physical constraints — print tolerance, motor torque, balance — taught more about real engineering than coursework alone.

## Features

- 3D-printed arm with follower/leader configuration
- PPO reinforcement learning via Stable Baselines3
- MuJoCo simulation environment for training
- OpenCV-based real-time camera pipeline
- Conda environment for reproducible setup

## Stack

| Tool | Details |
|------|---------|
| Language | Python 3.11 |
| RL Framework | Stable Baselines3 (PPO) |
| Simulation | MuJoCo + Gymnasium |
| Vision | OpenCV |
| Environment | Conda (`environment.clean.yml`) |

## Project Structure

```
Low-cost-armRobot/
  train.py              — RL training entry point (PPO, 100M steps)
  test_model.py         — Load and evaluate a trained model
  test-camera.py        — Camera pipeline test
  environment.clean.yml — Conda environment spec
  Robot_Learning/
    robot_env.py        — Custom Gymnasium environment
  Simulation/
    ArmRobot/           — MuJoCo XML scene + STL assets
  design/pictures/      — Photos of physical arm build
  models/               — Saved model checkpoints
```

## Quick Start

```bash
conda env create -f environment.clean.yml
conda activate armrobot-clean

# Run headless training (faster)
python train.py --no-render

# Quick test run
python train.py --test_run --no-render

# Evaluate a trained model
python test_model.py
```

## Training Details

- Algorithm: PPO (Proximal Policy Optimization)
- Observation: 128×128 wrist camera image
- Task: Pick-and-place
- Default: 100M steps with auto-checkpointing + TensorBoard logging
