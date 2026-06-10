# DIY Low-Cost Robot Arm — ML Control

> Built from the [low_cost_robot](https://github.com/AlexanderKoch-Koch/low_cost_robot) open-source design by Alexander Koch. 3D-printed at UTCC, trained with Python RL (Jan 2025).

---

## About

This project is my personal build of Alexander Koch's low-cost robot arm — a ~$250 follower arm + ~$180 leader arm setup using Dynamixel servos and 3D-printed parts. I printed all components in UTCC's 3D printer lab, assembled by hand, then built a reinforcement learning pipeline on top to control it via camera vision.

Working through physical constraints — print tolerances, motor torque, servo calibration — taught more about real-world robotics than any coursework.

**Source hardware design & wiring:** [AlexanderKoch-Koch/low_cost_robot](https://github.com/AlexanderKoch-Koch/low_cost_robot)

---

## Hardware (from original design)

### Follower Arm (~$258)

| Part | Cost |
|------|------|
| 2× Dynamixel XL430-W250 | $100 |
| 4× Dynamixel XL330-M288 | $96 |
| Waveshare Serial Bus Servo Driver Board | $10 |
| Voltage Reducer | $10 |
| 12V Power Supply | $12 |
| Misc (clamp, wires, idler wheels) | ~$30 |

### Leader Arm (~$183)

| Part | Cost |
|------|------|
| 6× Dynamixel XL330-M077 | $144 |
| Waveshare Serial Bus Servo Driver Board | $10 |
| 5V Power Supply | $6 |
| Misc | ~$23 |

3D-printed STL files: `Low-cost-armRobot/hardware/`  
Assembly video (original): https://youtu.be/RckrXOEoWrk

---

## ML Stack

| Tool | Details |
|------|---------|
| Language | Python 3.11 |
| RL Framework | Stable Baselines3 (PPO) |
| Simulation | MuJoCo + Gymnasium |
| Vision | OpenCV |
| Environment | Conda (`environment.clean.yml`) |

---

## Repo Structure

```
Low-cost-armRobot/
  train.py                — RL training entry point (PPO, 100M steps)
  test_model.py           — Load and evaluate a trained model
  test-camera.py          — Camera pipeline test
  environment.clean.yml   — Conda environment spec
  Robot_Learning/
    robot_env.py          — Custom Gymnasium environment
  Simulation/
    ArmRobot/             — MuJoCo XML scene + STL assets
  design/pictures/        — Photos of physical build
  models/                 — Saved model checkpoints
```

---

## Quick Start

```bash
conda env create -f Low-cost-armRobot/environment.clean.yml
conda activate armrobot-clean

# Train RL agent
python Low-cost-armRobot/train.py

# Run simulation
python Low-cost-armRobot/Simulation/simulation.py

# Test camera pipeline
python Low-cost-armRobot/test-camera.py
```

---

## Credits

All hardware design, servo selection, wiring schematics, and STL files are from **[AlexanderKoch-Koch/low_cost_robot](https://github.com/AlexanderKoch-Koch/low_cost_robot)** — an open-source project by Alexander Koch ([@alexkoch_ai](https://x.com/alexkoch_ai)).

The RL training pipeline, MuJoCo environment, and camera integration in this repo are my own additions built on top of that foundation.
