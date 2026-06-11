# RoArm-M2-S — Setup & ROS 2 Integration

> Hardware by [Waveshare](https://www.waveshare.com/wiki/RoArm-M2-S). This repo covers full setup: USB drivers, ROS 2 workspace, URDF model, serial bridge node, and 3D attachment files (2025).

---

## About

The RoArm-M2-S is a 6-DOF desktop robotic arm from Waveshare — ESP32-based, supports JSON serial control, ROS 2, and Python SDK out of the box.

This repo documents my full setup from unboxing to motion planning in RViz: CP210x USB-UART driver, a ROS 2 package with the arm's URDF model, a serial bridge node that streams joint commands to the physical arm, and STEP/DXF files for camera and tool attachments.

The longer goal: integrate this arm into an automation workflow for a family metalwork business in Pattani, where it handles physical sorting and positioning tasks.

**Official product page:** https://www.waveshare.com/wiki/RoArm-M2-S

---

## Hardware Specs (RoArm-M2-S)

| Spec | Detail |
|------|--------|
| DOF | 6 |
| Controller | ESP32 |
| Communication | USB serial (CP210x), Wi-Fi |
| Control protocol | JSON over serial / HTTP |
| Power | 7.4V LiPo or DC adapter |
| End effector | Gripper (included) |
| 3D files | STEP + DXF provided |

---

## Repo Contents

```
roarm_ws_em0/src/
  roarm/                        — URDF model ROS 2 package (ament_cmake)
    urdf/roarm.urdf              — Full 6-DOF URDF model
    meshes/                      — STL mesh files (L1–L4, base, gripper)
    launch/roarm.launch.py       — Launches RViz with joint state publisher
    rviz/rviz_basic_settings.rviz
  serial_ctrl/                  — Serial bridge node (ament_python)
    serial_ctrl/serial_ctrl_py.py — Subscribes to joint_states → sends JSON to arm

RoArm-M2-S_STEP/               — STEP files for 3D tool/camera attachments
RoArm-M2-S_3D/                 — DXF and additional 3D design files
CP210x_Universal_Windows_Driver/ — USB-UART driver (Windows)
```

---

## Quick Start

### 1. USB Driver

Install CP210x driver before connecting the arm:
- **Windows:** run installer in `CP210x_Universal_Windows_Driver/`
- **macOS/Linux:** driver built-in (should appear as `/dev/tty.SLAB_USBtoUART`)

### 2. Build ROS 2 Workspace

```bash
cd roarm_ws_em0
colcon build
source install/setup.bash
```

### 3. Visualize in RViz

```bash
ros2 launch roarm roarm.launch.py
# Opens RViz with URDF model + joint state sliders
```

### 4. Run Serial Bridge

```bash
ros2 run serial_ctrl serial_ctrl
# Subscribes to /joint_states, sends JSON commands to the arm over USB
```

---

## Serial Protocol

The `serial_ctrl` node sends JSON commands to the arm's ESP32 controller:

```json
{
  "T": 102,
  "base": 0.0,
  "shoulder": 1.57,
  "elbow": -1.0,
  "hand": 1.57,
  "spd": 0,
  "acc": 0
}
```

All joint values in **radians**. `hand` offset: `rad + π`.

---

## References

- **Product wiki:** https://www.waveshare.com/wiki/RoArm-M2-S
- **Waveshare GitHub (official):** https://github.com/waveshareteam/RoArm-M2
- **ROS 2 docs:** https://docs.ros.org
