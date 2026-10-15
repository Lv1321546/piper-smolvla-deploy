# Gamepad_PiPER attribution

Upstream: https://github.com/kehuanjack/Gamepad_PiPER
Pinned commit: aa00c80e100d687a5f4a386b79b83ece06b3a30f
License: MIT; see LICENSE (Copyright 2025 科幻木影).

`piper.urdf` is copied unchanged from upstream `piper/piper.urdf`.
The Linux joystick axis/button mapping in `e14/gamepad.py` is adapted from
upstream `src/gamepad_base.py` (`_setup_platform_mappings`).

This is a selective integration, not an installation of upstream main.py.
Upstream visualization, automatic homing, mode toggles, and dependency pins are
not imported. A local NumPy URDF FK/Jacobian/DLS solver replaces Trac-IK so the
existing Python 3.12 / LeRobot environment needs only pygame as an extra package.
The original reference checkout is outside this deployment repository.
No mesh files are needed for kinematics; URDF visual meshes are not bundled.

Upstream's controller is adapted to the deployment's CAN feedback checks, manual
gripper API, measured next-state dataset convention, and explicit execution gate.
