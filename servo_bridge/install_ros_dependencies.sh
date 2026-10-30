#!/usr/bin/env bash
# Explicit installer, never called by runtime or on import.
set -euo pipefail
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 22.04 ]]; then
  echo "This integration is tested on Ubuntu 22.04 / ROS 2 Humble only." >&2
  exit 1
fi
sudo apt-get install --no-install-recommends -y \
  ros-humble-moveit-servo ros-humble-moveit-kinematics \
  ros-humble-robot-state-publisher ros-humble-urdfdom-py \
  python3-numpy python3-scipy python3-yaml python3-pykdl python3-pygame
