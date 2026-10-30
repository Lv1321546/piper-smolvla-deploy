#!/usr/bin/env bash
# Run ROS in a child environment. Never change the caller's Conda or bashrc.
set -eo pipefail
SERVO_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH
export PYTHONNOUSERSITE=1
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-83}"
export ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
cd -- "$SERVO_REPO_DIR"
exec /usr/bin/python3 -m servo_bridge.runtime "$@"
