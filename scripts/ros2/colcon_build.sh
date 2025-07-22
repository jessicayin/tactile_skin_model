#!/bin/bash -e
if [ "x$ROS_DISTRO" = "x" ]; then
    echo "please first initialize ROS env (e.g., conda activate _ros2_env)"
    exit
fi

if git rev-parse --git-dir > /dev/null 2>&1; then
  REPOROOT=`git rev-parse --show-toplevel`
else
  REPOROOT=$HOME/gum_ws/src/GUM
fi

if [ `basename $PWD` = "gum_ws" ]; then
    colcon build --base-paths $REPOROOT/scripts/ros2/ $REPOROOT/gum/devices/metahand/ros/  \
	--packages-select gum_ros2 meta_hand_description  \
    --cmake-args "-DPython3_EXECUTABLE=`which python3`" --cmake-clean-cache \
	--symlink-install   --event-handlers console_direct+
else
    echo please invoke from gum_ws
fi
echo "please"
echo 
echo "source ./install/setup.$(basename $SHELL)"
