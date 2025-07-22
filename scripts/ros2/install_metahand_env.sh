#!/bin/bash -e 
shopt -s expand_aliases
if type -P micromamba; then
    echo "micromamba detect, using micromamba inplace of mamba"
    alias mamba=micromamba
fi

REPOROOT=`git rev-parse --show-toplevel`

usage="$(basename "$0") [-h] [-e ENV_NAME] --
Install the tac_neural environment
where:
    -h  show this help text
    -e  name of the environment, default=_metahand
"

options=':hr:c:e:f'
while getopts $options option; do
  case "$option" in
    h) echo "$usage"; exit;;
    e) ENV_NAME=$OPTARG;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2; echo "$usage" >&2; exit 1;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2; echo "$usage" >&2; exit 1;;
  esac
done

# if ENV_NAME is not set, then set it to _metahand
if [ -z "$ENV_NAME" ]; then
        ENV_NAME=_metahand
fi

unset PYTHONPATH LD_LIBRARY_PATH

# # to base mamba env
eval "$(conda shell.bash hook)"

# # remove any exisiting env
conda remove -y -n $ENV_NAME --all

mamba create -y --name $ENV_NAME python=3.8

conda activate $ENV_NAME

# Install ROS2
mamba install -y -c tingfan -c conda-forge -c robostack-staging ros-humble-ros-base ros-humble-foxglove-bridge ros-humble-realsense2-camera ros-humble-ros1-bridge compilers cmake pkg-config make ninja colcon-common-extensions ros-humble-allegro-hand-controllers  ros-humble-rosbag2-storage-mcap  ros-humble-compressed-image-transport ros-humble-allegro-hand-controllers ros-humble-xacro

# Install fairo-tag deps
mamba install -y -c conda-forge gtsam
pip install sophuspy scipy matplotlib

# Install policy deployment dependency
pip install termcolor tensorboardX gitpython gym
mamba install -y -c pytorch -c nvidia/label/cuda-11.8.0 pytorch=*=py3.8_cuda11.8_cudnn8.7.0_0 torchvision

# Install pip dependencies
pip install -r $REPOROOT/scripts/ros2/metahand_requirements.txt

# Remove pyudev directory that causes OSError
rm -r $CONDA_PREFIX/lib/udev

# Install gum package
pip install -e $REPOROOT
