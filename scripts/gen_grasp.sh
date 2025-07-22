#!/bin/bash

# CACHE can be some existing output folder, does not matter
# numEnvs=20000, headless=True, episodeLen=50 to save time
# pipeline need to be cpu to get the pairwise contact
# no custom PD because bug in CPU mode
# NOTE: comment out or disable setting GPU / CUDA_VISIBLE_DEVICE for non-headless mode

GPUS=$1 
SCALE=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=AllegroHandGraspIHT headless=True pipeline=cpu \
task.env.numEnvs=1000 test=False \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.object.type=cylinder \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.51 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True \
${EXTRA_ARGS}
