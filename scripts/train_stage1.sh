#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py -m task=AllegroHandHoraIHT headless=True launcher=basic seed=${SEED} \
train.ppo.multi_gpu=False \
experiment/stage@stage=stage1 \
outputs_root_dir="./models" \
task.env.rotation_axis="-z" \ 
#TODO: remove rotation axis, doesn't do anything
++train.experiment_name="experiment_name" \
task.env.numEnvs=1 \
#reward function hyperparameters
task.env.reward.rotate_reward_scale=700 \
task.env.reward.position_penalty_scale=500 \
task.env.reward.obj_linvel_penalty_scale=10 \
task.env.reward.pencil_z_dist_penalty_scale=1000 \
task.env.reward.contact_force_reward_scale=500 \
task.use_wandb=False \
task.env.randomization.randomizeMassLower=0.1 \
task.env.randomization.randomizeMassUpper=0.35 \