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

python scripts/labs/dexit/train.py -m task=AllegroHandHoraIHT launcher=basic \
train.ppo.multi_gpu=False experiment/stage@stage=stage2_proprio_reskin \
checkpoint="./models/oracle_policy.pth" \ 
outputs_root_dir="./outputs/stage2/" \
++train.experiment_name="stage2_proprio_reskin" \
task.use_wandb=False \
#hyperparameters from stage 1
task.env.reward.rotate_reward_scale=700 \
task.env.reward.position_penalty_scale=500 \
task.env.reward.obj_linvel_penalty_scale=10 \
task.env.reward.pencil_z_dist_penalty_scale=1000 \
task.env.translation_direction=positive \
task.env.randomization.randomizeMass=True \
task.env.randomization.randomizeMassLower=0.1 \
task.env.randomization.randomizeMassUpper=0.35 \
task.env.reward.contact_force_reward_scale=500 \
task.env.enableCameraSensors=True \
task.env.rgbd_camera.enable=True \
task.env.privInfo.enableNetContactF=True \
task.env.object.subtype="cylinder" \
++task.collect_sim_data.save_data=False 