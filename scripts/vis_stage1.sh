#!/bin/bash

python train.py task=AllegroHandHoraIHT launcher=basic headless=False seed=0 ++task.env.enableDebugVis=True task.env.numEnvs=9 test=True experiment/stage@stage=stage1 task.env.object.type=cylinder train.algo=PPO train.ppo.priv_info=True train.ppo.proprio_adapt=False checkpoint="./models/oracle_policy.pth" task.env.translation_direction="positive" 