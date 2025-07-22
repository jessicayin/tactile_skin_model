# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import datetime
import os
from typing import Optional, Type

import isaacgym
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

import iht.utils.hydra_resolvers
from iht.tasks import env_from_config
from iht.utils.misc import (
    git_diff_config,
    git_hash,
    import_fn,
    resolve_checkpoint,
    set_np_formatting,
    set_seed,
)

# from iht.algo.policy import Policy #commenting out gets rid of this error:
#ImportError: cannot import name 'Policy' from partially initialized module 'iht.algo' (most likely due to a circular import) (/home/robotdev/private-tactile-iht/iht/algo/init.py)


# If config_path is given, this overrides the config arg
def run(config: DictConfig, config_path: Optional[str] = None, experiment_name: Optional[str] = None):
    if config_path is not None:
        assert config is None
        config = OmegaConf.load(config_path)
    if experiment_name is not None:
        config.experiment_name = experiment_name
    if config.checkpoint:
        config.checkpoint = resolve_checkpoint(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    if config.train.ppo.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        config.sim_device = f"cuda:{rank}"
        config.rl_device = f"cuda:{rank}"
        config.graphics_device_id = int(rank)
        # sets seed. if seed is -1 will pick a random one
        config.seed = set_seed(config.seed + rank)
    else:
        rank = 0
        config.seed = set_seed(config.seed)

    cprint("Start Building the Environment", "green", attrs=["bold"])
    env = env_from_config(config)

    output_dif = os.path.join(config.outputs_root_dir, config.train.ppo.output_name)
    if rank == 0:
        os.makedirs(output_dif, exist_ok=True)

    agent_cls: Type[Policy] = import_fn(  # type: ignore
        f"iht.algo.{config.train.algo}"
    )
    agent = agent_cls(env, output_dif, full_config=config)
    if config.test:
        assert config.train.load_path
        agent.restore_test(config.train.load_path)
        agent.rollout(None)
    else:
        if rank <= 0:
            date = str(datetime.datetime.now().strftime("%m%d%H"))
            print(git_diff_config("./"))
            if config.train.algo == "ProprioAdapt":
                gitdiff_suffix = f"_{config.train.padapt.cache_suffix}"
            else:
                gitdiff_suffix = ""
            os.system(f"git diff HEAD > {output_dif}/gitdiff{gitdiff_suffix}.patch")
            with open(
                os.path.join(output_dif, f"config_{date}_{git_hash()}.yaml"), "w"
            ) as f:
                f.write(OmegaConf.to_yaml(config))
        agent.restore_train(
            config.train.load_path,
            only_teacher_weights=config.train.padapt.get(
                "only_load_teacher_weights", True
            ),
        )
        agent.train()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", type=argparse.FileType("r"), help="Path to hydra config."
    )
    args = parser.parse_args()
    run(None, args.config_path)
