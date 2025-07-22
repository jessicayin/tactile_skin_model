# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------
import hashlib
import importlib
import os
import pathlib
import random
import shlex
import subprocess
import tempfile
from typing import Callable, List, Sequence

import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def wprint(msg: str):
    """Prints a warning message"""
    cprint(msg, "red", attrs=["bold"])


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def git_diff_config(name):
    cmd = f"git diff --unified=0 {name}"
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def set_np_formatting():
    """formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def set_seed(seed):
    import torch  # to avoid isaacgym errors

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed


def get_rank():
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size():
    # return number of gpus
    if "LOCAL_WORLD_SIZE" in os.environ.keys():
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        world_size = 1
    return world_size


def resolve_checkpoint(checkpoint_pattern: str) -> str:
    if "*" in checkpoint_pattern:
        from glob import glob

        _ckpt = glob(checkpoint_pattern)
        if len(_ckpt) != 1:
            raise ValueError(
                f"Expected exactly 1 checkpoint at {checkpoint_pattern}, "
                f"but {len(_ckpt)} were found."
            )
        checkpoint_pattern = _ckpt[0]
    return to_absolute_path(checkpoint_pattern)


def import_fn(func_name: str) -> Callable:
    args_ = func_name.split(".")
    module = importlib.import_module(".".join(args_[:-1]))
    return getattr(module, args_[-1])


def flatten_cfg(cfg: DictConfig) -> List[str]:
    result = []
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            result.extend(flatten_cfg(v))
        else:
            result.append((k, v))
    return result


def multi_gpu_aggregate_stats(values):
    # lazy imports to avoid isaacgym errors
    import torch
    import torch.distributed as dist

    if type(values) is not list:
        single_item = True
        values = [values]
    else:
        single_item = False
    rst = []
    for v in values:
        if type(v) is list:
            v = torch.stack(v)
        if get_world_size() > 1:
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v = v / get_world_size()
        if v.numel() == 1:
            v = v.item()
        rst.append(v)
    if single_item:
        rst = rst[0]
    return rst


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        import torch  # to avoid isaacgym errors

        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean


def hash_config(config: DictConfig) -> str:
    md5 = hashlib.md5()
    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=config, f=fp.name)
        with open(fp.name, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                md5.update(data)

    return md5.hexdigest()


class _hide_attribs:
    def __init__(self, config: DictConfig, attribs: Sequence[str]) -> None:
        self.attribs = attribs
        self.config = config
        self.old_values = []

    def __enter__(self) -> None:
        for attrib in self.attribs:
            old_value = OmegaConf.select(self.config, attrib)
            if old_value is not None:
                OmegaConf.update(self.config, attrib, None)
            self.old_values.append(old_value)

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        for attrib, val in zip(self.attribs, self.old_values):
            if val is not None:
                OmegaConf.update(self.config, attrib, val)


def maybe_fix_eval_config(
    config: DictConfig,
    requires_stream: bool = False,
    create_config_subdir: bool = False,
) -> pathlib.Path:
    if not config.checkpoint:
        raise ValueError("A policy checkpoint must be provided.")
    if config.task.env.object.subtype is None:
        raise ValueError("Need to specify an object subtype.")
    if config.train.ppo.multi_gpu:
        wprint("Multi-GPU not supported, ignoring flag.")
        config.train.ppo.multi_gpu = False
    if not config.test:
        wprint("Setting test=True, ignoring flag.")
        config.test = True
    if not config.task.on_evaluation:
        wprint("Setting task.on_evaluation=True, ignoring flag.")
        config.task.on_evaluation = True
    if requires_stream:
        if not config.task.env.enableCameraSensors:
            wprint("Enabling camera sensors, ignoring flag.")
            config.task.env.enableCameraSensors = True
        if config.task.env.numEnvs > 1:
            wprint("Only 1 environment supported, overriding flag.")
            config.task.env.numEnvs = 1
        config.task.env.camera_style = "hora-tactile"
    if config.visualize:
        config.headless = False
        config.task.env.enableDebugVis = True
    if not config.randomize_eval:
        config.task.env.randomization.randomizeMass = False
        config.task.env.randomization.randomizeCOM = False
        config.task.env.randomization.randomizeFriction = False
        config.task.env.randomization.randomizePDGains = False
        config.task.env.randomization.randomizeScaleList = [0.62]
        config.task.env.forceScale = 0
    eval_path = pathlib.Path(config.task.eval_results_dir)
    if create_config_subdir:
        # Temporarily remove attribs that shouldn't be part of the hash
        # (subtype is removed because eval script creates a separate subdir
        #  for the subtype)
        with _hide_attribs(
            config,
            [
                "task.env.object.subtype",
                "eval_all_objects",
                "visualize",
                "headless",
                "task.eval_results_dir",
                "task.env.enableDebugVis",
                "task.env.camera_look_at_env",
                "task.log_metrics",
            ],
        ):
            eval_path /= f"config.{hash_config(config)}"
        config.task.eval_results_dir = str(eval_path)
    config.checkpoint = resolve_checkpoint(config.checkpoint)
    config.seed = set_seed(config.seed)
    return eval_path
