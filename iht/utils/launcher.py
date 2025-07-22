import pathlib
import tempfile
from typing import Optional, Sequence

import submitit
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def launch_submitit(
    config: DictConfig, mode: str, extra_flags: Sequence[str] = None
) -> None:
    """Launches a dexit job with submitit.

    This sets up the launch command automatically depending on whether multi-gpu is
    True or False.

    Args:
        config (DictConfig): the configuration for the job.
        mode (str): whether it's a "train" or "eval" job.
    """
    if mode not in ["train", "eval"]:
        raise ValueError("mode must be either 'train' or 'eval'.")
    launcher_cfg = HydraConfig.get().launcher
    assert launcher_cfg.tasks_per_node == 1 and launcher_cfg.nodes == 1
    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=config, f=fp.name)
        if config.train.ppo.multi_gpu:
            exec_cmd = [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={launcher_cfg.gpus_per_node}",
            ]
        else:
            exec_cmd = ["python"]
        exec_cmd.extend(["gum/labs/dexit/main.py", f"{fp.name}", f"--mode={mode}"])
        exec_cmd.extend(extra_flags or [])
        function = submitit.helpers.CommandFunction(exec_cmd)
        executor = submitit.AutoExecutor(
            folder=pathlib.Path(launcher_cfg.submitit_folder).parent / "train_logs",
            cluster="local",
        )
        executor.update_parameters(
            timeout_min=launcher_cfg.timeout_min,
            cpus_per_task=launcher_cfg.cpus_per_task,
            gpus_per_node=launcher_cfg.gpus_per_node,
            mem_gb=launcher_cfg.mem_gb,
        )
        job = executor.submit(function)
        print(job.result())
