import sys
import traceback

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

import iht.utils.hydra_resolvers
from iht.utils.launcher import launch_submitit


@hydra.main(config_name="config", config_path="configs")
def main(config: DictConfig):
    try:
        launcher_cfg = HydraConfig.get().launcher
        if "submitit" in launcher_cfg._target_:
            launch_submitit(config, "train")
        else:
            # lazy import so that submitit launch is faster
            from iht.train import run

        run(config)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
