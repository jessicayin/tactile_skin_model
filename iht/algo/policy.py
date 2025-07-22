import abc
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type


from omegaconf import DictConfig, OmegaConf

import iht.utils.tensor as tensor_utils
from iht.algo.models.models import ActorCritic
from iht.utils.misc import (
    import_fn,
    maybe_fix_eval_config,
    set_np_formatting,
)
import torch
if TYPE_CHECKING:
    from iht.tasks import AllegroHandHora


class Policy(object):
    def __init__(self, env: "AllegroHandHora", output_dir, full_config, silent=False):
        self.model: ActorCritic
        self._init_impl(env, output_dir, full_config, silent=silent)

    @abc.abstractmethod
    def _init_impl(self, env: "AllegroHandHora", output_dir, full_config, silent=False):
        pass

    @abc.abstractmethod
    def _build_input_dict(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    def is_log_node(self) -> bool:
        return not self.multi_gpu or (self.multi_gpu and self.rank == 0)

    def rollout(
        self,
        thread_stop: Optional[threading.Event] = None,
        env_started_event: Optional[threading.Event] = None,
        eval_start: Optional[threading.Event] = None,
        step_delay: Optional[float] = None,
        override_fn: Optional[Callable[[], Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """Evaluates the policy.

        Args:
            thread_stop (Optional[threading.Event], optional): if given,
                stops evaluation once it's set. Defaults to None.
            env_started_event (Optional[threading.Event], optional): if given, this event
                is set once the environment has been reset and CPU second has passed.
                Defaults to None.
            eval_start (Optional[threading.Event], optional): if given, policy evaluation
                is blocked until this event is set (but the environment is still reset).
                Defaults to None, in which policy evaluation is not blocked and
                immediately proceeds.
            override_fn (Optional[Callable[[], Dict[str, torch.Tensor]]], optional):
                if given, it will be called before inferece to override keys in the model
                input with the dict that this function outputs. Defaults to None.
            step_delay: (Optional[float]): if given, sleeps for this amount of seconds
                after every step.
        """
        self.set_eval()
        obs_dict = self.env.reset()
        if env_started_event is not None:
            time.sleep(1.0)
            env_started_event.set()
        if eval_start is not None:
            print("Waiting for signal that evaluation can start")
            eval_start.wait()
        print("Start policy evaluation loop.")
        while True:
            if thread_stop is not None:
                if self.env.pause:
                    time.sleep(0.001)
                    continue
                if thread_stop.is_set():
                    return
            input_dict = self._build_input_dict(obs_dict)
            if override_fn:
                input_dict.update(
                    tensor_utils.all_to(
                        override_fn(), device=self.device, dtype=torch.float32
                    )
                )
            mu, extrin = self.model.act_inference(input_dict)
            assert extrin is not None
            mu = torch.clamp(mu, -1.0, 1.0)
            self.env.step(mu, extrin_record=extrin)
            if step_delay is not None:
                time.sleep(step_delay)


def eval_policy(config: DictConfig):
    from iht.tasks import env_from_config

    eval_path = maybe_fix_eval_config(config, create_config_subdir=True)
    eval_path.mkdir(exist_ok=True, parents=True)
    with open(eval_path / "config.yaml", "w") as f:
        OmegaConf.save(config=config, f=f)
    set_np_formatting()
    config.task.eval_results_dir = str(eval_path / f"{config.task.env.object.subtype}")
    env: AllegroHandHora = env_from_config(config)
    # Loading agent agent
    agent_cls: Type[Policy] = import_fn(  # type: ignore
        f"iht.algo.{config.train.algo}"
    )
    agent = agent_cls(env, "null", full_config=config, silent=True)
    assert config.train.load_path
    agent.restore_test(config.train.load_path)
    agent.rollout()
