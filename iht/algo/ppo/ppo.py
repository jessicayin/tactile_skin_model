# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import copy
import os
import threading
import time
from glob import glob
from typing import TYPE_CHECKING, Dict, Optional

import wandb
import omegaconf
import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter


from iht.algo.models.models import ActorCritic
from iht.algo.models.running_mean_std import RunningMeanStd
from iht.algo.policy import Policy
from iht.algo.ppo.experience import ExperienceBuffer
from iht.utils.misc import (
    AverageScalarMeter,
    get_rank,
    get_world_size,
    multi_gpu_aggregate_stats,
)

if TYPE_CHECKING:
    from iht.tasks import AllegroHandHora

class PPO(Policy):
    def _init_impl(self, env: "AllegroHandHora", output_dif, full_config, silent=False):
        # ---- MultiGPU ----
        self.multi_gpu = full_config.train.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = get_rank()
            self.rank_size = get_world_size()
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            self.device = "cuda:" + str(self.rank)
            print(f"current rank: {self.rank} and use device {self.device}")
        else:
            self.rank = -1
            self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(action_space.high.copy()).float().to(self.device)
        )
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        # ---- Use wandb ---- set to False when running locally for debugging or visualizing
        self.use_wandb = full_config.task.use_wandb
        # ---- Priv Info ----
        self.priv_info_dim = self.env.priv_info_dim
        self.priv_info = self.ppo_config["priv_info"]
        self.proprio_adapt = self.ppo_config["proprio_adapt"]
        # ---- Critic Info
        self.asymm_actor_critic = self.ppo_config["asymm_actor_critic"]
        self.critic_info_dim = self.ppo_config["critic_info_dim"]
        # ---- Point Cloud Info
        # self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.exp_name = full_config.train.experiment_name
        self.nn_dir = os.path.join("./outputs/",self.output_dir, "stage1_nn", self.exp_name)
        self.tb_dif = os.path.join("./outputs/",self.output_dir, "stage1_tb")
        self.silent = silent
        self.writer = None
        if not self.silent:
            try:
                os.makedirs(self.nn_dir, exist_ok=True)
                os.makedirs(self.tb_dif, exist_ok=True)
            except FileExistsError:
                pass
            if self.use_wandb:
                wandb_config = omegaconf.OmegaConf.to_container(full_config, resolve=False, throw_on_missing=True)
                wandb.init(
                    # Set the project where this run will be logged
                    entity="null", #fill in here
                    config=wandb_config,
                    project="null", 
                    sync_tensorboard=True,
                    name=self.exp_name,
                    settings=wandb.Settings(start_method="thread")
                    )
            
            self.writer = SummaryWriter(self.tb_dif + '/' + self.exp_name)
        # ---- Model ----
        
        net_config = {
            "actor_units": self.network_config.mlp.units,
            "priv_mlp_units": self.network_config.priv_mlp.units,
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "priv_info": self.priv_info,
            "proprio_adapt": self.proprio_adapt,
            "priv_info_dim": self.priv_info_dim,
            "critic_info_dim": self.critic_info_dim,
            "asymm_actor_critic": self.asymm_actor_critic,
            # "use_gt_point_cloud_info": self.network_config.use_gt_point_cloud_info,
            "point_mlp_units": self.network_config.point_mlp.units,
            "use_fine_contact": self.env.contact_input == "fine",
            "contact_mlp_units": self.network_config.contact_mlp.units,
            "use_point_transformer": self.network_config.use_point_transformer,
            # "multi_axis": self.env.multi_axis,
            "enable_palm_reskin": self.env.enable_palm_reskin,
            "enable_palm_binary": self.env.enable_palm_binary
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)

        # if there is no stage0_nn folder, there will be no teacher
        # TODO: infer order of teacher axis from file name
        self.teacher_models = []
        self.teacher_running_mean_stds = []
        self.teacher_priv_mean_stds = []
        # self.teacher_point_cloud_mean_stds = []
        self.teacher_nn_dir = os.path.join(self.output_dir, "stage0_nn")
        self.teacher_model_names = sorted(
            glob(os.path.join(self.teacher_nn_dir, "*.pth"))
        )
        self.teacher_axis = torch.from_numpy(
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])
        )
        for teacher_name in self.teacher_model_names:
            checkpoint = torch.load(teacher_name)
            teacher_net_config = copy.deepcopy(net_config)
            teacher_net_config["multi_axis"] = False
            teacher_model = ActorCritic(teacher_net_config)
            teacher_model.load_state_dict(checkpoint["model"])
            teacher_model.to(self.device)
            teacher_model.eval()
            teacher_running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
            teacher_running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            teacher_running_mean_std.eval()
            teacher_priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
            teacher_priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
            teacher_priv_mean_std.eval()
            # teacher_point_cloud_mean_std = RunningMeanStd(
            #     3,
            # ).to(self.device)
            # teacher_point_cloud_mean_std.load_state_dict(
            #     checkpoint["point_cloud_mean_std"]
            # )
            # teacher_point_cloud_mean_std.eval()
            self.teacher_models.append(teacher_model)
            self.teacher_running_mean_stds.append(teacher_running_mean_std)
            self.teacher_priv_mean_stds.append(teacher_priv_mean_std)
            # self.teacher_point_cloud_mean_stds.append(teacher_point_cloud_mean_std)

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        # self.point_cloud_mean_std = RunningMeanStd(
        #     3,
        # ).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config["learning_rate"])
        self.weight_decay = self.ppo_config.get("weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.last_lr, weight_decay=self.weight_decay
        )
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config["e_clip"]
        self.clip_value = self.ppo_config["clip_value"]
        self.entropy_coef = self.ppo_config["entropy_coef"]
        self.critic_coef = self.ppo_config["critic_coef"]
        self.bounds_loss_coef = self.ppo_config["bounds_loss_coef"]
        self.distill_loss_coef = self.ppo_config["distill_loss_coef"]
        self.gamma = self.ppo_config["gamma"]
        self.tau = self.ppo_config["tau"]
        self.truncate_grads = self.ppo_config["truncate_grads"]
        self.grad_norm = self.ppo_config["grad_norm"]
        self.value_bootstrap = self.ppo_config["value_bootstrap"]
        self.normalize_advantage = self.ppo_config["normalize_advantage"]
        self.normalize_input = self.ppo_config["normalize_input"]
        self.normalize_value = self.ppo_config["normalize_value"]
        self.normalize_priv = self.ppo_config["normalize_priv"]
        # self.normalize_point_cloud = self.ppo_config["normalize_point_cloud"]
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config["horizon_length"]
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config["minibatch_size"]
        print("batch size: ", self.batch_size)
        print("minibatch size: ", self.minibatch_size)
        self.mini_epochs_num = self.ppo_config["mini_epochs"]
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config["kl_threshold"]
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot
        self.save_freq = self.ppo_config["save_frequency"]
        self.save_best_after = self.ppo_config["save_best_after"]
        # ---- Tensorboard Logger ----
        self.extra_info = {}

        # ---- Rollout GIFs ----
        self.gif_frame_counter = 0
        self.gif_save_every_n = 7500
        self.gif_save_length = 600
        self.gif_frames = []

        self.episode_rewards = AverageScalarMeter(20000)
        self.episode_lengths = AverageScalarMeter(20000)
        # for task has a success/failure definition, we can evaluate success
        self.episode_success = AverageScalarMeter(20000)
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_shape[0],
            self.actions_num,
            self.priv_info_dim,
            self.critic_info_dim,
            # self.point_cloud_buffer_dim,
            self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config["max_agent_steps"]
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0


    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms):
        self.writer.add_scalar(
            "performance/RLTrainFPS",
            self.agent_steps / self.rl_train_time,
            self.agent_steps,
        )
        self.writer.add_scalar(
            "performance/EnvStepFPS",
            self.agent_steps / self.data_collect_time,
            self.agent_steps,
        )
        if self.use_wandb:
            wandb.log({
                        "losses/actor_loss": torch.mean(a_losses).item(),
                        "losses/bounds_loss": torch.mean(b_losses).item(),
                        "losses/critic_loss": torch.mean(c_losses).item(),
                        "losses/entropy": torch.mean(entropies).item(),
                        "info/last_lr": self.last_lr,
                        "info/e_clip": self.e_clip,
                        "info/kl": torch.mean(kls).item(),
                        "info/grad_norms": torch.mean(grad_norms).item()
                        })
        self.writer.add_scalar(
            "losses/actor_loss", torch.mean(a_losses).item(), self.agent_steps
        )
        self.writer.add_scalar(
            "losses/bounds_loss", torch.mean(b_losses).item(), self.agent_steps
        )
        self.writer.add_scalar(
            "losses/critic_loss", torch.mean(c_losses).item(), self.agent_steps
        )
        self.writer.add_scalar(
            "losses/entropy", torch.mean(entropies).item(), self.agent_steps
        )

        self.writer.add_scalar("info/last_lr", self.last_lr, self.agent_steps)
        self.writer.add_scalar("info/e_clip", self.e_clip, self.agent_steps)
        self.writer.add_scalar("info/kl", torch.mean(kls).item(), self.agent_steps)
        self.writer.add_scalar(
            "info/grad_norms", torch.mean(grad_norms).item(), self.agent_steps
        )

        for k, v in self.extra_info.items():
            if isinstance(v, torch.Tensor) and len(v.shape) != 0:
                continue
            if self.use_wandb:
                wandb.log({f"{k}": v})
            self.writer.add_scalar(f"{k}", v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_priv:
            self.priv_mean_std.eval()
        # if self.normalize_point_cloud:
        #     self.point_cloud_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_priv:
            self.priv_mean_std.train()
        # if self.normalize_point_cloud:
        #     self.point_cloud_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict["obs"])
        priv_info = obs_dict["priv_info"]
        if self.normalize_priv:
            priv_info = self.priv_mean_std(obs_dict["priv_info"])
        # if self.normalize_point_cloud:
        #     point_cloud = self.point_cloud_mean_std(
        #         obs_dict["point_cloud_info"].reshape(-1, 3)
        #     ).reshape((processed_obs.shape[0], -1, 3))
        # else:
        #     point_cloud = obs_dict["point_cloud_info"]
        input_dict = {
            "obs": processed_obs,
            "priv_info": priv_info,
            "rot_axis_buf": obs_dict["rot_axis_buf"],
            "critic_info": obs_dict["critic_info"],
            # "point_cloud_info": point_cloud,
        }
        res_dict = self.model.act(input_dict)
        res_dict["values"] = self.value_mean_std(res_dict["values"], True)
        return res_dict

    def teacher_inference(self, obs_dict, teacher_id):
        processed_obs = self.teacher_running_mean_stds[teacher_id](obs_dict["obs"])
        priv_info = self.teacher_priv_mean_stds[teacher_id](obs_dict["priv_info"])
        # point_cloud = self.teacher_point_cloud_mean_stds[teacher_id](
        #     obs_dict["point_cloud_info"].reshape(-1, 3)
        # ).reshape((processed_obs.shape[0], -1, 3))
        input_dict = {
            "obs": processed_obs,
            "priv_info": priv_info,
            # "point_cloud_info": point_cloud,
        }
        return self.teacher_models[teacher_id].act_inference(input_dict)

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = (
            self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size
        )

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            (
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                grad_norms,
            ) = self.train_epoch()
            self.storage.data_dict = None

            (
                a_losses,
                b_losses,
                c_losses,
                entropies,
                kls,
                grad_norms,
            ) = multi_gpu_aggregate_stats(
                [a_losses, b_losses, c_losses, entropies, kls, grad_norms]
            )
            mean_rewards, mean_lengths, mean_success = multi_gpu_aggregate_stats(
                [
                    torch.Tensor([self.episode_rewards.get_mean()])
                    .float()
                    .to(self.device),
                    torch.Tensor([self.episode_lengths.get_mean()])
                    .float()
                    .to(self.device),
                    torch.Tensor([self.episode_success.get_mean()])
                    .float()
                    .to(self.device),
                ]
            )
            for k, v in self.extra_info.items():
                if type(v) is not torch.Tensor:
                    v = torch.Tensor([v]).float().to(self.device)
                self.extra_info[k] = multi_gpu_aggregate_stats(v[None].to(self.device))

            if self.is_log_node():
                all_fps = self.agent_steps / (time.time() - _t)
                last_fps = (
                    self.batch_size
                    if not self.multi_gpu
                    else self.batch_size * self.rank_size
                ) / (time.time() - _last_t)
                _last_t = time.time()
                info_string = (
                    f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                    f"Last FPS: {last_fps:.1f} | "
                    f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                    f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                    f"Current Best: {self.best_rewards:.2f}"
                )
                print(info_string)
                if not self.silent:
                    self.write_stats(
                        a_losses, c_losses, b_losses, entropies, kls, grad_norms
                    )
                    self.writer.add_scalar(
                        "episode_rewards/step", mean_rewards, self.agent_steps
                    )
                    self.writer.add_scalar(
                        "episode_lengths/step", mean_lengths, self.agent_steps
                    )
                    self.writer.add_scalar(
                        "episode_success/step", mean_success, self.agent_steps
                    )
                    if self.use_wandb:
                        wandb.log({
                            "episode_rewards/step": mean_rewards,
                            "episode_lengths/step": mean_lengths,
                            "episode_success/step": mean_success,
                        })
                    checkpoint_name = f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}"
                    if self.save_freq > 0:
                        if (self.epoch_num % self.save_freq == 0) and (
                            mean_rewards <= self.best_rewards
                        ):
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            self.save(os.path.join(self.nn_dir, f"last"))

                    if (
                        mean_rewards > self.best_rewards
                        and self.agent_steps >= self.save_best_after
                    ):
                        print(f"save current best reward: {mean_rewards:.2f}")
                        # remove previous best file
                        prev_best_ckpt = os.path.join(
                            self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth"
                        )
                        if os.path.exists(prev_best_ckpt):
                            os.remove(prev_best_ckpt)
                        self.best_rewards = mean_rewards
                        self.save(
                            os.path.join(self.nn_dir, f"best_reward_{mean_rewards:.2f}")
                        ) #add "/" in front and use absolute path, otherwise the model will not save and there will not be an error message for this

        print("max steps achieved")

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.normalize_priv:
            weights["priv_mean_std"] = self.priv_mean_std.state_dict()
        # if self.normalize_point_cloud:
        #     weights["point_cloud_mean_std"] = self.point_cloud_mean_std.state_dict()
        if self.value_mean_std:
            weights["value_mean_std"] = self.value_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn, **kwargs):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        # if self.normalize_point_cloud:
        #     self.point_cloud_mean_std.load_state_dict(
        #         checkpoint["point_cloud_mean_std"]
        #     )

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        # if self.normalize_point_cloud:
        #     self.point_cloud_mean_std.load_state_dict(
        #         checkpoint["point_cloud_mean_std"]
            # )

    def _build_input_dict(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # if self.normalize_point_cloud:
        #     point_cloud = self.point_cloud_mean_std(
        #         obs_dict["point_cloud_info"].reshape(-1, 3)
        #     ).reshape((obs_dict["obs"].shape[0], -1, 3))
        # else:
        #     point_cloud = obs_dict["point_cloud_info"]
        return {
            "obs": self.running_mean_std(obs_dict["obs"]),
            "priv_info": self.priv_mean_std(obs_dict["priv_info"])
            if self.normalize_priv
            else obs_dict["priv_info"],
            # "point_cloud_info": point_cloud,
        }

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls, grad_norms = [], [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                (
                    value_preds,
                    old_action_log_probs,
                    advantage,
                    old_mu,
                    old_sigma,
                    returns,
                    actions,
                    obs,
                    priv_info,
                    critic_info,
                    # point_cloud_info,
                    rot_axis_buf,
                ) = self.storage[i]

                teacher_mu = torch.zeros_like(old_mu)
                for teacher_id in range(len(self.teacher_models)):
                    teacher_hash = self.teacher_axis[teacher_id]
                    teacher_hash = (
                        (teacher_hash[0] + 2) * 5
                        + (teacher_hash[1] + 2) * 25
                        + (teacher_hash[2] + 2) * 125
                    )
                    rot_axis_hash = (
                        (rot_axis_buf[:, 0] + 2) * 5
                        + (rot_axis_buf[:, 1] + 2) * 25
                        + (rot_axis_buf[:, 2] + 2) * 125
                    )
                    batch_idx = rot_axis_hash == teacher_hash
                    teacher_dict = {
                        "obs": obs[batch_idx],
                        "priv_info": priv_info[batch_idx],
                        # "point_cloud_info": point_cloud_info[batch_idx],
                    }
                    teacher_mu[batch_idx], *_ = self.teacher_inference(
                        teacher_dict, teacher_id
                    )

                obs = self.running_mean_std(obs)
                # if self.normalize_point_cloud:
                #     point_cloud_info = self.point_cloud_mean_std(
                #         point_cloud_info.reshape(-1, 3)
                #     ).reshape((obs.shape[0], -1, 3))
                batch_dict = {
                    "prev_actions": actions,
                    "obs": obs,
                    "priv_info": self.priv_mean_std(priv_info)
                    if self.normalize_priv
                    else priv_info,
                    "rot_axis_buf": rot_axis_buf,
                    "critic_info": critic_info,
                    # "point_cloud_info": point_cloud_info,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.e_clip, self.e_clip
                )
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros_like(mu)
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]
                ]
                d_loss = torch.mean((mu - teacher_mu.detach()) ** 2)
                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                    + d_loss * self.distill_loss_coef
                )
                self.optimizer.zero_grad()
                loss.backward()

                if self.multi_gpu:
                    # batch all_reduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(
                                    param.grad.data
                                )
                                / self.rank_size
                            )
                            offset += param.numel()

                grad_norms.append(
                    torch.norm(
                        torch.cat([p.reshape(-1) for p in self.model.parameters()])
                    )
                )
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            kls.append(av_kls)

            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            if self.multi_gpu:
                lr_tensor = torch.tensor([self.last_lr], device=self.device)
                dist.broadcast(lr_tensor, 0)
                lr = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.last_lr

        self.rl_train_time += time.time() - _t
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data("obses", n, self.obs["obs"])
            self.storage.update_data("priv_info", n, self.obs["priv_info"])
            self.storage.update_data("rot_axis_buf", n, self.obs["rot_axis_buf"])
            self.storage.update_data("critic_info", n, self.obs["critic_info"])
            # self.storage.update_data(
            #     "point_cloud_info", n, self.obs["point_cloud_info"]
            # )
            for k in ["actions", "neglogpacs", "values", "mus", "sigmas"]:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict["actions"], -1.0, 1.0)

            # render() is called during env.step()
            # to save time, save gif only per gif_save_every_n steps
            # 1 step = #gpu * #envs agent steps
            record_frame = False
            if (
                self.gif_frame_counter >= self.gif_save_every_n
                and self.gif_frame_counter % self.gif_save_every_n
                < self.gif_save_length
            ):
                record_frame = True
            record_frame = record_frame and int(os.getenv("LOCAL_RANK", "0")) == 0
            self.env.enable_camera_sensors = record_frame
            self.gif_frame_counter += 1

            self.obs, rewards, self.dones, infos = self.env.step(actions)

            # if record_frame and self.env.with_camera:
            # if record_frame:
            #     self.gif_frames.append(self.env.capture_frame())
            #     # add frame to GIF
            #     if (
            #         self.writer is not None
            #         and len(self.gif_frames) == self.gif_save_length
            #     ):
            #         frame_array = np.array(self.gif_frames)[None]  # add batch axis
            #         self.writer.add_video(
            #             "rollout_gif",
            #             frame_array,
            #             global_step=self.agent_steps,
            #             dataformats="NTHWC",
            #             fps=20,
            #         )
            #         if self.use_wandb:
            #             wand_frame_array = np.transpose(frame_array[0], [0, 3, 1, 2])
            #             wandb.log({"rollout_gif": wandb.Video(wand_frame_array, format='gif', fps=20)})
            #         self.writer.flush()
            #         self.gif_frames.clear()

            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data("dones", n, self.dones)
            rewards = rewards.to(self.device)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * infos["time_outs"].unsqueeze(1).float()
                )
            self.storage.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            assert isinstance(infos, dict), "Info Should be a Dict"
            self.extra_info = infos
            if "episode_success" in self.extra_info.keys():
                eps_success = self.extra_info["episode_success"]
                self.episode_success.update(eps_success[done_indices])

            not_dones = (1.0 - self.dones.float()).to(self.device)

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict["values"]

        self.agent_steps = (
            (self.agent_steps + self.batch_size)
            if not self.multi_gpu
            else self.agent_steps + self.batch_size * self.rank_size
        )
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict["returns"]
        values = self.storage.data_dict["values"]
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict["values"] = values
        self.storage.data_dict["returns"] = returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
