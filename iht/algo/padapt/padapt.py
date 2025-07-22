# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
import os
import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from gum.labs.dexit.algo.policy import Policy

from omegaconf import DictConfig
from tensorboardX import SummaryWriter
import wandb
from termcolor import cprint

from gum.labs.dexit.algo.models.models import ActorCritic
from gum.labs.dexit.algo.models.running_mean_std import RunningMeanStd
from gum.labs.dexit.utils.env import priv_info_dict_from_config, priv_info_dim_from_dict
from gum.labs.dexit.utils.misc import (
    AverageScalarMeter,
    multi_gpu_aggregate_stats,
    tprint,
)

if TYPE_CHECKING:
    from gum.labs.dexit.tasks import AllegroHandHoraIHT

import torch
import torch.distributed as dist

def proprio_adapt_net_config(dexit_config: DictConfig) -> Dict[str, Any]:
    network_config = dexit_config.train.network
    ppo_config = dexit_config.train.ppo
    padapt_config = dexit_config.train.padapt
    env_config = dexit_config.task.env
    # priv_info_dim = priv_info_dim_from_dict(
    #     priv_info_dict_from_config(env_config.privInfo)
    # )
    if env_config.privInfo["enableNetContactF"] == True: #train with force balance penalty
        priv_info_dim=39
    else:
        priv_info_dim=33
    stage_1_used_point_cloud_info = env_config.hora.point_cloud_sampled_dim > 0

    use_perceived_point_cloud_info = padapt_config["use_perceived_point_cloud"]

    if (
        env_config.hora.perceived_point_cloud_sampled_dim > 0
        and not use_perceived_point_cloud_info
    ):
        cprint(
            "It appears you want to use perceived point clouds but train.padapt.use_perceived_point_cloud is False",
            "red",
            attrs=["bold"],
        )

    use_gt_point_cloud_info = stage_1_used_point_cloud_info and (
        not use_perceived_point_cloud_info
    )
    use_perceived_point_cloud_uncertainty = padapt_config.get(
        "use_perceived_point_cloud_uncertainty", False
    )
    assert not (
        padapt_config["visual_distillation"] and not env_config.rgbd_camera["enable"]
    ), "Depth cam must be enabled for visual distillation"
    return {
        "actor_units": network_config.mlp.units,
        "priv_mlp_units": network_config.priv_mlp.units,
        "actions_num": env_config.numActions,
        "input_shape": (dexit_config.task.env.numObservations,),
        "proprio_history_len": env_config.hora.propHistoryLen,
        "priv_info": ppo_config["priv_info"],
        "proprio_adapt": ppo_config["proprio_adapt"],
        "priv_info_dim": priv_info_dim,
        "critic_info_dim": ppo_config["critic_info_dim"],  # only for compatibility
        "asymm_actor_critic": ppo_config[
            "asymm_actor_critic"
        ],  # only for compatibility
        # "use_gt_point_cloud_info": use_gt_point_cloud_info,
        "use_gt_point_cloud_info": False,
        "use_perceived_point_cloud_info": use_perceived_point_cloud_info,
        "use_perceived_point_cloud_uncertainty": use_perceived_point_cloud_uncertainty,
        # point_mlp_units used only if use_perceived_point_cloud_info=True
        "point_mlp_units": network_config.point_mlp.units,
        "with_noisy_obj_quat": padapt_config["with_noisy_obj_quat"],
        "noisy_obj_quat_xyz_only": padapt_config["noisy_obj_quat_xyz_only"],
        "with_debias_obj_xyz": padapt_config["with_debias_obj_xyz"],
        "with_goal_quat": padapt_config["with_goal_quat"],
        "visual_distillation": padapt_config["visual_distillation"],
        "use_point_transformer": network_config.use_point_transformer,
        "conv_with_batch_norm": padapt_config["conv_with_batch_norm"],
        "use_deformable_conv": padapt_config["use_deformable_conv"],
        "separate_temporal_fusion": padapt_config["separate_temporal_fusion"],
        "use_temporal_transformer": padapt_config["use_temporal_transformer"],
        "use_position_encoder": padapt_config["use_position_encoder"],
        "use_fine_contact": padapt_config["expert_fine_contact"],  # stage 1 input only
        "contact_mlp_units": network_config.contact_mlp.units,
        "contact_distillation": padapt_config["contact_distillation"],
        "enable_palm_reskin": env_config["enable_palm_reskin"],
        "enable_palm_binary": env_config["enable_palm_binary"]
    }


class ProprioAdapt(Policy):
    def _init_impl(self, env: "AllegroHandHoraIHT", output_dir, full_config, silent=False):
        # ---- MultiGPU ----
        self.multi_gpu = full_config.train.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            self.device = "cuda:" + str(self.rank)
            print(f"current rank: {self.rank} and use device {self.device}")
        else:
            self.rank = -1
            self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.padapt_config = full_config.train.padapt
        self.env_task_config = full_config.task.env
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        # ---- Priv Info ----
        self.priv_info = self.ppo_config["priv_info"]
        self.priv_info_dim = self.env.priv_info_dim
        self.proprio_adapt = self.ppo_config["proprio_adapt"]
        self.proprio_hist_dim = self.env.prop_hist_len
        # ---- Critic Info
        self.asymm_actor_critic = self.ppo_config["asymm_actor_critic"]
        self.critic_info_dim = self.ppo_config["critic_info_dim"]
        # ---- Point Cloud Info
        stage_1_used_point_cloud_info = self.env.point_cloud_sampled_dim > 0

        # ---- Padapt Config
        self.action_imitation = self.padapt_config["action_imitation"]
        self.with_noisy_obj_quat = self.padapt_config["with_noisy_obj_quat"]
        self.noisy_obj_quat_xyz_only = self.padapt_config["noisy_obj_quat_xyz_only"]
        self.with_goal_quat = self.padapt_config["with_goal_quat"]
        self.student_cache_suffix = self.padapt_config["cache_suffix"]
        self.iter_per_step = self.padapt_config["iter_per_step"]
        self.use_fine_contact = self.padapt_config["expert_fine_contact"]
        self.visual_distillation = self.padapt_config["visual_distillation"]
        self.contact_distillation = self.padapt_config["contact_distillation"]
        self.enable_palm_reskin = self.env_task_config["enable_palm_reskin"]
        self.enable_palm_binary = self.env_task_config["enable_palm_binary"]

        assert not (
            self.visual_distillation and not self.env.enable_depth_camera
        ), "Depth cam must be enabled for visual distillation"

        # ---- Model ----
        # Net config reads this value from config file, make sure there are no
        # surprises here
        assert full_config.task.env.hora.propHistoryLen == self.proprio_hist_dim
        net_config = proprio_adapt_net_config(full_config)
        #shape error is here, input is 139?
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        teacher_net_config = copy.deepcopy(net_config)
        teacher_net_config["proprio_adapt"] = False
        teacher_net_config["visual_distillation"] = False
        teacher_net_config["use_gt_point_cloud_info"] = stage_1_used_point_cloud_info
        teacher_net_config["use_perceived_point_cloud_info"] = False
        teacher_net_config["use_perceived_point_cloud_uncertainty"] = False
        self.teacher_model = ActorCritic(teacher_net_config)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        self.normalize_priv = self.ppo_config["normalize_priv"]
        self.normalize_point_cloud = self.ppo_config["normalize_point_cloud"]
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.priv_mean_std.eval()
        self.point_cloud_mean_std = RunningMeanStd(
            3,
        ).to(self.device)
        self.point_cloud_mean_std.eval()
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, 32)).to(self.device)
        self.sa_mean_std.train()
        self.noisy_obj_quat_mean_std = RunningMeanStd((self.proprio_hist_dim, 7)).to(
            self.device
        )
        self.noisy_obj_quat_mean_std.train()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join("/home/robotdev/gum_ws/src/GUM/outputs/", self.output_dir, f"{self.student_cache_suffix}_nn")
        self.tb_dir = os.path.join("/home/robotdev/gum_ws/src/GUM/outputs/", self.output_dir, f"{self.student_cache_suffix}_tb")
        self.silent = silent
        self.writer = None
        if self.is_log_node() and not self.silent:
            os.makedirs(self.nn_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_dir)
        # ---- Rollout GIFs ----
        self.gif_frame_counter = 0
        self.gif_save_every_n = 2500
        self.gif_save_length = 600
        self.gif_frames = []
        # ---- Misc ----
        self.extra_info = {}
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        # for task has a success/failure definition, we can evaluate success
        self.episode_success = AverageScalarMeter(20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        # ---- Timing ----
        self.data_collect_time = 0
        self.optim_time = 0
        self.all_time = 0
        # ---- Optim ----
        adapt_params = []
        adapt_param_names = []
        freeze_names = []
        for name, p in self.model.named_parameters():
            if (
                "adapt_tconv" in name
                or "depth" in name
                or "contact" in name
                or "point_mlp" in name
                or "pc_modulator" in name
                or "xyz_debias" in name
            ):
                adapt_param_names.append(name)
                adapt_params.append(p)
            else:
                freeze_names.append(name)
                p.requires_grad = False
        print("Training Parameter List:")
        print(adapt_param_names)
        print("Freeze Parameter List:")
        print(freeze_names)

        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.step_length = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

    def _build_input_dict(
        self, obs_dict: Dict[str, torch.Tensor], include_priv_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        input_dict = {
            "obs": self.running_mean_std(obs_dict["obs"]).detach(),
            "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
            "noisy_obj_quat": self.noisy_obj_quat_mean_std(
                obs_dict["noisy_obj_quaternion"].detach()
            ),
            "goal_quat": obs_dict["goal_quaternion"].detach(),
            "fine_contact_info": obs_dict["fine_contact_info"].detach()
            if self.contact_distillation
            else None,
            "depth_buf": obs_dict["depth_buffer"].detach()
            if self.visual_distillation
            else None,
        }
        if "perceived_point_cloud_info" in obs_dict:
            cloud, uncert = obs_dict["perceived_point_cloud_info"]
            input_dict["perceived_point_cloud_info"] = (
                self._maybe_normalize_point_cloud(cloud),
                uncert,
            )
        if include_priv_info:
            input_dict["priv_info"] = (
                self.priv_mean_std(obs_dict["priv_info"])
                if self.normalize_priv
                else obs_dict["priv_info"]
            )
            input_dict["point_cloud_info"] = self._maybe_normalize_point_cloud(
                obs_dict["point_cloud_info"]
            )

        if self.enable_palm_reskin:
            input_dict["palm_reskin_info"] = obs_dict["palm_reskin_info"]
        
        if self.enable_palm_binary:
            input_dict["palm_binary_info"] = obs_dict["palm_binary_info"]

        return input_dict

    def _maybe_normalize_point_cloud(self, point_cloud: torch.Tensor) -> torch.Tensor:
        if self.normalize_point_cloud:
            return self.point_cloud_mean_std(point_cloud.reshape(-1, 3)).reshape(
                point_cloud.shape[0], -1, 3
            )
        else:
            return point_cloud

    def compute_obj_pose_loss(self, gt_obj_pose_history: torch.Tensor) -> torch.Tensor:
        normalized_gt_obj_pose = self.noisy_obj_quat_mean_std.forward(
            gt_obj_pose_history, eval_only=True
        ).detach()
        return torch.nn.functional.mse_loss(
            self.model.noisy_obj_quat_last[..., -3:],
            normalized_gt_obj_pose[..., -3:].detach(),
        )

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps = (
            self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size
        )

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while self.agent_steps <= 1e9:
            _net_t = time.time()
            input_dict = self._build_input_dict(obs_dict, include_priv_info=True)
            latent_loss_value = 0
            action_loss_value = 0
            object_pose_loss_value = 0
            for _ in range(self.iter_per_step):
                mu, _, _, e = self.model._actor_critic(input_dict)
                with torch.no_grad():
                    e_gt = self.teacher_model.extrin_from_priv_info(input_dict)
                loss = ((e - e_gt) ** 2).mean()
                latent_loss_value += loss.item()
                if self.action_imitation:
                    mu_gt, *_ = self.teacher_model.act_inference(input_dict)
                    action_loss = ((mu - mu_gt.detach()) ** 2).mean()
                    action_loss_value += action_loss.item()
                    loss = loss + action_loss

                # Loss on predict object xyz bias
                object_pose_loss = self.compute_obj_pose_loss(
                    obs_dict["obj_quaternion_gt"]
                )
                object_pose_loss_value += 2 * object_pose_loss.item()
                loss = loss + object_pose_loss

                self.optim.zero_grad()
                loss.backward()

                if self.multi_gpu:
                    # batch all_reduce ops:
                    # see https://github.com/entity-neural-network/incubator/pull/220
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

                self.optim.step()
            latent_loss_value /= self.iter_per_step
            action_loss_value /= self.iter_per_step
            object_pose_loss_value /= self.iter_per_step

            self.optim_time += time.time() - _net_t

            record_frame = False
            if (
                self.gif_frame_counter >= self.gif_save_every_n
                and self.gif_frame_counter % self.gif_save_every_n
                < self.gif_save_length
            ):
                record_frame = False
            record_frame = record_frame and int(os.getenv("LOCAL_RANK", "0")) == 0
            self.gif_frame_counter += 1

            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)

            _env_t = time.time()
            obs_dict, r, done, infos = self.env.step(mu)
            self.data_collect_time += time.time() - _env_t

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
            #         self.writer.flush()
            #         self.gif_frames.clear()

            # ---- statistics
            self.step_reward += r.to(self.device)
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            self.extra_info = infos
            if "episode_success" in self.extra_info.keys():
                eps_success = self.extra_info["episode_success"]
                self.episode_success.update(eps_success[done_indices])

            not_dones = (1.0 - done.float()).to(self.device)
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            (
                mean_rewards,
                mean_lengths,
                mean_success,
                mean_action_loss,
                mean_latent_loss,
                mean_object_pose_loss,
            ) = multi_gpu_aggregate_stats(
                [
                    torch.Tensor([self.mean_eps_reward.get_mean()])
                    .float()
                    .to(self.device),
                    torch.Tensor([self.mean_eps_length.get_mean()])
                    .float()
                    .to(self.device),
                    torch.Tensor([self.episode_success.get_mean()])
                    .float()
                    .to(self.device),
                    torch.Tensor([action_loss_value]).float().to(self.device),
                    torch.Tensor([latent_loss_value]).float().to(self.device),
                    torch.Tensor([object_pose_loss_value]).float().to(self.device),
                ]
            )
            for k, v in self.extra_info.items():
                if type(v) is not torch.Tensor:
                    v = torch.Tensor([v]).float().to(self.device)
                self.extra_info[k] = multi_gpu_aggregate_stats(v[None].to(self.device))

            if self.is_log_node():
                self.agent_steps = (
                    (self.agent_steps + self.batch_size)
                    if not self.multi_gpu
                    else self.agent_steps + self.batch_size * self.rank_size
                )
                if not self.silent:
                    self.log_tensorboard(
                        mean_rewards,
                        mean_lengths,
                        mean_success,
                        mean_action_loss,
                        mean_latent_loss,
                        mean_object_pose_loss,
                    )

                    if self.agent_steps % 1e8 == 0:
                        self.save(
                            os.path.join(
                                self.nn_dir, f"model_{self.agent_steps // 1e8}00m"
                            )
                        ) #add "/" in front and use absolute path, otherwise the model will not save and there will not be an error message for this
                        self.save(os.path.join(self.nn_dir, f"model_last"))

                    if mean_rewards > self.best_rewards:
                        self.save(os.path.join(self.nn_dir, f"model_best"))
                        self.best_rewards = mean_rewards

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
                    f"Network Time: {self.optim_time / 60:.1f} min | "
                    f"Current Best: {self.best_rewards:.2f}"
                )
                tprint(info_string)

    def log_tensorboard(
        self,
        mean_rewards,
        mean_lengths,
        mean_success,
        mean_action_loss,
        mean_latent_los,
        mean_object_xyz_bias_loss,
    ):
        self.writer.add_scalar("episode_rewards/step", mean_rewards, self.agent_steps)
        self.writer.add_scalar("episode_lengths/step", mean_lengths, self.agent_steps)
        self.writer.add_scalar("episode_success/step", mean_success, self.agent_steps)
        self.writer.add_scalar("action_loss/step", mean_action_loss, self.agent_steps)
        self.writer.add_scalar("latent_loss/step", mean_latent_los, self.agent_steps)
        self.writer.add_scalar(
            "object_pose_loss/step", mean_object_xyz_bias_loss, self.agent_steps
        )
        for k, v in self.extra_info.items():
            if isinstance(v, torch.Tensor) and len(v.shape) != 0:
                continue
            self.writer.add_scalar(f"{k}", v, self.agent_steps)

    def restore_train(self, fn, only_teacher_weights=True):
        print("Loading checkpoint at", fn)
        checkpoint = torch.load(fn)
        model_ckpt = checkpoint["model"]
        cprint("careful, using non-strict matching", "red", attrs=["bold"])
        if only_teacher_weights:
            # for the stage 2 model, we still need to restore policy weights
            is_policy_key = lambda k: "actor_mlp." in k or "value." in k or "mu." in k
            policy_ckpt = {k: v for k, v in model_ckpt.items() if is_policy_key(k)}
            self.model.load_state_dict(policy_ckpt, strict=False)
        else:
            self.model.load_state_dict(model_ckpt, strict=False)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        self.teacher_model.load_state_dict(model_ckpt, strict=False)
        if self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(
                checkpoint["point_cloud_mean_std"]
            )

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        self.noisy_obj_quat_mean_std.load_state_dict(checkpoint["noisy_obj_quat_std"])
        self.model.load_state_dict(checkpoint["model"])
        self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights["sa_mean_std"] = self.sa_mean_std.state_dict()
        if self.noisy_obj_quat_mean_std:
            weights["noisy_obj_quat_std"] = self.noisy_obj_quat_mean_std.state_dict()
        torch.save(weights, f"{name}.ckpt")
