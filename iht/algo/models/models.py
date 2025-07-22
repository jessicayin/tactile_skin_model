# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from .block import ConvTransform, Modulator, TemporalConv, TemporalTransformer


class MLP(nn.Module):
    def __init__(self, units, input_size, with_last_activation=True):
        super(MLP, self).__init__()
        # use with_last_activation=False when we need the network to output raw values before activation
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        if not with_last_activation:
            layers.pop()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        policy_input_dim = kwargs.get("input_shape")[0]
        actions_num = kwargs.get("actions_num")
        self.units = kwargs.get("actor_units")
        self.proprio_history_len = kwargs.get("proprio_history_len")
        self.priv_mlp = kwargs.get("priv_mlp_units")
        # See explanation below for differences between use_gt_point_cloud_info
        # and use_perceived_point_cloud_info
        self.use_gt_point_cloud_info = kwargs.get("use_gt_point_cloud_info")
        self.use_perceived_point_cloud_info = kwargs.get(
            "use_perceived_point_cloud_info"
        )
        self.use_perceived_point_cloud_uncertainty = kwargs.get(
            "use_perceived_point_cloud_uncertainty"
        )
        self.use_fine_contact = kwargs.get("use_fine_contact")
        self.point_mlp_units = kwargs.get("point_mlp_units")
        self.use_point_transformer = kwargs.get("use_point_transformer")
        self.contact_mlp_units = kwargs.get("contact_mlp_units")
        self.with_noisy_obj_quat = kwargs.get("with_noisy_obj_quat", False)
        self.noisy_obj_quat_xyz_only = kwargs.get("noisy_obj_quat_xyz_only", False)
        # self.with_debias_obj_xyz = kwargs.get("debias_obj_xyz", True)
        self.debias_object_xyz = False
        self.with_goal_quat = kwargs.get("with_goal_quat", False)
        self.visual_distillation = kwargs.get("visual_distillation", False)
        self.contact_distillation = kwargs.get("contact_distillation", False)
        self.separate_temporal_fusion = kwargs.get("separate_temporal_fusion")
        out_size = self.units[-1]
        self.priv_info = kwargs["priv_info"]
        self.priv_info_stage2 = kwargs["proprio_adapt"]
        self.enable_palm_reskin= kwargs["enable_palm_reskin"]
        self.enable_palm_binary = kwargs["enable_palm_binary"]
        self.with_noisy_obj_quat = self.with_noisy_obj_quat and self.priv_info_stage2

        if self.use_gt_point_cloud_info and self.use_perceived_point_cloud_info:
            # If use_gt_pc_info is True:
            #   - Stage 1 model: uses point_mlp
            #   - Stage 2 model: this indicates that stage1 model used point_mlp,
            #       so output of fusion model includes 32 dims to match stage 1 extrins.
            #       No point_mlp is created for this.
            # If use_perceived_pc_info is True:
            #   - Stage 1 model: ignored
            #   - Stage 2 model: ouput of fusion model is only 8 dims
            #       A point_mlp will be created to process the perceived point clouds
            raise ValueError(
                "At most one of use_gt_point_cloud_info or "
                "use_perceived_point_cloud_info can be true."
            )
        if (
            self.use_perceived_point_cloud_uncertainty
            and not self.use_perceived_point_cloud_info
        ):
            raise ValueError(
                "Tried to pass point cloud uncertainty as input w/o using "
                "the point cloud input itself."
            )

        if self.priv_info:
            policy_input_dim += self.priv_mlp[-1]
            # the output of env_mlp and proprioceptive regression
            # should both be before activation
            self.env_mlp = MLP(
                units=self.priv_mlp,
                input_size=kwargs["priv_info_dim"],
                with_last_activation=False,
            )

            if self.priv_info_stage2:
                if self.separate_temporal_fusion:
                    # only proprioception is encoded
                    temporal_fusing_input_dim = 32
                    temporal_fusing_output_dim = 32
                else:
                    temporal_fusing_input_dim = 32
                    if self.visual_distillation:
                        temporal_fusing_input_dim += 32
                    if self.contact_distillation:
                        temporal_fusing_input_dim += 32
                    if self.with_noisy_obj_quat:
                        temporal_fusing_input_dim += (
                            3 if self.noisy_obj_quat_xyz_only else 7
                        )
                    if self.with_goal_quat:
                        temporal_fusing_input_dim += 8
                    if self.enable_palm_reskin:
                        temporal_fusing_input_dim += 48
                    if self.enable_palm_binary:
                        # temporal_fusing_input_dim += 16
                        temporal_fusing_input_dim += 48 
                    temporal_fusing_output_dim = 8
                    if (
                        self.use_gt_point_cloud_info
                        or self.use_perceived_point_cloud_uncertainty
                    ):
                        # Means stage 1 used point cloud, match its extrinsics size
                        temporal_fusing_output_dim += 32
                use_temporal_transformer = kwargs.get("use_temporal_transformer")
                use_position_encoder = kwargs.get("use_position_encoder")
                if use_temporal_transformer:
                    self.adapt_tconv = TemporalTransformer(
                        embedding_dim=32,
                        n_head=2,
                        depth=2,
                        output_dim=temporal_fusing_output_dim,
                        input_dim=temporal_fusing_input_dim,
                        use_pe=use_position_encoder,
                    )
                else:
                    self.adapt_tconv = TemporalConv(
                        temporal_fusing_input_dim, temporal_fusing_output_dim
                    )

        if self.use_gt_point_cloud_info or self.use_perceived_point_cloud_info:
            # point_mlp only created if stage 1 or if using perceived point clouds
            if not self.priv_info_stage2 or self.use_perceived_point_cloud_info:
                if self.use_point_transformer:
                    self.point_mlp = TemporalTransformer(
                        32, 2, 1, 32, use_pe=False, pre_ffn=True, input_dim=3
                    )
                else:
                    self.point_mlp = MLP(units=self.point_mlp_units, input_size=3)
            # In stage 2, the policy input dim will be the same even if not
            # using point mlp, because extrin need to match those from stage 1
            policy_input_dim += self.point_mlp_units[-1]
        if self.contact_distillation:
            self.contact_mlp_s2 = MLP(units=self.contact_mlp_units, input_size=4) #input_size=7 for allegrohandhora, 4 for iht

        if self.visual_distillation:
            conv_with_batch_norm = kwargs.get("conv_with_batch_norm")
            use_deformable_conv = kwargs.get("use_deformable_conv")
            self.depth_conv = ConvTransform(conv_with_batch_norm, use_deformable_conv)
            if self.separate_temporal_fusion:
                self.depth_tfuse = nn.Linear(6 * 32, 32)
                self.all_fuse = nn.Linear(32 + 32, 40)

        self.multi_axis = kwargs.get("multi_axis")
        if self.multi_axis:
            self.task_mlp = MLP(units=[16, 16], input_size=3)
            policy_input_dim += 16

        self.asymm_actor_critic = kwargs["asymm_actor_critic"]
        self.critic_info_dim = kwargs["critic_info_dim"]
        self.actor_mlp = MLP(units=self.units, input_size=policy_input_dim)
        self.value = (
            MLP(
                units=self.units + [1],
                input_size=policy_input_dim + self.critic_info_dim,
            )
            if self.asymm_actor_critic
            else torch.nn.Linear(out_size, 1)
        )
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
            requires_grad=True,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

        if self.use_perceived_point_cloud_uncertainty:
            self.pc_modulator = Modulator()

        # Make sure this model is only created if using object pose xyz
        self.with_debias_obj_xyz = (
            self.with_noisy_obj_quat
            and self.noisy_obj_quat_xyz_only
            and self.with_debias_obj_xyz
        )
        if self.with_debias_obj_xyz:
            self.xyz_debias = MLP(
                (512, 128, 32, 3),
                temporal_fusing_input_dim * self.proprio_history_len,
                with_last_activation=False,
            )

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, *_ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            "neglogpacs": -distr.log_prob(selected_action).sum(
                1
            ),  # self.neglogp(selected_action, mu, sigma, logstd),
            "values": value,
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, _, _, extrin = self._actor_critic(obs_dict)
        return mu, extrin

    def _privileged_pred(self, joint_x, visual_x, tactile_x, noisy_obj_quat, palm_reskin_x, palm_binary_x):
        #NOTE: may need to adjust reskin depending on sampling frequency
        # if self.with_debias_obj_xyz:
        #     noisy_obj_quat = self.debias_object_xyz(joint_x, noisy_obj_quat)
        self.noisy_obj_quat_last = noisy_obj_quat
        # three part: modality specific transform, cross modality fusion,
        batch_dim, _ = joint_x.shape[:2]
        # ---- modality specific transform [*_x to *_t]
        joint_t = joint_x
        visual_t = None
        if self.visual_distillation:
            n, t, c, h, w = visual_x.shape
            visual_x = visual_x.reshape(n * t, c, h, w)
            visual_t = self.depth_conv(visual_x)
            visual_t = visual_t.reshape(n, t, -1)

        tactile_t = None
        if self.contact_distillation:
            # import ipdb; ipdb.set_trace()
            # contact_feat = self.contact_mlp_s2(tactile_x)
            # valid_mask = tactile_x[..., [-1]] >= 0
            # import ipdb; ipdb.set_trace()
            # tactile_t = torch.sum(contact_feat * valid_mask, dim=2) / (
            #     torch.sum(valid_mask, dim=2) + 1e-9
            # )
            contact_feat = self.contact_mlp_s2(tactile_x)
            tactile_t = contact_feat
            # tactile_t = tactile_x
                
        # ---- cross modality fusion, and temporal fusion
        if self.separate_temporal_fusion:
            # temporal fusion first and then cross modality
            joint_t_t = self.adapt_tconv(joint_t)
            if self.visual_distillation:
                visual_t = visual_t.reshape(batch_dim, -1)
                visual_t_t = self.depth_tfuse(visual_t)
                joint_visual_t_t = torch.cat(
                    [joint_t_t, visual_t_t], dim=-1
                )  # cross modality fusion
            else:
                joint_visual_t_t = joint_t_t
            extrin_pred = self.all_fuse(joint_visual_t_t)
        else:
            # cross modality first and then temporal fusion
            # if the visual feature updates asynchronously, the temporal dimension would not match
            info_list = [joint_t]
            if self.visual_distillation:
                if visual_t.shape[1] != joint_t.shape[1]:
                    # image sensing is typically slower than proprioception sensing (30 Hz v.s. 300Hz)
                    # e.g. temporal length of joint_t is 30 while for visual_t it's 6
                    # we can naively repeat 5 times for each image feature, as in the initial implementation
                    # ----
                    # num_repeat = joint_t.shape[1] // visual_t.shape[1]
                    # visual_t = visual_t[:, None].repeat(1, num_repeat, 1, 1).transpose(1, 2)
                    # visual_t = visual_t.reshape(batch_dim, t_dim, -1)
                    # ----
                    # or we can randomize the repeat with the following logic
                    # 1. get uniform repeat number (assume joint_t len_t is 30 while visual_t is 6)
                    #    uniform sampling will be [5, 5, 5, 5, 5, 5]
                    num_uniform_repeat = [
                        joint_t.shape[1] // visual_t.shape[1]
                    ] * visual_t.shape[1]
                    # 2. choose which element needs to be changed, 0 to 6
                    #    then choose the index of the list
                    num_randomized = random.randint(
                        0, len(num_uniform_repeat)
                    )  # how many elements are randomized
                    rand_ids = random.choices(
                        range(len(num_uniform_repeat)), k=num_randomized
                    )
                    unique_ids = list(set(rand_ids))
                    # 3. do the randomization, offset by 1 or 2
                    for rand_id in unique_ids:
                        offset = random.randint(1, 2)
                        num_uniform_repeat[rand_id] -= offset
                        if rand_id == len(num_uniform_repeat) - 1:
                            num_uniform_repeat[rand_id - 1] += offset
                        else:
                            num_uniform_repeat[rand_id + 1] += offset
                    assert sum(num_uniform_repeat) == joint_t.shape[1]
                    # 4. do the repeat
                    visual_t_list = []
                    for i, num_repeat in enumerate(num_uniform_repeat):
                        v = visual_t[:, [i]].repeat(1, num_repeat, 1)
                        visual_t_list.append(v)
                    visual_t = torch.cat(visual_t_list, dim=1)
                info_list.append(visual_t)

            if self.contact_distillation:
                info_list.append(tactile_t)

            if self.with_noisy_obj_quat:
                # the observation is (xr,yr,zr,wr,xt,yt,zt) for legacy reasons
                info_list.append(
                    noisy_obj_quat[..., 4:]
                    if self.noisy_obj_quat_xyz_only
                    else noisy_obj_quat
                )
            if self.enable_palm_reskin:
                palm_reskin_t = palm_reskin_x.reshape(palm_reskin_x.shape[0], palm_reskin_x.shape[1], -1)
                info_list.append(palm_reskin_t)
            if self.enable_palm_binary:
                palm_binary_t = palm_binary_x.reshape(palm_binary_x.shape[0], palm_binary_x.shape[1], -1)
                info_list.append(palm_binary_t)
            merge_t_t = torch.cat(info_list, dim=-1)
            extrin_pred = self.adapt_tconv(merge_t_t)

        return extrin_pred

    def debias_object_xyz(
        self, joint_x: torch.Tensor, noisy_obj_quat: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat(
            [
                joint_x,
                noisy_obj_quat[..., -3:],
            ],
            dim=-1,
        )
        predicted_bias = self.xyz_debias(x.flatten(start_dim=1))
        obj_quat_debiased_xyz = noisy_obj_quat.clone()
        obj_quat_debiased_xyz[..., -3:] -= predicted_bias.unsqueeze(1)
        return obj_quat_debiased_xyz

    def point_cloud_embedding(self, point_cloud_info: torch.Tensor) -> torch.Tensor:
        if self.use_point_transformer:
            return self.point_mlp(point_cloud_info)
        else:
            pcs = self.point_mlp(point_cloud_info)
            return torch.max(pcs, 1)[0]

    def extrin_from_priv_info(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        extrin = self.env_mlp(obs_dict["priv_info"])
        if self.use_gt_point_cloud_info:
            pcs = self.point_cloud_embedding(obs_dict["point_cloud_info"])
            extrin = torch.cat([extrin, pcs], dim=-1)
        return torch.tanh(extrin)

    def extrin_from_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        extrin = self._privileged_pred(
            obs_dict["proprio_hist"],
            obs_dict["depth_buf"],
            obs_dict["fine_contact_info"],
            obs_dict.get("noisy_obj_quat", None),
            obs_dict.get("palm_reskin_info", None),
            obs_dict.get("palm_binary_info", None)
        )
        # if self.use_perceived_point_cloud_info:
        #     cloud, cloud_uncert = obs_dict["perceived_point_cloud_info"]
        #     pcs = self.point_cloud_embedding(cloud)
        #     if self.use_perceived_point_cloud_uncertainty:
        #         assert extrin.shape[1] == 40
        #         pc_latent = self.pc_modulator(
        #             extrin[:, 8:], pcs, cloud_uncert.unsqueeze(-1)
        #         )
        #         extrin = torch.cat([extrin[:, :8], pc_latent], dim=-1)
        #     else:
        #         assert extrin.shape[1] == 8
        #         extrin = torch.cat([extrin, pcs], dim=-1)

        return torch.tanh(extrin)


    def _actor_critic(self, obs_dict):
        obs = obs_dict["obs"]
        extrin = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.extrin_from_obs(obs_dict)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.extrin_from_priv_info(obs_dict)
                obs = torch.cat([obs, extrin], dim=-1)
                if self.multi_axis:
                    task_emb = self.task_mlp(obs_dict["rot_axis_buf"])
                    obs = torch.cat([obs, task_emb], dim=-1)

        x = self.actor_mlp(obs)
        critic_obs = (
            torch.cat([obs, obs_dict["critic_info"]], dim=-1)
            if self.asymm_actor_critic
            else x
        )
        
        value = self.value(critic_obs)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, sigma, value, extrin

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        mu, logstd, value, extrin = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            "prev_neglogp": torch.squeeze(prev_neglogp),
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "extrin": extrin,
        }
        return result
