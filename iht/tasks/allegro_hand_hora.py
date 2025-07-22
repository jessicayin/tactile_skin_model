# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# isort: off
from isaacgym import gymapi, gymtorch

# isort: on

import os
import pathlib
import pickle
import subprocess
from collections import OrderedDict
import glob
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import cv2
import numpy as np
import omegaconf
import open3d as o3d
import open3d.core as o3c
import torch
import torchvision
import trimesh
from isaacgym.torch_utils import (
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    tensor_clamp,
    to_torch,
    torch_rand_float,
)
from termcolor import cprint

from isaacgym import gymutil
import iht.utils.point_cloud as point_cloud_utils
import iht.utils.tensor as tensor_utils
from iht.utils.env import (
    data_root,
    get_object_subtype_info,
    priv_info_dict_from_config,
    robot_desc_root,
)
from iht.utils.metrics_logger import TensorLogger
from iht.utils.misc import get_rank, get_world_size, tprint
from isaacgym import gymutil
import datetime
from .base.vec_task import VecTask

def vulkan_device_id_from_cuda_device_id(orig: int) -> int:
    """Map a CUDA device index to a Vulkan one.

    Used to populate the value of `graphic_device_id`, which in IsaacGym is a vulkan
    device ID.

    This prevents a common segfault we get when the Vulkan ID, which is by default 0,
    points to a device that isn't present in CUDA_VISIBLE_DEVICES.
    """
    # Get UUID of the torch device.
    # All of the private methods can be dropped once this PR lands:
    #     https://github.com/pytorch/pytorch/pull/99967
    try:
        # orig = 0  # local only
        cuda_uuid = torch.cuda._raw_device_uuid_nvml()[
            torch.cuda._parse_visible_devices()[orig]
        ]  # type: ignore
        assert cuda_uuid.startswith("GPU-")
        cuda_uuid = cuda_uuid[4:]
    except AttributeError:
        print("detect cuda / vulkan relation can only be done for pytorch 2.0")
        return get_rank()

    try:
        vulkaninfo_lines = subprocess.run(
            ["vulkaninfo"],
            # We unset DISPLAY to avoid this error:
            # https://github.com/KhronosGroup/Vulkan-Tools/issues/370
            env={k: v for k, v in os.environ.items() if k != "DISPLAY"},
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).stdout.split("\n")
    except FileNotFoundError:
        print(
            "vulkaninfo was not found; try `apt install vulkan-tools` or `apt install vulkan-utils`."
        )
        return get_rank()

    vulkan_uuids = [
        s.partition("=")[2].strip()
        for s in vulkaninfo_lines
        if s.strip().startswith("deviceUUID")
    ]
    vulkan_uuids = list(dict(zip(vulkan_uuids, vulkan_uuids)).keys())
    vulkan_uuids = [uuid for uuid in vulkan_uuids if not uuid.startswith("0000")]
    out = vulkan_uuids.index(cuda_uuid)
    print(f"Using graphics_device_id={out}", cuda_uuid)
    return out


from typing import Any


def priv_info_dim_from_dict(priv_info_dict: Dict[str, Tuple[int, int]]) -> int:
    return max([v[1] for k, v in priv_info_dict.items()])


class AllegroHandHora(VecTask):
    def __init__(
        self, config, sim_device, graphics_device_id, headless, full_config=None
    ):
        self.config = config
        
        # before calling init in VecTask, need to do
        # 1. setup randomization
        self._setup_domain_rand_config(config["env"]["randomization"])
        # 2. setup privileged information
        self._setup_priv_option_config(config["env"]["privInfo"])
        # 3. setup object assets
        self._setup_object_info(config["env"]["object"])
        # 4. setup reward
        self._setup_reward_config(config["env"]["reward"])
        # 5. setup curriculum
        self._setup_curriculum(config["env"]["curriculum"])
        # 6. setup depth camera (visual policy training)
        # self._setup_visual_observation(config["env"]["rgbd_camera"])
        # unclassified config
        # self.obs_with_binary_contact = config["env"]["obs_with_binary_contact"]
        self.base_obj_scale = config["env"]["baseObjScale"]
        self.save_init_pose = config["env"]["genGrasps"]
        self.aggregate_mode = self.config["env"]["aggregateMode"]
        self.up_axis = "z"
        self.rotation_axis = config["env"]["rotation_axis"]
        self.hand_orientation = self.config["env"]["handOrientation"]
        self.reset_z_threshold = self.config["env"]["reset_height_threshold"]
        self.grasp_cache_name = self.config["env"]["grasp_cache_name"]
        self.canonical_pose_category = config["env"]["genGraspCategory"]
        self.num_pose_per_cache = "50k"
        self.disable_gravity_at_beginning = self.config["env"][
            "disable_gravity_at_beginning"
        ]
        self.enable_palm_reskin = config["env"]["enable_palm_reskin"]
        self.enable_palm_binary = config["env"]["enable_palm_binary"]
        #enable_palm_reskin is 3 axis forces, enable_palm_binary is binary contact. they should not both be true in stage2 training

        # Important: map CUDA device IDs to Vulkan ones.
        # graphics_device_id = vulkan_device_id_from_cuda_device_id(graphics_device_id)
        graphics_device_id = 0

        super().__init__(config, sim_device, graphics_device_id, headless)

        self.eval_done_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        self.debug_viz = self.config["env"]["enableDebugVis"]
        self.max_episode_length = self.config["env"]["episodeLength"]
        self.dt = self.sim_params.dt

        self.translation_direction = self.config["env"]["translation_direction"]
        self.obj_goal_pos = self._set_obj_goal_pos()
        
        
        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            env_look_id = self.config["env"]["camera_look_at_env"]
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.envs[env_look_id] if env_look_id is not None else None,
                cam_pos,
                cam_target,
            )

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.zeros(
            self.num_allegro_hand_dofs, dtype=torch.float, device=self.device
        )
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )
        self.force_sensors = gymtorch.wrap_tensor(force_sensor).view(
            self.num_envs, -1, 6
        )
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_allegro_hand_dofs
        ]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        # object apply random forces parameters
        self.force_scale = self.config["env"].get("forceScale", 0.0)
        self.random_force_prob_scalar = self.config["env"].get(
            "randomForceProbScalar", 0.0
        )
        self.force_decay = self.config["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.config["env"].get("forceDecayInterval", 0.08)
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device
        )
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )

        if self.randomize_scale and self.scale_list_init:
            self.saved_grasping_states = {}
            for s in self.randomize_scale_list:
                cache_dir = data_root / "cache"
                cache_name = "_".join(
                    [
                        self.grasp_cache_name,
                        "grasp",
                        self.canonical_pose_category,
                        self.num_pose_per_cache,
                        f's{str(s).replace(".", "")}',
                    ]
                )

                if not os.path.exists(cache_dir / f"{cache_name}.npy"):
                    cache_name = "/".join(
                        [
                            self.grasp_cache_name,
                            self.canonical_pose_category,
                            f's{str(s).replace(".", "")}_{self.num_pose_per_cache}',
                        ]
                    )
                self.saved_grasping_states[str(s)] = (
                    torch.from_numpy(np.load(cache_dir / f"{cache_name}.npy"))
                    .float()
                    .to(self.device)
                )

        self.rot_axis_buf = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.rot_axis_task = None
        if type(self.rotation_axis) == omegaconf.listconfig.ListConfig:
            axis_set = np.array(self.rotation_axis)
            if axis_set.ndim == 2:
                self.rot_axis_task = torch.from_numpy(axis_set)
                self.num_axis = self.rot_axis_task.shape[0]
                self.rot_axis_buf[:] = self.rot_axis_task[
                    torch.randint(0, self.num_axis, size=(self.num_envs,))
                ]
            else:
                axis = np.array(self.rotation_axis)
                axis = axis / np.linalg.norm(axis)
                self.rot_axis_buf[:] = to_torch(
                    axis, device=self.device, dtype=torch.float
                )[None]
        else:
            sign, axis = self.rotation_axis[0], self.rotation_axis[1]
            axis_index = ["x", "y", "z"].index(axis)
            self.rot_axis_buf[:, axis_index] = 1
            self.rot_axis_buf[:, axis_index] = (
                -self.rot_axis_buf[:, axis_index]
                if sign == "-"
                else self.rot_axis_buf[:, axis_index]
            )

        # useful buffers
        self.init_pose_buf = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
        )
        # there is an extra dim [self.control_freq_inv] because we want to get a mean over multiple control steps
        self.torques = torch.zeros(
            (self.num_envs, self.control_freq_inv, self.num_actions),
            device=self.device,
            dtype=torch.float,
        )
        self.dof_vel_finite_diff = torch.zeros(
            (self.num_envs, self.control_freq_inv, self.num_dofs),
            device=self.device,
            dtype=torch.float,
        )

        # --- calculate velocity at control frequency instead of simulated frequency
        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()
        self.ft_pos_prev = self.fingertip_pos.clone()
        self.ft_rot_prev = self.fingertip_orientation.clone()
        self.dof_vel_prev = self.dof_vel_finite_diff.clone()

        self.obj_linvel_at_cf = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.obj_angvel_at_cf = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.ft_linvel_at_cf = torch.zeros(
            (self.num_envs, 4 * 3), device=self.device, dtype=torch.float
        )
        self.ft_angvel_at_cf = torch.zeros(
            (self.num_envs, 4 * 3), device=self.device, dtype=torch.float
        )
        self.dof_acc = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float
        )
        # ----

        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [
            int,
            float,
        ], "assume p_gain and d_gain are only scalars"
        self.p_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.p_gain
        )
        self.d_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.d_gain
        )

        # debug and understanding statistics
        self.evaluate = self.config["on_evaluation"]
        self.eval_results_dir = pathlib.Path(self.config["eval_results_dir"])
        self.stat_sum_rewards = [
            0 for _ in self.object_subtype_list
        ]  # all episode reward
        self.stat_sum_episode_length = [
            0 for _ in self.object_subtype_list
        ]  # average episode length
        self.stat_sum_rotate_rewards = [
            0 for _ in self.object_subtype_list
        ]  # rotate reward, with clipping
        self.stat_sum_rotate_penalty = [
            0 for _ in self.object_subtype_list
        ]  # rotate penalty with clipping
        self.stat_sum_unclip_rotate_rewards = [
            0 for _ in self.object_subtype_list
        ]  # rotate reward, with clipping
        self.stat_sum_unclip_rotate_penalty = [
            0 for _ in self.object_subtype_list
        ]  # rotate penalty with clipping
        self.extrin_log = []
        self.env_evaluated = [0 for _ in self.object_subtype_list]
        self.evaluate_iter = 0

        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))


    def _set_obj_goal_pos(self):
        if self.translation_direction == "positive":
            direction = 1
        elif self.translation_direction == "negative":
            direction = -1
        elif self.translation_direction == "both":
            direction = 0
        else:
            print("invalid translation direction")
        if direction == 0:
            half_num_envs = self.num_envs // 2
            random_x_values_positive = np.random.uniform(0.03, 0.1, size=half_num_envs)
            random_x_values_negative = np.random.uniform(-0.1, -0.03, size=self.num_envs-half_num_envs)
            random_x_values = np.concatenate([random_x_values_positive, random_x_values_negative])
        else:
            self.x_min_range = 0.03 * direction
            self.x_max_range = 0.1 * direction
            random_x_values = np.random.uniform(self.x_min_range, self.x_max_range, size=self.num_envs)

        #set y and z value
        self.y_value = 0.02 
        self.z_value = 0.547
        obj_goal_pos = torch.tensor(np.array([random_x_values, [self.y_value] * self.num_envs, [self.z_value] * self.num_envs]), device=self.device).T
        return obj_goal_pos

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()
        allegro_hand_dof_props = self._parse_hand_dof_props()
        hand_pose, obj_pose = self._init_object_pose()

        #  ---------
        self.allegro_hand_start_pose = hand_pose
        self.allegro_pose = torch.tensor(
            [
                [
                    hand_pose.p.x,
                    hand_pose.p.y,
                    hand_pose.p.z,
                    hand_pose.r.x,
                    hand_pose.r.y,
                    hand_pose.r.z,
                    hand_pose.r.w,
                ]
            ]
        ).to(self.device)
        # ----------

        # compute aggregate size
        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(
            self.hand_asset
        )
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(
            self.hand_asset
        )
        max_agg_bodies = self.num_allegro_hand_bodies + 2
        max_agg_shapes = self.num_allegro_hand_shapes + 2

        self.envs = []
        # Used for record video during training, NOT FOR POLICY OBSERVATION
        self.vid_record_tensor = None
        # For visual policy training
        self.image_tensor, self.depth_tensor, self.seg_tensor = (
            [],
            [],
            [],
        )
        # self.camera_intrinsics = {
        #     "fx": [],
        #     "fy": [],
        #     "cx": torch.tensor([self.rgbd_buffer_width / 2], device=self.device),
        #     "cy": torch.tensor([self.rgbd_buffer_height / 2], device=self.device),
        # }
        # self.extrinsic_matrix, self.intrinsic_matrix = [], []
        # self.render_obj_point_clouds = []
        self.object_init_state = []

        self.hand_indices = []
        self.object_indices = []
        self.object_type_at_env = []

        self.contact_map = {}
        self.contact_map_min = np.inf
        self.contact_map_div = 0

        # self.obj_point_clouds = []  # from GT mesh for stage 1
        # obj_perceived_pcs_list: List[torch.Tensor] = []  # from perception for stage 2
        # obj_perceived_pcs_uncerts: List[float] = []
        allegro_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        object_rb_count = 1
        self.object_rb_handles = list(
            range(allegro_hand_rb_count, allegro_hand_rb_count + object_rb_count)
        )
        self.obj_scales = []

        for i in range(num_envs):
            tprint(f"{i + 1} / {num_envs}")
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True
                )

            # add hand - collision filter = -1
            # to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(
                env_ptr, self.hand_asset, hand_pose, "hand", i, -1, 1
            )
            
            self.gym.set_actor_dof_properties(
                env_ptr, hand_actor, allegro_hand_dof_props
            )
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)

            # set hand scale
            if self.randomize_hand_scale:
                hand_scale = np.random.uniform(0.95, 1.05)
                self.gym.set_actor_scale(env_ptr, hand_actor, hand_scale)
                self._update_priv_buf(env_id=i, name="hand_scale", value=hand_scale)
            
            # add object
            object_subtype = self.config["env"]["object"]["subtype"]
            if object_subtype is None:
                object_type_id = np.random.choice(
                    len(self.object_subtype_list), p=self.object_type_prob
                )
            else:
                object_type_id = self.object_subtype_list.index(object_subtype)

            self.object_type_at_env.append(object_type_id)
            object_asset = self.object_asset_list[object_type_id]

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, obj_pose, "object", i, 0, 2
            )

            # add palm reskin sensors

            if self.enable_palm_reskin or self.enable_palm_binary:
                self._create_reskin_sensors(env_ptr, i)

            self.object_init_state.append(
                [
                    obj_pose.p.x,
                    obj_pose.p.y,
                    obj_pose.p.z,
                    obj_pose.r.x,
                    obj_pose.r.y,
                    obj_pose.r.z,
                    obj_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)

            obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(
                    self.randomize_scale_list[i % num_scales] - 0.025,
                    self.randomize_scale_list[i % num_scales] + 0.025,
                )
    
            # if self.point_cloud_sampled_dim > 0:
            #     self.obj_point_clouds.append(
            #         self.asset_point_clouds[object_type_id] * obj_scale
            #     )
            # if self.fast_depth_simulation:
            #     self.render_obj_point_clouds.append(
            #         self.render_asset_point_clouds[object_type_id] * obj_scale
            #     )

            # if (
            #     self.perceived_point_cloud_sampled_dim > 0
            #     and not self.sim_point_cloud_registration
            # ):
            # if self.object_type == "rolling_pin":
            #     #NOTE: not sure if this is useful
            #     #assumes only using one noisy point cloud from sample_cylinder
            #     obj_perceived_pcs_list.append(self.perceived_point_clouds[object_type_id].to(device=self.device, dtype=torch.float)
            #         * obj_scale
            #     )
            #     obj_perceived_pcs_uncerts.append(
            #     self.perceived_pcs_uncertainty[object_type_id]
            #     )
            # else:
            #     obj_perceived_pcs_list.append(
            #         tensor_utils.random_choose(
            #             self.perceived_point_clouds[object_type_id]
            #         ).to(device=self.device, dtype=torch.float)
            #         * obj_scale
            #     )

            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self.obj_scales.append(obj_scale)
            self._update_priv_buf(env_id=i, name="obj_scale", value=obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [
                    np.random.uniform(
                        self.randomize_com_lower, self.randomize_com_upper
                    ),
                    np.random.uniform(
                        self.randomize_com_lower, self.randomize_com_upper
                    ),
                    np.random.uniform(
                        self.randomize_com_lower, self.randomize_com_upper
                    ),
                ]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name="obj_com", value=obj_com)

            obj_friction = 1.0
            obj_restitution = 0.0  # default is 0
            # TODO: bad engineering because of urgent modification
            if self.randomize_friction:
                rand_friction = np.random.uniform(
                    self.randomize_friction_lower, self.randomize_friction_upper
                )
                obj_restitution = np.random.uniform(0, 1)

                hand_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, hand_actor
                )
                for p in hand_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, hand_actor, hand_props
                )

                object_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, object_handle
                )
                for p in object_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, object_handle, object_props
                )
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name="obj_friction", value=obj_friction)
            self._update_priv_buf(
                env_id=i, name="obj_restitution", value=obj_restitution
            )

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(
                        self.randomize_mass_lower, self.randomize_mass_upper
                    )
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            self._update_priv_buf(env_id=i, name="obj_mass", value=prop[0].mass)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
        print()  # to correct the tprint the loop


        # self.obj_point_clouds = to_torch(
        #     np.array(self.obj_point_clouds), device=self.device, dtype=torch.float
        # )

        # if self.sim_point_cloud_registration:
        #     cprint(
        #         "Point cloud registration sim is still experimental and is "
        #         "not fully supported yet",
        #         "yellow",
        #         attrs=["bold"],
        #     )
        #     self.obj_perceived_point_clouds = torch.empty(
        #         (self.num_envs, self.perceived_point_cloud_sampled_dim, 4),
        #         device=self.device,
        #     )
        #     self.obj_perceived_pcs_uncertainty = None
        #     self.obj_perceived_pcs_pose = torch.eye(4, device=self.device).repeat(
        #         self.num_envs, 1, 1
        #     )
        # elif len(obj_perceived_pcs_list) > 0:
        #     self.obj_perceived_point_clouds = torch.stack(obj_perceived_pcs_list)
        #     self.obj_perceived_pcs_uncertainty = to_torch(
        #         obj_perceived_pcs_uncerts, device=self.device, dtype=torch.float
        #     )
        #     self.obj_perceived_pcs_pose = None
        # else:
        #     self.obj_perceived_point_clouds = None
        #     self.obj_perceived_pcs_uncertainty = None
        #     self.obj_perceived_pcs_pose = None

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(
            self.object_rb_handles, dtype=torch.long, device=self.device
        )
        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        self.object_type_at_env = to_torch(
            self.object_type_at_env, dtype=torch.long, device=self.device
        )
        # if self.fast_depth_simulation and self.enable_depth_camera:
        #     self.render_obj_point_clouds = to_torch(
        #         np.array(self.render_obj_point_clouds),
        #         device=self.device,
        #         dtype=torch.float,
        #     )
        #     self.extrinsic_matrix = torch.stack(self.extrinsic_matrix).to(
        #         self.rl_device
        #     )
        #     self.intrinsic_matrix = torch.stack(self.intrinsic_matrix).to(
        #         self.rl_device
        #     )
          



    def reset_idx(self, env_ids):
        # update gravity
        if self.grav_curriculum:
            # this is env steps instead of agent step
            env_steps = self.gym.get_frame_count(self.sim) * len(self.envs)
            agent_steps = env_steps // self.control_freq_inv
            agent_steps = max(agent_steps - self.grav_increase_start, 0)
            num_change = agent_steps // self.grav_increase_interval
            grav_new = self.grav_start + self.grav_increase_amount * num_change
            grav_new = max(grav_new, self.grav_end)
            prop = self.gym.get_sim_params(self.sim)
            grav_current = prop.gravity.z
            if abs(grav_current - grav_new) > 1e-3:
                print("------------------")
                print(f"set gravity to {grav_new:.2f}")
                print("------------------")
                prop.gravity.z = grav_new
                self.gym.set_sim_params(self.sim, prop)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower,
                self.randomize_p_gain_upper,
                (len(env_ids), self.num_actions),
                device=self.device,
            ).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower,
                self.randomize_d_gain_upper,
                (len(env_ids), self.num_actions),
                device=self.device,
            ).squeeze(1)

        # fingertip point clouds, sample at the beginning of each episode
        # if self.fingertip_point_cloud_sampled_dim > 0:
        #     self.fingertip_point_cloud_this_episode[
        #         env_ids, 0
        #     ] = self.fingertip_point_clouds[
        #         np.random.randint(200, size=self.fingertip_point_cloud_sampled_dim)
        #     ].clone()
        #     self.fingertip_point_cloud_this_episode[
        #         env_ids, 1
        #     ] = self.fingertip_point_clouds[
        #         np.random.randint(200, size=self.fingertip_point_cloud_sampled_dim)
        #     ].clone()
        #     self.fingertip_point_cloud_this_episode[
        #         env_ids, 2
        #     ] = self.fingertip_point_clouds[
        #         np.random.randint(200, size=self.fingertip_point_cloud_sampled_dim)
        #     ].clone()
        #     self.fingertip_point_cloud_this_episode[
        #         env_ids, 3
        #     ] = self.fingertip_point_clouds[
        #         np.random.randint(200, size=self.fingertip_point_cloud_sampled_dim)
        #     ].clone()

        self.random_obs_noise_e[env_ids] = torch.normal(
            0,
            self.random_obs_noise_e_scale,
            size=(len(env_ids), self.num_dofs),
            device=self.device,
            dtype=torch.float,
        )
        self.random_action_noise_e[env_ids] = torch.normal(
            0,
            self.random_action_noise_e_scale,
            size=(len(env_ids), self.num_dofs),
            device=self.device,
            dtype=torch.float,
        )

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[
                (env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)
            ]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            if self.saved_grasping_states[scale_key].ndim == 3:
                # per-object case
                obj_type_id_s = self.object_type_at_env[s_ids]
                grasping_state_scale_obj = self.saved_grasping_states[scale_key][
                    obj_type_id_s
                ]
                sampled_pose_idx = np.random.randint(
                    grasping_state_scale_obj.shape[1],
                    size=len(grasping_state_scale_obj),
                )
                sampled_pose = grasping_state_scale_obj[
                    np.arange(len(grasping_state_scale_obj)), sampled_pose_idx
                ].clone()
            else:
                # single object (category) case:
                sampled_pose_idx = np.random.randint(
                    self.saved_grasping_states[scale_key].shape[0], size=len(s_ids)
                )
                sampled_pose = self.saved_grasping_states[scale_key][
                    sampled_pose_idx
                ].clone()
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[
                :, 16:
            ]
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
            pos = sampled_pose[:, :16]
            self.allegro_hand_dof_pos[s_ids, :] = pos
            self.allegro_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, : self.num_allegro_hand_dofs] = pos
            self.cur_targets[s_ids, : self.num_allegro_hand_dofs] = pos
            self.init_pose_buf[s_ids, :] = pos
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        new_goal_pos = self._set_obj_goal_pos()

        self.obj_goal_pos[env_ids] = new_goal_pos[env_ids]

        penetration_thresh_low = 1 - self.penetration_distance_threshold_noise
        penetration_thresh_high = 1 + self.penetration_distance_threshold_noise
        penetration_thresh_size = torch.Size((env_ids.shape[0], self.obj_points.shape[0], 16)) #only generate random numbers for envs that are being reset

        new_noisy_penetration_distance_threshold = torch.ones((self.num_envs, self.obj_points.shape[0], 16), device=self.device)
        new_noisy_penetration_distance_threshold[env_ids] = (penetration_thresh_low + (penetration_thresh_high - penetration_thresh_low) * torch.rand(penetration_thresh_size, device=self.device)) * self.penetration_distance_threshold
        self.noisy_penetration_distance_threshold[env_ids] = new_noisy_penetration_distance_threshold[env_ids]

        
        #collect data for sim2real dataset
        # if self.config["collect_sim_data"]["save_data"]:
        #     self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        #     data_dir = self.config["collect_sim_data"]["save_data_dir"]
        #     self.data_path = data_dir + "/{}/".format(self.timestamp)
        #     print("saving this rollout to {folder}".format(folder = self.data_path))
        #     try:
        #         os.makedirs(self.data_path)
        #         os.mkdir(os.path.join(self.data_path, "sim_frames"))
        #     except:
        #         print("error creating save data folders")
        #         return
        #     self.traj_list = []
        #     self.reskin_forces_history = []
        #     self.actor_root_states_history = []
        #     self.reskin_signs_history = []
        #     self.correct_reskin_signs_history = []
        #     self.correct_reskin_forces_history = []
        #     self.allegro_dof_states_history = []
        #     self.avg_object_pos_max_list = []
        #     self.avg_object_pos_min_list = []
        #     self.reskin_rigid_contacts = []
        #     self.isaacgym_reskin_sensors = []
        #     self.rigid_body_states_history = []
        #     self.reskin_penetration_distances = []
        #     self.reskin_signs_history = []
        #     self.reskin_output_history = []
        #     self.track_edge_case = []
        #     self.traj_list = []
        #     self.allegro_dof_states_history = []
        #     self.actor_root_states_history = []
        
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices),
        )
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(hand_indices),
                len(env_ids),
            )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices),
            len(env_ids),
        )

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.noisy_obj_quat_buf[env_ids] = 0
        self.goal_quaternion_buf[env_ids] = 0
        self.dof_vel_finite_diff[:] = 0
        self.at_reset_buf[env_ids] = 1


    def _update_obj_pose_bufs(
        self, at_reset_env_ids: torch.Tensor, not_at_reset_envs_mask: torch.Tensor
    ) -> None:
        obj_pose_cfg = self.config["env"]["hora"]["object_pose"]

        # For environments that have just been reset, sample an object pose bias
        # to use for the whole episode
        if at_reset_env_ids.shape[0] > 0:

            def unif_min_v_to_v(v_):
                return v_ * (
                    2 * torch.rand_like(self.noisy_obj_quat_pos_bias[at_reset_env_ids])
                    - 1
                )

            self.noisy_obj_quat_pos_bias[at_reset_env_ids] = unif_min_v_to_v(
                self.noisy_obj_pos_bias_max
            )
            self.noisy_obj_quat_rpy_bias[at_reset_env_ids] = unif_min_v_to_v(
                self.noisy_obj_rpy_bias_max
            )

        # Apply noise to object pose
        noisy_obj_quat = tensor_utils.noisy_quat_from_pos_rot(
            self.object_pos,
            self.object_rot,
            noise_rpy_bias=self.noisy_obj_quat_rpy_bias,
            noise_pos_bias=self.noisy_obj_quat_pos_bias,
            noise_rpy_scale=self.noisy_obj_rpy_scale,
            noise_pos_scale=self.noisy_obj_pos_scale,
        ).unsqueeze(1)

        # Compute the mask of poses that will be repeated due to slower perception rate
        # or due to random drops
        slow_mask = self._compute_obj_pose_slow_mask(
            obj_pose_cfg["base_slowness"],
            obj_pose_cfg["drop_rate"],
            not_at_reset_envs_mask,
        )
        # Update the buffer that has the noisy object pose history
        self._update_obj_pose_buf_helper(
            noisy_obj_quat,
            self.noisy_obj_quat_buf,
            at_reset_env_ids,
            slow_mask,
            obj_pose_cfg["to_hand_frame"],
        )
        # Update also the auxiliary ground-truth object pose history buffer
        self._update_obj_pose_buf_helper(
            torch.cat([self.object_rot, self.object_pos], dim=-1).unsqueeze(1),
            self.obj_quat_gt_buf,
            at_reset_env_ids,
            slow_mask,
            obj_pose_cfg["to_hand_frame"],
        )

    def _compute_obj_pose_slow_mask(
        self,
        slowness: int,
        obj_pose_drop_rate: float,
        not_at_reset_envs_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        slow_mask: torch.Tensor = None
        # Now environments that are in the middle of an episode
        if not_at_reset_envs_mask.any():
            if self.noisy_obj_pose_slowness_cnt % slowness != 0:
                # If not at the next (slower) step, the new obj pose is the same
                # as before
                slow_mask = not_at_reset_envs_mask
            else:
                # Even if it's supposed to be updated with the current pose
                # Add some small probability of failure (note that this means
                # it's not updated at least until the next step of the slow cycle)
                slow_mask = not_at_reset_envs_mask & (
                    torch.rand(
                        not_at_reset_envs_mask.shape[0],
                        device=not_at_reset_envs_mask.device,
                    )
                    < obj_pose_drop_rate
                )
        self.noisy_obj_pose_slowness_cnt += 1
        return slow_mask

    def _update_obj_pose_buf_helper(
        self,
        source_pose_tensor: torch.Tensor,
        target_pose_buffer: torch.Tensor,
        at_reset_env_ids: torch.Tensor,
        slow_mask: Optional[torch.Tensor],
        to_hand_frame: bool,
    ) -> None:
        if to_hand_frame:
            source_pose_tensor = tensor_utils.from_world_frame(
                source_pose_tensor[..., 4:].permute(1, 0, 2),
                self.allegro_pose[:, 3:],
                self.allegro_pose[:, :3],
                quaternions=source_pose_tensor[..., :4].permute(1, 0, 2),
                quat_first=True,
            ).permute(1, 0, 2)
        else:
            # to avoid in-place modification below of external tensor
            source_pose_tensor = source_pose_tensor.clone()

        # Environments that were reset start with known object pose
        target_pose_buffer[at_reset_env_ids] = source_pose_tensor[at_reset_env_ids]

        # For non reset environments, the buffer is [old_pose_buffer_0:t-1, obj_pose_t]
        # But first, handle object pose repetitions as indicated in the slow mask
        if slow_mask is not None and slow_mask.any():
            source_pose_tensor[slow_mask, 0] = target_pose_buffer[slow_mask, -1]
        target_pose_buffer[:] = torch.cat(
            [target_pose_buffer[:, 1:].clone(), source_pose_tensor],
            dim=1,
        )

    def _get_reward_scale_by_name(self, name):
        env_steps = self.gym.get_frame_count(self.sim) * len(self.envs)
        agent_steps = env_steps // self.control_freq_inv
        agent_steps *= get_world_size()
        init_scale, final_scale, curr_start, curr_end = self.reward_scale_dict[name]
        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            # discretize to [0, 0.05, 1.0] instead of continuous value
            # during batch collection, avoid reward confusion
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1
        if self.evaluate:
            curr_progress = 1
        return init_scale + (final_scale - init_scale) * curr_progress

   
    def _show_iht_debug_viz(self):
        # source: https://github.com/carolinahiguera/NCF-Policy/blob/5a2c58ca1bc42cc666688a1ed08b35cc25df94fa/NCF_policies/isaacgymenvs/tasks/ncf_tacto/factory_task_bowl_dishrack.py#L566
        # visualizes goal position and object center to qualitatively evaluate iht policy
        sphere_pose = gymapi.Transform()
        sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
        epsilon = 0.0005

        # import pdb; pdb.set_trace()
        for i in range(self.num_envs):
            obj_sphere_transform = gymapi.Transform()
            obj_sphere_transform.p = gymapi.Vec3(*[float(self.object_pos[i][0]), float(self.object_pos[i][1]), float(self.object_pos[i][2])])
            obj_sphere_transform.r = gymapi.Quat(*[float(self.object_rot[i][0]), float(self.object_rot[i][1]), float(self.object_rot[i][2]), float(self.object_rot[i][3])])

            if ((self.object_pos[i] - self.obj_goal_pos[i])**2).sum(-1) < epsilon:
                obj_sphere = gymutil.WireframeSphereGeometry(0.05, 8, 8, sphere_pose, color = (0, 1, 0))
            else:
                obj_sphere = gymutil.WireframeSphereGeometry(0.05, 8, 8, sphere_pose, color = (0, 0, 0)) #black
            gymutil.draw_lines(
                obj_sphere, self.gym, self.viewer, self.envs[i], obj_sphere_transform
            )
        
        for i in range(self.num_envs):
            goal_sphere_transform = gymapi.Transform()
            goal_sphere_transform.p = gymapi.Vec3(*[float(self.obj_goal_pos[i][0]), float(self.obj_goal_pos[i][1]), float(self.obj_goal_pos[i][2])])
            goal_sphere_transform.r = gymapi.Quat(0, 0, 0, 1)
            if ((self.object_pos[i] - self.obj_goal_pos[i])**2).sum(-1) < epsilon:
                goal_sphere = gymutil.WireframeSphereGeometry(0.05, 8, 8, sphere_pose, color = (0, 1, 0))
            else:
                goal_sphere = gymutil.WireframeSphereGeometry(0.05, 8, 8, sphere_pose, color=(1, 1, 0)) #yellow
            gymutil.draw_lines(
                goal_sphere, self.gym, self.viewer, self.envs[i], goal_sphere_transform
            )


    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        # cur* but need for reward is here
        self.compute_reward()

        # calibration
        # self.reset_buf[:] = 0
        
        #find indices of non-zero elements in reset buffer, which corresponds to environments that need to be reset and return as tensor
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        # if self.metrics_logger is not None: #iht does not use metrics logger
        #     # no data, just sync all updates done this step
        #     self.metrics_logger.update({}, sync=True)

        self.debug_viz = True
        if self.viewer and self.object_type in ["rolling_pin", "pool_noodle", "65mm_cylin"]:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # self._show_iht_debug_viz()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            # self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.prev_targets + self.action_scale * self.actions
        save_targets = targets.cpu().detach().numpy()  # Convert the tensor to a NumPy array
        # if self.config["collect_sim_data"]["save_data"]:
        #     self.traj_list.append(save_targets)
        # print(save_targets)
        self.cur_targets[:] = tensor_clamp(
            targets,
            self.allegro_hand_dof_lower_limits,
            self.allegro_hand_dof_upper_limits,
        )
        # get prev* buffer here
        self.prev_targets[:] = self.cur_targets
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.ft_rot_prev[:] = self.fingertip_orientation
        self.ft_pos_prev[:] = self.fingertip_pos
        self.dof_vel_prev[:] = self.dof_vel_finite_diff

    def _fill_obs_dict_from_bufs(self) -> Dict[str, torch.Tensor]:
        # stage 1 buffers
        self.obs_dict["priv_info"] = self.priv_info_buf.to(self.rl_device)
        # self.obs_dict["point_cloud_info"] = self.point_cloud_buf.to(self.rl_device)
        # observation buffer for critic
        self.obs_dict["critic_info"] = self.critic_info_buf.to(self.rl_device)
        self.obs_dict["rot_axis_buf"] = self.rot_axis_buf.to(self.rl_device)
        # stage 2 buffers
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict["noisy_obj_quaternion"] = self.noisy_obj_quat_buf.to(
            self.rl_device
        )
        self.obs_dict["obj_quaternion_gt"] = self.obj_quat_gt_buf.to(self.rl_device)
        self.obs_dict["goal_quaternion"] = self.goal_quaternion_buf.to(self.rl_device)
        # if self.enable_depth_camera:
        #     self.obs_dict["depth_buffer"] = self.depth_buf.to(self.rl_device)
        # if self.perceived_point_cloud_sampled_dim > 0:
        #     self.obs_dict[
        #         "perceived_point_cloud_info"
        #     ] = self.perceived_point_cloud_buf.to(self.rl_device)
    
        if self.enable_palm_reskin:
            self.obs_dict["palm_reskin_info"] = self.palm_reskin_buffer.to(self.rl_device)
        if self.enable_palm_binary:
            self.obs_dict["palm_binary_info"] = self.palm_binary_buffer.to(self.rl_device)

    def reset(self):
        super().reset()
        self._fill_obs_dict_from_bufs()
        return self.obs_dict

    def step(self, actions, extrin_record: Optional[torch.Tensor] = None):
        # Save extrinsics if evaluating on just one object.
        if (
            extrin_record is not None
            and self.config["env"]["object"]["subtype"] is not None
        ):
            # Put a (z vectors, is done) tuple into the log.
            self.extrin_log.append(
                (
                    extrin_record.detach().cpu().numpy().copy(),
                    self.eval_done_buf.detach().cpu().numpy().copy(),
                )
            )

        super().step(actions)
        self._fill_obs_dict_from_bufs()
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras


    def update_low_level_control(self, step_id):
        previous_dof_pos = self.allegro_hand_dof_pos.clone()
        self._refresh_gym()
        random_action_noise_t = torch.normal(
            0,
            self.random_action_noise_t_scale,
            size=self.allegro_hand_dof_pos.shape,
            device=self.device,
            dtype=torch.float,
        )
        noise_action = (
            self.cur_targets + self.random_action_noise_e + random_action_noise_t
        )
        if self.torque_control:
            dof_pos = self.allegro_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff[:, step_id] = dof_vel.clone()
            torques = self.p_gain * (noise_action - dof_pos) - self.d_gain * dof_vel
            torques = torch.clip(torques, -self.torque_limit, self.torque_limit).clone()
            self.torques[:, step_id] = torques
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torques)
            )
        else:
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(noise_action)
            )

    def update_rigid_body_force(self):
        if self.force_scale > 0.0 or self.disable_gravity_at_beginning:
            self.rb_forces *= torch.pow(
                self.force_decay, self.dt / self.force_decay_interval
            )
            # apply new forces
            obj_mass = [
                self.gym.get_actor_rigid_body_properties(
                    env, self.gym.find_actor_handle(env, "object")
                )[0].mass
                for env in self.envs
            ]
            obj_mass = to_torch(obj_mass, device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (
                torch.less(torch.rand(self.num_envs, device=self.device), prob)
            ).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(
                    self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                    device=self.device,
                )
                * obj_mass[force_indices, None]
                * self.force_scale
            )
            # first 0.5s (10 timesteps in 20Hz control frequency) is without gravity
            if self.disable_gravity_at_beginning:
                disable_indices = torch.less(self.progress_buf, 10)
                self.rb_forces[disable_indices, self.object_rb_handles, :2] = 0
                prop = self.gym.get_sim_params(self.sim)
                grav_current = prop.gravity.z
                self.rb_forces[disable_indices, self.object_rb_handles, 2] = (
                    -obj_mass[disable_indices] * grav_current
                )
                # reset rb force at the end
                reset_grav_indices = self.progress_buf == 10
                self.rb_forces[reset_grav_indices, self.object_rb_handles, :] = 0

            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE
            )

    def check_termination(self, object_pos):
        term_by_max_eps = torch.greater_equal(
            self.progress_buf, self.max_episode_length
        )
        if self.hand_orientation == "up":
            reset_z = torch.less(object_pos[:, -1], self.reset_z_threshold)
            resets = reset_z
            resets = torch.logical_or(resets, term_by_max_eps)
        else:
            raise NotImplementedError
        return resets

       
    def get_time(self):
        sim_time = self.gym.get_sim_time(self.sim)
        return sim_time

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        self.fingertip_states = self.rigid_body_states[:, self.fingertip_handles]
        self.fingertip_pos = self.fingertip_states[:, :, :3].reshape(self.num_envs, -1)
        self.fingertip_orientation = self.fingertip_states[:, :, 3:7].reshape(
            self.num_envs, -1
        )
        self.fingertip_linvel = self.fingertip_states[:, :, 7:10].reshape(
            self.num_envs, -1
        )
        self.fingertip_angvel = self.fingertip_states[:, :, 10:13].reshape(
            self.num_envs, -1
        )

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config["randomizeMass"]
        self.randomize_mass_lower = rand_config["randomizeMassLower"]
        self.randomize_mass_upper = rand_config["randomizeMassUpper"]
        self.randomize_com = rand_config["randomizeCOM"]
        self.randomize_com_lower = rand_config["randomizeCOMLower"]
        self.randomize_com_upper = rand_config["randomizeCOMUpper"]
        self.randomize_friction = rand_config["randomizeFriction"]
        self.randomize_friction_lower = rand_config["randomizeFrictionLower"]
        self.randomize_friction_upper = rand_config["randomizeFrictionUpper"]
        self.randomize_scale = rand_config["randomizeScale"]
        self.randomize_hand_scale = rand_config["randomize_hand_scale"]
        self.scale_list_init = rand_config["scaleListInit"]
        self.randomize_scale_list = rand_config["randomizeScaleList"]
        self.randomize_scale_lower = rand_config["randomizeScaleLower"]
        self.randomize_scale_upper = rand_config["randomizeScaleUpper"]
        self.randomize_pd_gains = rand_config["randomizePDGains"]
        self.randomize_p_gain_lower = rand_config["randomizePGainLower"]
        self.randomize_p_gain_upper = rand_config["randomizePGainUpper"]
        self.randomize_d_gain_lower = rand_config["randomizeDGainLower"]
        self.randomize_d_gain_upper = rand_config["randomizeDGainUpper"]
        self.random_obs_noise_e_scale = rand_config["obs_noise_e_scale"]
        self.random_obs_noise_t_scale = rand_config["obs_noise_t_scale"]
        self.random_action_noise_e_scale = rand_config["action_noise_e_scale"]
        self.random_action_noise_t_scale = rand_config["action_noise_t_scale"]
        # stage 2 specific
        self.noisy_obj_rpy_scale = rand_config["noisy_obj_rpy_scale"]
        self.noisy_obj_pos_scale = rand_config["noisy_obj_pos_scale"]
        self.noisy_obj_rpy_bias_max = rand_config["noisy_obj_rpy_bias"]
        self.noisy_obj_pos_bias_max = rand_config["noisy_obj_pos_bias"]

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config["enableObjPos"]
        self.enable_priv_obj_mass = p_config["enableObjMass"]
        self.enable_priv_obj_scale = p_config["enableObjScale"]
        self.enable_priv_obj_com = p_config["enableObjCOM"]
        self.enable_priv_obj_friction = p_config["enableObjFriction"]
        self.enable_priv_net_contact = p_config["enableNetContactF"]
        self.contact_input_dim = p_config["contact_input_dim"]
        self.contact_form = p_config["contact_form"]
        self.contact_input = p_config["contact_input"]
        self.contact_binarize_threshold = p_config["contact_binarize_threshold"]
        self.enable_priv_obj_orientation = p_config["enable_obj_orientation"]
        self.enable_priv_obj_linvel = p_config["enable_obj_linvel"]
        self.enable_priv_obj_angvel = p_config["enable_obj_angvel"]
        self.enable_priv_fingertip_position = p_config["enable_ft_pos"]
        self.enable_priv_fingertip_orientation = p_config["enable_ft_orientation"]
        self.enable_priv_fingertip_linvel = p_config["enable_ft_linvel"]
        self.enable_priv_fingertip_angvel = p_config["enable_ft_angvel"]
        self.enable_priv_hand_scale = p_config["enable_hand_scale"]
        self.enable_priv_obj_restitution = p_config["enable_obj_restitution"]
        self.priv_info_dict = priv_info_dict_from_config(p_config)

    def _update_priv_buf(self, env_id, name, value):
        # normalize to -1, 1
        if eval(f"self.enable_priv_{name}"):
            s, e = self.priv_info_dict[name]
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            self.priv_info_buf[env_id, s:e] = value

    def _setup_object_info(self, o_config):
        self.object_type = o_config["type"]
        (
            self.object_subtype_list,
            self.object_type_prob,
            self.asset_files_dict,
        ) = get_object_subtype_info(
            data_root,
            o_config["type"],
            o_config["sampleProb"],
            max_num_objects=self.config["env"]["object"].get("max_num", None),
        )
        primitive_list = self.object_type.split("+")
        print("---- Primitive List ----")
        print(primitive_list)
        print("---- Object List ----")
        print(f"using {len(self.object_subtype_list)} training objects")
        assert len(self.object_subtype_list) == len(self.object_type_prob)

    def _setup_curriculum(self, c_config):
        self.g_c_config = c_config["gravity"]
        self.grav_curriculum = self.g_c_config["enable"]
        self.grav_start = self.g_c_config["start"]
        self.grav_end = self.g_c_config["end"]
        self.grav_increase_start = self.g_c_config["increase_start"] * 1e6
        self.grav_increase_interval = self.g_c_config["increase_interval"] * 1e6
        self.grav_increase_amount = self.g_c_config["increase_amount"]

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.priv_info_dim = priv_info_dim_from_dict(self.priv_info_dict)
        hora_config = self.config["env"]["hora"]
        self.prop_hist_len = hora_config["propHistoryLen"]
        self.critic_obs_dim = hora_config["critic_obs_dim"]
        # self.point_cloud_sampled_dim = hora_config["point_cloud_sampled_dim"]
        # self.fingertip_point_cloud_sampled_dim = hora_config[
        #     "fingertip_point_cloud_sampled_dim"
        # ]
        # self.perceived_point_cloud_sampled_dim = hora_config[
        #     "perceived_point_cloud_sampled_dim"
        # ]
        # self.sim_point_cloud_registration = hora_config["sim_point_cloud_registration"]
        # if self.sim_point_cloud_registration:
        #     self.sim_point_cloud_registration_new_samples = int(
        #         hora_config["sim_point_cloud_registration_ratio"]
        #         * self.perceived_point_cloud_sampled_dim
        #     )
        # self.point_cloud_buffer_dim = (
        #     self.point_cloud_sampled_dim + 4 * self.fingertip_point_cloud_sampled_dim
        # )
        # self.perceived_point_cloud_buffer_dim = (
        #     self.perceived_point_cloud_sampled_dim
        #     + 4 * self.fingertip_point_cloud_sampled_dim
        # )
        self.priv_info_buf = torch.zeros(
            (num_envs, self.priv_info_dim), device=self.device, dtype=torch.float
        )
        self.critic_info_buf = torch.zeros(
            (num_envs, self.critic_obs_dim), device=self.device, dtype=torch.float
        )
        # self.fingertip_point_cloud_this_episode = torch.zeros(
        #     (num_envs, 4, self.fingertip_point_cloud_sampled_dim, 3),
        #     device=self.device,
        #     dtype=torch.float,
        # )
        # self.point_cloud_buf = torch.zeros(
        #     num_envs,
        #     self.point_cloud_buffer_dim,
        #     3,
        #     device=self.device,
        #     dtype=torch.float,
        # )
        # self.perceived_point_cloud_buf = torch.zeros(
        #     num_envs,
        #     self.perceived_point_cloud_buffer_dim,
        #     3,
        #     device=self.device,
        #     dtype=torch.float,
        # )
        # fixed noise per-episode, for different hardware have different this value
        self.random_obs_noise_e = torch.zeros(
            (num_envs, self.config["env"]["numActions"]),
            device=self.device,
            dtype=torch.float,
        )
        self.random_action_noise_e = torch.zeros(
            (num_envs, self.config["env"]["numActions"]),
            device=self.device,
            dtype=torch.float,
        )

        # ---- stage 2 buffers
        # stage 2 related buffers
        if self.enable_palm_reskin:
            self.palm_reskin_buffer = torch.zeros((num_envs, self.prop_hist_len, 16, 3)).to(self.device)
        else:
            self.palm_reskin_buffer = None
        
        if self.enable_palm_binary:
            self.palm_binary_buffer = torch.zeros((num_envs, self.prop_hist_len, 16, 3)).to(self.device)
        else:
            self.palm_binary_buffer = None

        self.proprio_hist_buf = torch.zeros(
            (
                num_envs,
                self.prop_hist_len,
                32 
                # if not self.obs_with_binary_contact else 36,
            ),
            device=self.device,
            dtype=torch.float,
        )
        # a bit unintuitive: first 4 is quaternion and last 3 is position, due to development order
        self.noisy_obj_quat_buf = torch.zeros(
            (num_envs, self.prop_hist_len, 7), device=self.device, dtype=torch.float
        )
        self.obj_quat_gt_buf = torch.zeros_like(self.noisy_obj_quat_buf)
        self.noisy_obj_pose_slowness_cnt = 0
        self.goal_quaternion_buf = torch.zeros(
            (num_envs, self.prop_hist_len, 8), device=self.device, dtype=torch.float
        )
        # noise bias that gets sampled everytime the env resets
        self.noisy_obj_quat_pos_bias = self.noisy_obj_quat_buf.new_zeros(num_envs, 3)
        self.noisy_obj_quat_rpy_bias = self.noisy_obj_quat_buf.new_zeros(num_envs, 3)
        # buffer for stage 2 vision policy distillation
        # if self.enable_depth_camera:
        #     # there are several options
        #     # 1. ground-truth depth images: verification of concept
        #     # 2. noisy depth images: try to reduce the gap
        #     # 3. randomly sampled point-cloud
        #     # future techniques can also be improved
        #     self.depth_slowness = self.config["env"]["hora"]["render_slowness"]
        #     self.depth_buf = torch.zeros(
        #         (
        #             num_envs,
        #             self.prop_hist_len // self.depth_slowness,
        #             2,
        #             self.rgbd_buffer_height,
        #             self.rgbd_buffer_width,
        #         ),
        #         device=self.device,
        #         dtype=torch.float,
        #     )
        #     self.random_depth_noise_e = torch.zeros(
        #         (num_envs, self.rgbd_buffer_height, self.rgbd_buffer_width),
        #         device=self.device,
        #         dtype=torch.float,
        #     )
        #     if self.fast_depth_simulation:
        #         self.random_depth_noise_e = self.random_depth_noise_e.to(self.rl_device)
        #         self.dense_point_cloud_buf = torch.zeros(
        #             num_envs,
        #             self.num_dense_points_object,
        #             3,
        #             device=self.rl_device,
        #             dtype=torch.float,
        #         )
        #         self.dense_fingertip_cloud_buf = torch.zeros(
        #             num_envs,
        #             4,
        #             self.num_dense_points_finger,
        #             3,
        #             device=self.rl_device,
        #             dtype=torch.float,
        #         )
        # else:
        #     self.depth_buf = None

    def _setup_reward_config(self, r_config):
        # the list
        self.reward_scale_dict = {}
        for k, v in r_config.items():
            if "scale" in k:
                if type(v) is not omegaconf.listconfig.ListConfig:
                    v = [v, v, 0, 0]
                else:
                    assert len(v) == 4
                self.reward_scale_dict[k.replace("_scale", "")] = v
        self.angvel_clip_min = r_config["angvelClipMin"]
        self.angvel_clip_max = r_config["angvelClipMax"]
        self.perp_angvel_clip_min = r_config["perp_angvel_clip_min"]
        self.perp_angvel_clip_max = r_config["perp_angvel_clip_max"]

    def _parse_hand_dof_props(self):
        # set allegro_hand dof properties
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []

        for i in range(self.num_allegro_hand_dofs):
            # another option, just do it for now, parse directly from Nvidia's Calibrated Value
            # avoid frequently or adding another URDF
            # allegro_hand_dof_lower_limits = [
            #     -0.5585,
            #     -0.27924,
            #     -0.27924,
            #     -0.27924,
            #     0.27924,
            #     -0.331603,
            #     -0.27924,
            #     -0.27924,
            #     -0.5585,
            #     -0.27924,
            #     -0.27924,
            #     -0.27924,
            #     -0.5585,
            #     -0.27924,
            #     -0.27924,
            #     -0.27924,
            # ]
            # allegro_hand_dof_upper_limits = [
            #     0.5585,
            #     1.727825,
            #     1.727825,
            #     1.727825,
            #     1.57075,
            #     1.1518833,
            #     1.727825,
            #     1.76273055,
            #     0.5585,
            #     1.727825,
            #     1.727825,
            #     1.727825,
            #     0.5585,
            #     1.727825,
            #     1.727825,
            #     1.727825,
            # ]
            # allegro_hand_dof_props["lower"][i] = allegro_hand_dof_lower_limits[i]
            # allegro_hand_dof_props["upper"][i] = allegro_hand_dof_upper_limits[i]

            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props["lower"][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props["upper"][i])
            allegro_hand_dof_props["effort"][i] = self.torque_limit
            if self.torque_control:
                allegro_hand_dof_props["stiffness"][i] = 0.0
                allegro_hand_dof_props["damping"][i] = 0.0
                allegro_hand_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            else:
                allegro_hand_dof_props["stiffness"][i] = self.config["env"][
                    "controller"
                ]["pgain"]
                allegro_hand_dof_props["damping"][i] = self.config["env"]["controller"][
                    "dgain"
                ]
            allegro_hand_dof_props["friction"][i] = 0.01
            allegro_hand_dof_props["armature"][i] = 0.001

        self.allegro_hand_dof_lower_limits = to_torch(
            self.allegro_hand_dof_lower_limits, device=self.device
        )
        self.allegro_hand_dof_upper_limits = to_torch(
            self.allegro_hand_dof_upper_limits, device=self.device
        )
        return allegro_hand_dof_props

    def _init_object_pose(self):
        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        angle_rad = np.deg2rad(120)
        
        if self.hand_orientation == "up":
            allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0), -np.pi / 2
            ) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        elif self.hand_orientation == "down":
            allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 1, 0), np.pi / 2
            ) * gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi / 2)
        else:
            raise NotImplementedError
        # allegro_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi/2) * gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), angle_rad) #tilt 30deg against gravity
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_hand_start_pose.p.x
        pose_dx, pose_dy, pose_dz = -0.01, -0.04, 0.15

        object_start_pose.p.x = allegro_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = allegro_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_hand_start_pose.p.z + pose_dz

        object_start_pose.p.y = allegro_hand_start_pose.p.y - 0.01
        # TODO: this weird thing is an unknown issue
        # object_start_pose.r.x = allegro_hand_start_pose.r.x
        # object_start_pose.r.y = allegro_hand_start_pose.r.y
        # object_start_pose.r.z = allegro_hand_start_pose.r.z
        # object_start_pose.r.w = allegro_hand_start_pose.r.w

        if self.save_init_pose:
            object_start_pose.p.z = self.reset_z_threshold + 0.015
        else:
            if self.hand_orientation == "up":
                object_start_pose.p.z = self.reset_z_threshold + 0.005
            elif self.hand_orientation == "down":
                object_start_pose.p.z = 0.35
            else:
                raise NotImplementedError
        return allegro_hand_start_pose, object_start_pose


    def _get_contact_force(self):
        # a wrapper for different contact force option
        # option 1: force sensor / net contact force / rigid contact lambda
        if self.contact_form == "force_sensor":
            # raw force sensor is (num_env, num_sensor, 6 [force / torque])
            contacts = self.force_sensors[..., :3]
            if self.contact_input == "binary":
                contacts = (
                    (contacts**2).sqrt().sum(-1) > self.contact_binarize_threshold
                ).float()
        elif self.contact_form == "gpu_contact":
            # raw contact force is (num_envs, num_body, 3)
            contacts = self.contact_forces[:, self.contact_handles]
            if self.contact_input == "binary":
                contacts = (
                    (contacts**2).sqrt().sum(-1) > self.contact_binarize_threshold
                ).float()  # (num_envs, k x 4)
            elif self.contact_input == "fingertip_dof": #this is to calculate force variance
                _forces = self.gym.acquire_dof_force_tensor(self.sim)
                forces = gymtorch.wrap_tensor(_forces).reshape(self.num_envs, 16) #16: number of dofs
                joints0_3 = [0, 1, 2, 3]
                joints12_15 = [4, 5, 6, 7]
                joints4_7 = [8, 9, 10, 11]
                joints8_11 = [12, 13, 14, 15]
                #sum the forces on the finger
                self.fingertip_forces = torch.abs(torch.stack([forces[:, joints0_3].sum(-1), forces[:, joints12_15].sum(-1), forces[:, joints4_7].sum(-1), forces[:, joints8_11].sum(-1)], dim=1))
                contacts = self.fingertip_forces #(num_envs, 4)
            assert contacts.shape[1] == self.contact_input_dim
        elif self.contact_form == "cpu_contact":
            contacts = self.force_sensors[..., :3]
            # contacts = self.contact_forces[:, self.contact_handles] #fingertip contacts only
            if self.contact_input != "fine":
                return contacts
        else:
            raise NotImplementedError
        return contacts


def compute_hand_reward(
    object_linvel_penalty,
    object_linvel_penalty_scale: float,
    rotate_reward,
    rotate_reward_scale: float,
    pose_diff_penalty,
    pose_diff_penalty_scale: float,
    torque_penalty,
    torque_pscale: float,
    work_penalty,
    work_pscale: float,
    dof_vel_penalty,
    dof_vel_penalty_scale: float,
    undesired_rotation_penalty,
    undesired_rotation_penalty_scale: float,
    finger_base_penalty,
    finger_base_penalty_scale: float,
    z_dist_penalty,
    z_dist_penalty_scale: float,
    dof_acc_penalty,
    dof_acc_penalty_scale: float,
    position_penalty,
    position_penalty_scale: float,
    finger_obj_penalty,
    finger_obj_penalty_scale: float,
    contact_force_reward,
    contact_force_reward_scale: float,
):
    reward = rotate_reward_scale * rotate_reward
    reward = reward + object_linvel_penalty * object_linvel_penalty_scale
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + dof_vel_penalty * dof_vel_penalty_scale
    reward = reward + work_penalty * work_pscale
    reward = reward + undesired_rotation_penalty * undesired_rotation_penalty_scale
    reward = reward + finger_base_penalty * finger_base_penalty_scale
    reward = reward + z_dist_penalty * z_dist_penalty_scale
    reward = reward + dof_acc_penalty * dof_acc_penalty_scale
    reward = reward + position_penalty * position_penalty_scale
    reward = reward + finger_obj_penalty * finger_obj_penalty_scale
    reward = reward + contact_force_reward * contact_force_reward_scale
    # reward = reward + fast_rotation_penalty * fast_rotation_penalty_scale
    return reward
