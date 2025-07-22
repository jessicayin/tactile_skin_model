# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import pickle

import numpy as np

from isaacgym import gymtorch
from isaacgym.torch_utils import (
    to_torch,
    torch_rand_float,
)
import torch
from iht.tasks.allegro_hand_hora_iht import AllegroHandHoraIHT

class AllegroHandGraspIHT(AllegroHandHoraIHT):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        super().__init__(
            config,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
        )
        self.saved_grasping_states = torch.zeros(
                (0, 23), dtype=torch.float, device=self.device
            )
        self.canonical_pose_category = "iht_larger_cylinders"
        # this dict is to define a canonical (mean) grasp when generating grasps
        self.canonical_pose_dict = {
            "iht_larger_cylinders": [  #100% scale+
                {
                    "hand": [
                        0.1646,
                        0.5218,
                        1.1195,
                        1.0494,
                        1.0773,
                        0.6079,
                        0.6214,
                        0.9512,
                        -0.0972,
                        1.0060,
                        0.9404,
                        0.8993,
                        -0.1031,
                        1.2183,
                        0.9816,
                        0.3949,
                    ],
                    "object": [
                        0,
                        0,
                        0.52,
                        0,
                        -0.5,
                        0.0,
                        0.5,
                    ],
                }
            ],
            "iht_smaller_cylinders": [  
                {
                    "hand": [
                        -0.0410, 
                        1.1876, 
                        0.9265, 
                        1.1414, 
                        0.9776, 
                        0.2826, 
                        -0.0343, 
                        1.5472, 
                        0.0926, 
                        1.2257, 
                        1.0409, 
                        1.1082, 
                        0.1010, 
                        1.3063,
                        0.9461, 
                        1.0226
                    ],
                    "object": [
                        0,
                        0,
                        0.52,
                        0,
                        -0.5,
                        0.0,
                        0.5,
                    ],
                }
            ],
            
        }
        self.canonical_pose = self.canonical_pose_dict[self.canonical_pose_category]
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.sampled_init_pose = torch.zeros(
            (len(self.envs), 23), dtype=torch.float, device=self.device
        )

    def reset_idx(self, env_ids):
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

        # generate random values
        rand_floats = torch_rand_float(
            -1.0,
            1.0,
            (len(env_ids), self.num_allegro_hand_dofs * 2 + 5),
            device=self.device,
        )

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0
        success = self.progress_buf[env_ids] == self.max_episode_length
        all_states = torch.cat(
            [self.allegro_hand_dof_pos, self.root_state_tensor[self.object_indices, :7]],
            dim=1,
        )
        
        self.saved_grasping_states = torch.cat(
            [self.saved_grasping_states, all_states[env_ids][success]]
        )
        # self.saved_grasping_states = torch.cat([self.saved_grasping_states, self.sampled_init_pose[env_ids][success]])
        print("current cache size:", self.saved_grasping_states.shape[0])

        pose_threshold = int(
            eval(self.num_pose_per_cache[:-1]) * 1e3
        )  # threshold to save npy file of object pose quaternion + finger joint position
        if len(self.saved_grasping_states) >= pose_threshold:
            # grasp_cache_name: usually [public_allegro, internal_allegro, internal_allegro_full]
            # canonical_pose_category: [hora, thin, pencil]
            cache_dir = "/".join(
                ["cache", self.grasp_cache_name, self.canonical_pose_category]
            )
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = f's{str(self.base_obj_scale).replace(".", "")}_{self.num_pose_per_cache}'
            cache_name = f"{cache_dir}/{cache_name}.npy"
            np.save(
                cache_name,
                self.saved_grasping_states[:pose_threshold].cpu().numpy(),
            )
            exit()

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[
            env_ids
        ].clone()
        self.root_state_tensor[
            self.object_indices[env_ids], 0:2
        ] = self.object_init_state[env_ids, 0:2]
        self.root_state_tensor[
            self.object_indices[env_ids], self.up_axis_idx
        ] = self.object_init_state[env_ids, self.up_axis_idx]

        assert self.hand_orientation == "up"

        hand_randomize_amount = {
            "iht_larger_cylinders": 0.25,
            "iht_smaller_cylinders": 0.25,
        }[self.canonical_pose_category]
        obj_randomize_amount = {
            "iht_larger_cylinders": [0.1, 0.0, 0.0],  
            "iht_smaller_cylinders": [0.1, 0.0, 0.0],  
        }[self.canonical_pose_category]

        pose_ids = np.random.randint(0, len(self.canonical_pose), size=len(env_ids))
        hand_pose = to_torch(
            [self.canonical_pose[pose_id]["hand"] for pose_id in pose_ids],
            device=self.device,
        )
        hand_pose += (
            hand_randomize_amount
            * rand_floats[:, 5 : 5 + self.num_allegro_hand_dofs]
        )
        object_pose = to_torch(
            [self.canonical_pose[pose_id]["object"] for pose_id in pose_ids],
            device=self.device,
        )
        # print(self.root_state_tensor)
        self.root_state_tensor[self.object_indices[env_ids], 0:7] = object_pose[
            :, 0:7
        ]
        # state tensor: position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        for i in range(3):
            self.root_state_tensor[self.object_indices[env_ids], i] += (
                obj_randomize_amount[i]
                * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            )[:, 0]

        self.sampled_init_pose[env_ids] = torch.cat(
            [hand_pose, self.root_state_tensor[self.object_indices[env_ids], :7]],
            dim=-1,
        )
        
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )
        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices),
        )

        self.allegro_hand_dof_pos[env_ids, :] = hand_pose
        self.allegro_hand_dof_vel[env_ids, :] = 0
        self.prev_targets[env_ids, : self.num_allegro_hand_dofs] = hand_pose
        self.cur_targets[env_ids, : self.num_allegro_hand_dofs] = hand_pose

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

        self.at_reset_buf[env_ids] = 1

    def compute_reward(self):
        def list_intersect(li, hash_num):
            # 17 is the object index
            # 4, 8, 12, 16 are fingertip index
            # return number of contact with obj_id
            obj_id = self.rigid_body_states.shape[1] - 1
            query_list = [
                obj_id * hash_num + self.fingertip_handles[0],
                obj_id * hash_num + self.fingertip_handles[1],
                obj_id * hash_num + self.fingertip_handles[2],
                obj_id * hash_num + self.fingertip_handles[3],
            ]
            return len(np.intersect1d(query_list, li))
        assert self.device == "cpu"
        contacts = [self.gym.get_env_rigid_contacts(env) for env in self.envs]
        contact_list = [
            list_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000)
            for contact in contacts
        ]
        contact_condition = to_torch(contact_list, device=self.device)

        obj_pos = self.rigid_body_states[:, [-1], :3]
        finger_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        # the sampled pose need to satisfy (check 1 here):
        # 1) all fingertips is nearby objects
        cond1 = (torch.sqrt(((obj_pos - finger_pos) ** 2).sum(-1)) < 0.1).all(-1)
        # 2) at least two fingers are in contact with object
        cond2 = contact_condition >= 2
        # 3) object does not fall after a few iterations
        cond3 = torch.greater(obj_pos[:, -1, -1], self.reset_z_threshold)
        # 4) object is in contact with the palm
        def palm_intersect(li, hash_num):
            obj_id = self.rigid_body_states.shape[1] - 1
            query = obj_id * hash_num + self.palm_handle
            return len(np.intersect1d(query, li))

        palm_contact_list = [
            palm_intersect(np.unique([c[2] * 10000 + c[3] for c in contact]), 10000)
            for contact in contacts
        ]
        palm_contact_condition = to_torch(palm_contact_list, device=self.device)
        cond4 = palm_contact_condition == 1

        cond = cond1.float() * cond2.float() * cond3.float() * cond4.float()

        # reset if any of the above condition does not hold
        self.reset_buf[cond < 1] = 1
        self.reset_buf[self.progress_buf >= self.max_episode_length] = 1
