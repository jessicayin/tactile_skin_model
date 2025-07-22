# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from isaacgym import gymapi, gymtorch



import copy
import os
import pathlib
import pickle
import subprocess
from collections import OrderedDict
import glob
from typing import Dict, List, Optional, Tuple, Union

import cv2
import git
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
    quat_mul,
    to_torch,
)

import iht.utils.tensor as tensor_utils
from iht.tasks.allegro_hand_hora import *
from iht.tasks.allegro_hand_hora import AllegroHandHora
from iht.tasks.base.vec_task import VecTask


def priv_info_dict_from_config(
        priv_config: Dict[str, Any]
    ) -> Dict[str, Tuple[int, int]]:
        priv_dims = OrderedDict()
        priv_dims["net_contact"] = priv_config["contact_input_dim"]
        priv_dims["obj_orientation"] = 4
        priv_dims["obj_linvel"] = 3
        priv_dims["obj_angvel"] = 3
        priv_dims["fingertip_position"] = 3 * 4
        priv_dims["fingertip_orientation"] = 4 * 4
        priv_dims["fingertip_linvel"] = 4 * 3
        priv_dims["fingertip_angvel"] = 4 * 3
        priv_dims["hand_scale"] = 1
        priv_dims["obj_restitution"] = 1

        priv_info_dict = {
            "obj_position": (0, 3),
            "obj_scale": (3, 4),
            "obj_mass": (4, 5),
            "obj_friction": (5, 6),
            "obj_com": (6, 9),
            "obj_end1": (9, 12),
            "obj_end2": (12, 15),
        }
        start_index = 15
        for name, dim in priv_dims.items():
            # (lep) Address naming incosistencies w/o changing the rest of the code
            config_key = f"enable_{name}".replace("fingertip", "ft")
            config_key = config_key.replace("position", "pos")
            config_key = config_key.replace("_net_contact", "NetContactF")
            if priv_config[config_key]:
                priv_info_dict[name] = (start_index, start_index + dim)
                start_index += dim
        return priv_info_dict

class AllegroHandHoraIHT(AllegroHandHora):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        super().__init__(config, sim_device, graphics_device_id, headless)
        import ipdb; ipdb.set_trace()

        self.epsilon = 0.0005 #double check what this is
        self.reskin_norm_max_x = 200
        self.reskin_norm_max_y = 150
        self.reskin_norm_max_z = 200
        sensing_range = 0.01 #meters -> 1 cm
        self.penetration_distance_threshold = 0.0175 + sensing_range # 0.0175 = 1/2 cylinder length of reskin urdf [m]
        self.penetration_distance_threshold_noise = 0.15 #percentage -> 15%
        self.sensor_noise_percentage = 0.03 #percentage -> 3%
        self.obj_points = torch.tensor(np.load("/home/robotdev/private-tactile-iht/data/assets/obj_points_r0.002.npy").tolist()) #load pre-sampled points on object

        self.obj_points_with_envs = self.obj_points.unsqueeze(0).expand(self.num_envs, self.obj_points.shape[0], 3) 
        self.obj_points_with_envs = self.obj_points_with_envs.to(self.device)
        self.obj_scales = torch.tensor(self.obj_scales).unsqueeze(1).unsqueeze(1)
        self.obj_scales = self.obj_scales.expand(self.num_envs, self.obj_points.shape[0], 3).to(self.device)
        self.obj_points_with_envs = self.obj_points_with_envs * self.obj_scales

        scaled_heights = self.obj_scales[:, 0, 0] * self.object_length #obj scales are just repeated across the dimensions
        scaled_radii = self.obj_scales[:, 0, 0] * self.cylinder_radius
        scaled_surface_area = 2 * np.pi * scaled_radii * scaled_heights #(num_envs, num_obj_points)

        #calculate density of points on the surface of the cylinder
        self.num_obj_points = torch.ones((self.obj_points.shape[0])) * self.obj_points.shape[0]
        self.cylinder_point_density = (self.obj_points.shape[0] / scaled_surface_area) / 1e5 #divide by 1e5 to get a number in the ones magnitude
        self.cylinder_point_density = self.cylinder_point_density.unsqueeze(1).expand(self.num_envs, 16).to(self.device)

        #add noise to the penetration distance threshold
        penetration_thresh_low = 1 - self.penetration_distance_threshold_noise
        penetration_thresh_high = 1 + self.penetration_distance_threshold_noise
        penetration_thresh_size = (self.num_envs, self.obj_points.shape[0], 16)

        self.noisy_penetration_distance_threshold = (penetration_thresh_low + (penetration_thresh_high - penetration_thresh_low) * torch.rand(penetration_thresh_size, device=self.device)) * self.penetration_distance_threshold
        # if self.viewer and self.config["collect_sim_data"]["save_data"]:
        #     self.overall_max = 0
        #     self.overall_min = 0
        #     self.overall_goal_max = 0
        #     self.overall_goal_min = 0
        #     self.avg_object_pos_max_list = []
        #     self.avg_object_pos_min_list = []
        #     self.reskin_forces_history = []
        #     self.reskin_rigid_contacts = []
        #     self.isaacgym_reskin_sensors = []
        #     self.rigid_body_states_history = []
        #     self.reskin_penetration_distances = []
        #     self.reskin_signs_history = []
        #     self.correct_reskin_signs_history = []
        #     self.correct_reskin_forces_history = []
        #     self.traj_list = []
        #     self.allegro_dof_states_history = []
        #     self.actor_root_states_history = []
        #     self.track_edge_case = []
        #     self.sd_reskin_output_history = []
        #     self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


    def _create_object_asset(self):
        assets_root = data_root / "assets"
        # object file to asset

        hand_asset_file = self.config["env"]["asset"]["handAsset"]
        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        else:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.hand_asset = self.gym.load_asset(
            self.sim, str(assets_root), hand_asset_file, hand_asset_options
        )
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name)
            for name in [
                "link_3.0_tip",
                "link_15.0_tip",
                "link_7.0_tip",
                "link_11.0_tip",
            ]
        ]
        self.palm_handle = self.gym.find_asset_rigid_body_index(
            self.hand_asset, "palm_reskin_link"
        )
        
        self.contact_handles = []
        self.finger_base_index = []
        self.fingertip_index = []
        for link_name in ["link_3.0", "link_15.0", "link_7.0", "link_11.0"]:
            for submodule_name in (
                ["tip", "cylin", "base"]
                if "half_base" not in self.config["env"]["asset"]["handAsset"]
                else ["tip", "base"]
            ):
                handle = self.gym.find_asset_rigid_body_index(
                    self.hand_asset, f"{link_name}_{submodule_name}"
                )
                if handle > 0:
                    if submodule_name == "tip":
                        self.fingertip_index.append(len(self.contact_handles))
                    if submodule_name == "base":
                        self.finger_base_index.append(len(self.contact_handles))
                    self.contact_handles.append(handle)

        sensor_pose = gymapi.Transform()
        for handle in self.contact_handles:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = (
                True  # for example contacts
            )
            sensor_options.use_world_frame = (
                True  # report forces in world frame (easier to get vertical components)
            )
            self.gym.create_asset_force_sensor(
                self.hand_asset, handle, sensor_pose, sensor_options
            )

        # load object asset
        self.object_asset_list = []
        for object_type in self.object_subtype_list:
            self.object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.collapse_fixed_joints = False
            object_asset_options.fix_base_link = False
            object_asset_options.flip_visual_attachments = False
            object_asset_options.disable_gravity = False

            object_asset = self.gym.load_asset(
                self.sim, str(assets_root), self.object_asset_file, object_asset_options
            )
            self.object_asset_list.append(object_asset)

            from urdfpy import URDF
            obj_urdf = URDF.load(os.path.join(assets_root, self.object_asset_file))
            obj_link = obj_urdf.link_map["object"]
            self.object_length = obj_link.collisions[0].geometry.cylinder.length
            self.cylinder_radius = obj_link.collisions[0].geometry.cylinder.radius
        
        assert any([x is not None for x in self.object_asset_list])
    
    
    def _collect_reskin_rigid_contacts(self):

        self._get_palm_reskin_buffer(binary=True)
        self.reskin_forces_history.append(self.reskin_forces.cpu().numpy())
        self.reskin_signs_history.append(self.reskin_signs.cpu().numpy())
        self.correct_reskin_forces_history.append(self.correct_reskin_forces.cpu().numpy())
        self.correct_reskin_signs_history.append(self.correct_reskin_signs.cpu().numpy())
        self.allegro_dof_states_history.append(self.dof_state.cpu().numpy())
        self.actor_root_states_history.append(self.root_state_tensor.cpu().numpy())
        isaacgym_forces = self._get_palm_isaac_reskin_buffer(binary=False)
        self.isaacgym_reskin_sensors.append(isaacgym_forces.cpu().numpy())
        self.track_edge_case.append(self.sd_reskin_signs.cpu().numpy())
        self.sd_reskin_output_history.append(self.sd_reskin_output.cpu().numpy())

    def save_reskin_rigid_contacts(self):
        import pickle
        sd_reskin_signs_filename = self.data_path + "sd_reskin_signs.pkl"
        sd_reskin_output_filename = self.data_path + "sd_reskin_output.pkl"
        correct_signs_filename = self.data_path + "correct_reskin_signs_history.pkl"
        correct_forces_filename = self.data_path + "correct_reskin_forces_history.pkl"
        signs_filename = self.data_path + "reskin_signs_history.pkl"
        forces_filename = self.data_path + "reskin_forces_history.pkl"
        actor_root_states_filename = self.data_path + "actor_root_states.pkl"
        allegro_dof_states_filename = self.data_path + "allegro_dof_states.pkl"
        trajectory_filename = self.data_path + "joint_trajectory.npy"
        randomization_config_filename = self.data_path + "randomization_config.pkl"
        isaacgym_sensors_filename = self.data_path + "isaacgym_reskin_sensors.pkl"
        randomization_config = self.config["env"]["randomization"] #includes object physical properties
        goal_position_filename = self.data_path + "goal_position.npy"
        full_config_filename = self.data_path + "full_config.pkl"
        np.save(goal_position_filename, self.obj_goal_pos.cpu().numpy())
        np.save(trajectory_filename, self.traj_list)
        with open(sd_reskin_output_filename, 'wb') as f:
            pickle.dump(self.sd_reskin_output_history, f)
        with open(sd_reskin_signs_filename, 'wb') as f:
            pickle.dump(self.track_edge_case, f)
        with open(correct_forces_filename, 'wb') as f:
            pickle.dump(self.correct_reskin_forces_history, f)
        with open(correct_signs_filename, 'wb') as f:
            pickle.dump(self.correct_reskin_signs_history, f)
        with open(randomization_config_filename, 'wb') as f:
            pickle.dump(randomization_config, f)
        with open(signs_filename, 'wb') as f:
            pickle.dump(self.reskin_signs_history, f)
        with open(forces_filename, 'wb') as f:
            pickle.dump(self.reskin_forces_history, f)
        with open(actor_root_states_filename, 'wb') as f:
            pickle.dump(self.actor_root_states_history, f)
        with open(allegro_dof_states_filename, 'wb') as f:
            pickle.dump(self.allegro_dof_states_history, f)
        with open(full_config_filename, 'wb') as f:
            pickle.dump(self.config, f)
        with open(isaacgym_sensors_filename, 'wb') as f:
            pickle.dump(self.isaacgym_reskin_sensors, f)

    def _create_reskin_sensors(self, env_ptr, collision_group):
        """
        create reskin sensor on palm, sensing contact forces in 8.5mm radius from center of reskin_force_sensor.urdf
        need to follow order: create asset, create sensor, create actor
        can then reuse the asset + sensor to create new actors
        lower z to where allegro hand starts, length (0.035) of reskin cylinder will be at surface of palm
        """
        repo_root = pathlib.Path(git.Repo(".", search_parent_directories=True).working_tree_dir)
        data_root = repo_root / "data"
        asset_root = data_root / "assets"
        asset_file = "reskin_force_sensor.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        self.reskin_asset = self.gym.load_asset(
            self.sim, str(asset_root), asset_file, asset_options
        )

        self.sensor_reskin_coords = [
        [-0.03889, 0.01485, 0.5], #sensor 1
        [-0.03819, 0.03049, 0.5], #sensor 2
        [-0.02667, 0.00885, 0.5], #sensor 3
        [-0.02439, 0.01985, 0.5], #sensor 4
        [-0.02389, 0.03335, 0.5],
        [-0.01189, 0.02785, 0.5],
        [-0.01261, 0.01139, 0.5],
        [0, 0.01908, 0.5],
        [-0.00011, 0.03342, 0.5],
        [0.01211, 0.02785, 0.5],
        [0.01261, 0.01135, 0.5],
        [0.02696, 0.00891, 0.5],
        [0.02461, 0.01985, 0.5],
        [0.02381, 0.03350, 0.5],
        [0.03814, 0.03054, 0.5],
        [0.03917, 0.01492, 0.5],
        ]

        
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = False
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = True

        body_idx = self.gym.find_asset_rigid_body_index(self.reskin_asset, "reskin_force_sensor")
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0)) #surface of sensor
        _reskin_sensor = self.gym.create_asset_force_sensor(self.reskin_asset, body_idx, sensor_pose, sensor_props)
        reskin_pose = gymapi.Transform()

        for i in range(len(self.sensor_reskin_coords)):
            reskin_pose.p = gymapi.Vec3(*self.sensor_reskin_coords[i])
            reskin_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            reskin_actor = self.gym.create_actor(env_ptr, self.reskin_asset, reskin_pose, "reskin_sensor", collision_group, 0, 3) #collision group, bitwise filter for elements, seg id

    def _save_palm_reskin_forces(self, timestamp):
        #from one env, get the forces from all 16 sensors and save it to a buffer and np array
        reskin_forces_np = np.array(self.reskin_forces_history)
        filename = f'isaacgym_reskin_forces_{timestamp}.npy'
        np.save(filename, reskin_forces_np)

    def _get_palm_isaac_reskin_buffer(self, binary=False):
        self.reskin_index_force_sensors = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        reskin_forces = self.force_sensors[:,self.reskin_index_force_sensors, :3]
        return reskin_forces

    def filter_close_contacts(self):
        self.sensor_reskin_coords = [
        [-0.03889, 0.01485, 0.5], #sensor 1
        [-0.03819, 0.03049, 0.5], #sensor 2
        [-0.02667, 0.00885, 0.5], #sensor 3
        [-0.02439, 0.01985, 0.5], #sensor 4
        [-0.02389, 0.03335, 0.5],
        [-0.01189, 0.02785, 0.5],
        [-0.01261, 0.01139, 0.5],
        [0, 0.01908, 0.5],
        [-0.00011, 0.03342, 0.5],
        [0.01211, 0.02785, 0.5],
        [0.01261, 0.01135, 0.5],
        [0.02696, 0.00891, 0.5],
        [0.02461, 0.01985, 0.5],
        [0.02381, 0.03350, 0.5],
        [0.03814, 0.03054, 0.5],
        [0.03917, 0.01492, 0.5],
        ]
        cylinder_linvel = (self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)
        angdiff = tensor_utils.quat_to_axis_angle(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        )
        cylinder_angvel = angdiff / (self.control_freq_inv * self.dt) 
        object_rot_with_obj_points = self.object_rot.unsqueeze(1).expand(self.num_envs, self.obj_points.shape[0], 4) #expand to (self.num_envs, num_obj_points, 4) for quat_apply
        rotations = quat_apply(object_rot_with_obj_points, self.obj_points_with_envs) #self.obj_points_with_envs.shape = (self.num_envs, num_obj_points, 3) 
        rotations = rotations.reshape(self.num_envs, -1, 3) #expand back to (self.num_envs, num_obj_points, 3) for translation
        object_pos_with_obj_points = self.object_pos.unsqueeze(1).expand(self.num_envs, self.obj_points.shape[0], 3)
        cylinder_points = object_pos_with_obj_points + rotations 
        

        reskin_coords = torch.tensor(self.sensor_reskin_coords, device=self.device)
        cylinder_points_per_reskin = cylinder_points.unsqueeze(2).expand(self.num_envs, self.obj_points.shape[0], reskin_coords.shape[0], 3)
        reskin_coords_per_env = reskin_coords.unsqueeze(0).unsqueeze(0)
        reskin_coords_per_env = reskin_coords_per_env.expand(self.num_envs, self.obj_points.shape[0], reskin_coords.shape[0], 3)

        #subtract the x, y, z coordinates of the cylinder points from the reskin coordinates to get the displacement component in each axis. both variables are (num_envs, num_cylinder_points, sensor_num, 3)
        diff = cylinder_points_per_reskin - reskin_coords_per_env

        dist = torch.norm(diff, dim=-1) #take the norm to get the euclidean distance between the cylinder points and the reskin coordinates
        signed_dist = self.noisy_penetration_distance_threshold - dist

        return dist, cylinder_linvel, cylinder_angvel, signed_dist
    
    def _get_palm_reskin_buffer(self, binary=False):
        dist, cylinder_linvel, cylinder_angvel, signed_dist = self.filter_close_contacts()
        dist = dist.reshape(self.num_envs, -1, 16)
        #mask by euclidean distance from cylinder coordinate point to each sensor, (num_envs, num_cylinder_points, sensor_num)
        close_contacts_euclidean = torch.where(dist < self.noisy_penetration_distance_threshold, torch.tensor(1), torch.tensor(0))
        #expand to x,y,z dimensions for masking, (num_envs, num_cylinder_points, sensor_num, 3) 
        close_contacts_xyz_mask = close_contacts_euclidean.unsqueeze(-1).expand(-1, -1, -1, 3)
        # close_contacts_xyz_mask = close_contacts_euclidean.reshape(self.num_envs, -1, 16).unsqueeze(-1).expand(-1, -1, -1, 3)
        #keep the distances of the points that are close to the sensors, otherwise set to 0 (num_envs, num_cylinder_points, sensor_num),
        distances = torch.where(close_contacts_xyz_mask[:, :, :, 0] > 0, dist, torch.zeros_like(dist))
        #subtract the distances from threshold to get the penetration distance (num_cylinder_points, sensor_num)
        penetration_distances = self.noisy_penetration_distance_threshold - distances
        #filtered_penetration_distance.shape = (num_envs, num_cylinder_points, 16)
        filtered_penetration_distance = torch.where(penetration_distances < self.noisy_penetration_distance_threshold, penetration_distances, torch.zeros_like(penetration_distances))
        #expand linvel to (num_envs, num_cylinder_points, sensor_num, 3) - repeat across all sensors
        cylinder_linvel_xyz = cylinder_linvel.unsqueeze(1).unsqueeze(2).expand(self.num_envs, distances.shape[1], 16, 3)
        cylinder_angvel_xyz = cylinder_angvel.unsqueeze(1).unsqueeze(2).expand(self.num_envs, distances.shape[1], 16, 3)
        pd_linvel_xyz = cylinder_linvel_xyz * filtered_penetration_distance.unsqueeze(-1).expand_as(cylinder_linvel_xyz)
        pd_angvel_xyz = cylinder_angvel_xyz * filtered_penetration_distance.unsqueeze(-1).expand_as(cylinder_linvel_xyz)
        
        pd_linvel_x = pd_linvel_xyz[:, :, :, 0]
        pd_linvel_y = pd_linvel_xyz[:, :, :, 1]
        pd_angvel_x = pd_angvel_xyz[:, :, :, 0]
        pd_angvel_y = pd_angvel_xyz[:, :, :, 1]

        shear_force_x = torch.sum((pd_linvel_x + pd_angvel_x), dim=1) / self.cylinder_point_density
        shear_force_y = torch.sum((pd_linvel_y + pd_angvel_y), dim=1) / self.cylinder_point_density
        normal_force_z = torch.sum(filtered_penetration_distance, dim=1) / self.cylinder_point_density
        reskin_output = torch.stack([shear_force_x, shear_force_y, normal_force_z], dim=-1)
        epsilon = 1e-2 #threshold for binarization

        if binary == False: #continuous reskin output (not deployed to real world)
            reskin_binary = torch.where(torch.abs(reskin_output) < epsilon, torch.zeros_like(reskin_output), reskin_output)
            reskin_signs = torch.sign(reskin_binary)
            self.reskin_signs = reskin_signs
            self.reskin_forces = reskin_output
            return reskin_output
        else:
            reskin_output = torch.where(torch.abs(reskin_output) < epsilon, torch.zeros_like(reskin_output), reskin_output)
            reskin_signs = torch.sign(reskin_output) #all axes, signed three axis

            # noise simulation + ablations
            if self.config["env"]["reskin_noise"]["modality"] == "unsigned_three_axis":
                num_elements = reskin_signs.numel()
                num_change = int(num_elements * self.sensor_noise_percentage)
                mask = torch.zeros(num_elements, dtype=torch.bool)
                mask[:num_change] = True
                mask = mask[torch.randperm(num_elements)].reshape(reskin_signs.shape)
                random_tensor = torch.randint(0, 2, reskin_signs.shape)
                random_tensor = random_tensor.float().to(self.device)
                reskin_signs[mask] = random_tensor[mask]

            elif self.config["env"]["reskin_noise"]["modality"] == "signed_three_axis":
                num_elements = reskin_signs.numel()
                num_change = int(num_elements * self.sensor_noise_percentage)
                mask = torch.zeros(num_elements, dtype=torch.bool)
                mask[:num_change] = True
                mask = mask[torch.randperm(num_elements)].reshape(reskin_signs.shape) #z only
                random_tensor_xy = torch.randint(0, 3, (reskin_signs.shape[0], reskin_signs.shape[1], 2))
                random_tensor_xy -= 1
                random_tensor_z = torch.randint(0, 2, (reskin_signs.shape[0], reskin_signs.shape[1], 1))
                random_tensor = torch.cat((random_tensor_xy, random_tensor_z), dim=2)
                random_tensor = random_tensor.float().to(self.device)
                reskin_signs[mask] = random_tensor[mask]

            elif self.config["env"]["reskin_noise"]["modality"] == "normal_only":
                num_elements = reskin_signs[:, :, 2].numel() 
                num_change = int(num_elements * self.sensor_noise_percentage)
                mask = torch.zeros(num_elements, dtype=torch.bool)
                mask[:num_change] = True
                mask = mask[torch.randperm(num_elements)].reshape(reskin_signs[:, :, 2].shape)
                random_tensor_z = torch.randint(0, 2, (reskin_signs.shape[0], reskin_signs.shape[1], 1))
                random_tensor = random_tensor_z.float().to(self.device) 
                reskin_signs[:, :, 2][mask] = random_tensor[mask].squeeze(1) 

            elif self.config["env"]["reskin_noise"]["modality"] == "signed_shear_only":
                num_elements = reskin_signs[:, :, :2].numel()
                num_change = int(num_elements * self.sensor_noise_percentage)
                mask = torch.zeros(num_elements, dtype=torch.bool)
                mask[:num_change] = True
                mask = mask[torch.randperm(num_elements)].reshape(reskin_signs[:, :, :2].shape)
                random_tensor_xy = torch.randint(0, 3, (reskin_signs.shape[0], reskin_signs.shape[1], 2))
                random_tensor_xy -= 1
                random_tensor = random_tensor_xy.float().to(self.device)
                reskin_signs[:, :, :2][mask] = random_tensor[mask]
            else:
                raise ValueError("Invalid reskin noise modality, check that config file specifies one of the following: unsigned_three_axis, signed_three_axis, normal_only, signed_shear_only") 

            self.reskin_signs = reskin_signs
            self.reskin_forces = reskin_output
            return reskin_signs 

    
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
    def compute_observations(self):
        self._refresh_gym()
        # observation noise
        random_obs_noise_t = torch.normal(
            0,
            self.random_obs_noise_t_scale,
            size=self.allegro_hand_dof_pos.shape,
            device=self.device,
            dtype=torch.float,
        )
        noisy_joint_pos = (
            random_obs_noise_t + self.random_obs_noise_e + self.allegro_hand_dof_pos
        )
        # deal with normal observation, do sliding window
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)
        cur_tar_buf = self.cur_targets[:, None]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)
        cur_obs_buf = torch.cat([cur_obs_buf[:], self.obj_goal_pos.unsqueeze(1)], dim=-1) #(10, 1, 35)
        

        # if self.obs_with_binary_contact:
        #     tip_binary_contact = (
        #         (self.force_sensors[:, self.fingertip_index, :3] ** 2).sum(-1) > 0.1
        #     ).float()[:, None]
            # because of hardware limitation, we force the ring finger has no information about the contact
            # tip_binary_contact[:, :, -1] = 0
            # cur_obs_buf = torch.cat([cur_obs_buf, tip_binary_contact], dim=-1)
        self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        not_at_reset_env_ids = torch.where(self.at_reset_buf == 0)[0]
        self.obs_buf_lag_history[at_reset_env_ids, :, 0:16] = noisy_joint_pos[
            at_reset_env_ids
        ].unsqueeze(1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 16:32] = self.allegro_hand_dof_pos[
            at_reset_env_ids
        ].unsqueeze(1)
        # if self.obs_with_binary_contact:
        #     tip_binary_contact = (
        #         (self.force_sensors[:, self.fingertip_index, :3] ** 2).sum(-1) > 0.1
        #     ).float()[:, None]
            # because of hardware limitation, we force the ring finger has no
            # information about the contact
            # tip_binary_contact[:, :, -1] = 0
            # self.obs_buf_lag_history[at_reset_env_ids, :, 32:36] = tip_binary_contact[
            #     at_reset_env_ids
            # ]
        # select all of the joint angles in obs_buf_lag_history
        # first (-3:) selects bottom -3 of obs buffer with size (num_env, 80, num_obs) 
        # last (:-3) is the object goal position (x, y, z) which doesn't change
        joint_buf = (self.obs_buf_lag_history[:, -3:, :32].reshape(self.num_envs, -1)).clone()
        #update the joint angles in the history and add back the object goal position
        #t_buf is a temporary buffer?
        t_buf = torch.hstack((joint_buf, self.obs_buf_lag_history[:, -1:, 32:].reshape(self.num_envs, -1)))
        self.obs_buf[:, : t_buf.shape[1]] = t_buf
        # velocity reset
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]

        # stage 2 buffer
        if self.enable_palm_reskin:
            palm_reskin_contact = self._get_palm_reskin_buffer(binary=False)
            prev_palm_reskin_contact = self.palm_reskin_buffer[:, 1:].clone()
            cur_palm_reskin_contact = palm_reskin_contact[:, None]
            self.palm_reskin_buffer[:] = torch.cat(
                [prev_palm_reskin_contact, cur_palm_reskin_contact], dim=1
            )
            self.palm_reskin_buffer[at_reset_env_ids] = cur_palm_reskin_contact[
                at_reset_env_ids
            ]
        
        if self.enable_palm_binary:
            palm_binary_contact = self._get_palm_reskin_buffer(binary=True)
            prev_palm_binary_contact = self.palm_binary_buffer[:, 1:].clone()
            cur_palm_binary_contact = palm_binary_contact[:, None]
            self.palm_binary_buffer[:] = torch.cat([prev_palm_binary_contact, cur_palm_binary_contact], dim=1)
            self.palm_binary_buffer[at_reset_env_ids] = cur_palm_binary_contact[at_reset_env_ids]


        not_at_reset_envs_mask = self.at_reset_buf == 0
        self.at_reset_buf[at_reset_env_ids] = 0
        self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :32]
        if self.enable_priv_obj_end1 and self.enable_priv_obj_end2:
            object_ends = [
                [ 0, 0, -1* self.object_length / 2],
                [0, 0, self.object_length / 2]
            ]
            self.obj_end1 = self.object_pos + quat_apply(self.object_rot, to_torch(object_ends[0], device=self.device)[None].repeat(self.num_envs, 1))
            self.obj_end2 = self.object_pos + quat_apply(self.object_rot, to_torch(object_ends[1], device=self.device)[None].repeat(self.num_envs, 1))
        # Update private info buffers
        # the dict maps attribute name to the tensor from which it'll be updated
        update_priv_info_dict: Dict[str, torch.Tensor] = {
            "obj_position": self.object_pos,
            "net_contact": self._get_contact_force(),
            "obj_orientation": self.object_rot,
            "obj_linvel": self.obj_linvel_at_cf,
            "obj_angvel": self.obj_angvel_at_cf,
            "obj_end1": self.obj_end1,
            "obj_end2": self.obj_end2,
            "fingertip_position": self.fingertip_pos,
            "fingertip_orientation": self.fingertip_orientation,
            "fingertip_linvel": self.ft_linvel_at_cf,
            "fingertip_angvel": self.ft_angvel_at_cf,
        }
        for name, tensor in update_priv_info_dict.items():
            self._update_priv_buf(
                env_id=range(self.num_envs), name=name, value=tensor.clone()
            )

    
        self.critic_info_buf[:, 0:4] = self.object_rot
        self.critic_info_buf[:, 4:7] = self.obj_linvel_at_cf
        self.critic_info_buf[:, 7:10] = self.obj_angvel_at_cf
        # 4, 8, 12, 16 are fingertip indexes for rigid body states
        fingertip_states = self.rigid_body_states[:, self.fingertip_handles].clone()
        self.critic_info_buf[:, 10 : 10 + 13 * 4] = fingertip_states.reshape(
            self.num_envs, -1
        )
     
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
        self.enable_priv_obj_end1 = p_config["enable_obj_end1"]
        self.enable_priv_obj_end2 = p_config["enable_obj_end2"]
        self.priv_info_dict = priv_info_dict_from_config(p_config)
    
    def viz_quick_stats(self):
        curr_object_pos_max = max(self.object_pos[:,0])
        curr_object_pos_min = min(self.object_pos[:,0])
        curr_object_goal_max = max(self.obj_goal_pos[:,0])
        curr_object_goal_min = min(self.obj_goal_pos[:,0])
        self.avg_object_pos_max_list.append(self.overall_max)
        self.avg_object_pos_min_list.append(self.overall_min)
        avg_object_pos_max = sum(self.avg_object_pos_max_list)/len(self.avg_object_pos_max_list)
        avg_object_pos_min = sum(self.avg_object_pos_min_list)/len(self.avg_object_pos_min_list)
        print("current obj x-axis max: {max_pos}".format(max_pos=curr_object_pos_max))
        print("current obj x-axis min: {min_pos}".format(min_pos=curr_object_pos_min))
        print("current goal obj x-axis max: {max_pos}".format(max_pos=curr_object_goal_max))
        print("current goal obj x-axis min: {min_pos}".format(min_pos=curr_object_goal_min))
        print("avg obj x-axis max: {max_pos}".format(max_pos=avg_object_pos_max))
        print("avg obj x-axis min: {min_pos}".format(min_pos=avg_object_pos_min))
        if self.overall_max < curr_object_pos_max:
            self.overall_max = curr_object_pos_max
        if self.overall_goal_max < curr_object_goal_max:
            self.overall_goal_max = curr_object_goal_max
        if self.overall_goal_min < curr_object_goal_min:
            self.overall_goal_min = curr_object_goal_min
        if self.overall_min > curr_object_pos_min:
            self.overall_min = curr_object_pos_min
        print("overall obj x-axis max: {max_pos}".format(max_pos=self.overall_max))
        print("overall obj x-axis min: {min_pos}".format(min_pos=self.overall_min))
    
    def _save_rendered_frame(self):
        frames_dir = self.data_path + "sim_frames/"
        current_logs = sorted(glob.glob(frames_dir + '*.png'))
        if current_logs:
            last_log = int(current_logs[-1].split('_')[-1].split('.')[0])
            log_num = f'{last_log+1:03}'
        else:
            log_num = '000'
        frame_filename = 'frame_' + log_num + '.png'
        frame_save_path = frames_dir + frame_filename
        self.gym.write_viewer_image_to_file(self.viewer, frame_save_path)

    def compute_reward(self):

        #NOTE: sim data collection only
        # if self.config["collect_sim_data"]["save_data"]:
        #     self.gym.sync_frame_time(self.sim)
        #     self._collect_reskin_rigid_contacts()
        #     self._save_rendered_frame()
        #     self.save_reskin_rigid_contacts()
            
        # pose diff penalty
        pose_diff_penalty = ((self.allegro_hand_dof_pos - self.init_pose_buf) ** 2).sum(
            -1
        )
        # work and torque penalty
        # TODO: only consider -1 is incorrect, but need to find the new scale
        torque_penalty = (self.torques[:, -1] ** 2).sum(-1)
        work_penalty = (
            (
                torch.abs(self.torques[:, -1])
                * torch.abs(self.dof_vel_finite_diff[:, -1])
            ).sum(-1)
        ) ** 2
        dof_vel_penalty = (self.dof_vel_finite_diff[:, -1] ** 2).sum(-1)
        # Compute offset in radians. Radians -> radians / sec
        angdiff = tensor_utils.quat_to_axis_angle(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        )
        object_angvel = angdiff / (self.control_freq_inv * self.dt)
        vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)

        iht_reward = -1 * ((self.object_pos - self.obj_goal_pos)**2).sum(-1)
        # if self.viewer and self.object_type in ["rolling_pin", "pool_noodle"]:
            # self.viz_quick_stats()

        # linear velocity: use position difference instead of self.object_linvel
        object_linvel = (
            (self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)
        ).clone()
        # object_linvel_penalty = torch.norm(object_linvel, p=1, dim=-1)

        #NOTE: repurpose linear velocity penalty to be a TASK COMPLETION reward
        
        distances = ((self.object_pos - self.obj_goal_pos)**2).sum(-1)
        task_complete_mask = distances < self.epsilon
        task_complete = distances.clone()
        task_complete[task_complete_mask] = 1.0
        task_complete[~task_complete_mask] = 0.0
        object_linvel_penalty = task_complete

        # TODO: move this to a more appropriate place
        self.obj_angvel_at_cf = object_angvel
        self.obj_linvel_at_cf = object_linvel
        ft_angdiff = tensor_utils.quat_to_axis_angle(
            quat_mul(
                self.fingertip_orientation.reshape(-1, 4),
                quat_conjugate(self.ft_rot_prev.reshape(-1, 4)),
            )
        ).reshape(-1, 12)
        self.ft_angvel_at_cf = ft_angdiff / (self.control_freq_inv * self.dt)
        self.ft_linvel_at_cf = (self.fingertip_pos - self.ft_pos_prev) / (
            self.control_freq_inv * self.dt
        )

        dof_acc_penalty = (
            (self.dof_vel_finite_diff[:, -1] - self.dof_vel_prev[:, -1]) ** 2
        ).sum(-1)

        vec_cross = torch.cross(object_angvel, self.rot_axis_buf)
        unclip_rot_penalty = (vec_cross**2).sum(-1)
        undesired_rotation_penalty = torch.clip(
            unclip_rot_penalty, max=self.perp_angvel_clip_max
        )

        if len(self.finger_base_index) > 0:
            finger_base_contact = self.force_sensors[:, self.finger_base_index, :3]
            finger_base_contact = (finger_base_contact**2).sum(-1).sum(-1)
            finger_base_penalty = -1 * torch.clip(finger_base_contact, max=1.0) #multiply -1 to turn into reward
        else:
            finger_base_penalty = to_torch([0], device=self.device)

        position_penalty = -1* ((self.obj_end1[:, 2] - self.obj_end2[:, 2]) ** 2 + (self.obj_end1[:,1] - self.obj_end2[:,1])**2)

        drop_z_threshold = 0.48

        z_dist_penalty = torch.clip(self.object_pos[:,2] - drop_z_threshold, min = -1, max = 0)

        finger_obj_penalty = (
            (self.fingertip_pos - self.obj_end1.repeat(1, 4)) ** 2
        ).sum(-1)

        # penalize variance of applied forces from fingers to prevent one finger from dominating the gait
        fingertip_forces_var = torch.var(self.fingertip_forces, dim=1)
        contact_force_reward = -1 * torch.clip(fingertip_forces_var, max=1)
        

        self.rew_buf[:] = compute_hand_reward(
            object_linvel_penalty,
            self._get_reward_scale_by_name("obj_linvel_penalty"),
            iht_reward,
            self._get_reward_scale_by_name("rotate_reward"),
            pose_diff_penalty,
            self._get_reward_scale_by_name("pose_diff_penalty"),
            torque_penalty,
            self._get_reward_scale_by_name("torque_penalty"),
            dof_vel_penalty,
            self._get_reward_scale_by_name("dof_vel_penalty"),
            work_penalty,
            self._get_reward_scale_by_name("work_penalty"),
            undesired_rotation_penalty,
            self._get_reward_scale_by_name("undesired_rotation_penalty"),
            finger_base_penalty,
            self._get_reward_scale_by_name("finger_base_penalty"),
            z_dist_penalty,
            self._get_reward_scale_by_name("pencil_z_dist_penalty"),
            dof_acc_penalty,
            self._get_reward_scale_by_name("dof_acc_penalty"),
            position_penalty,
            self._get_reward_scale_by_name("position_penalty"),
            finger_obj_penalty,
            self._get_reward_scale_by_name("finger_obj_penalty"),
            contact_force_reward,
            self._get_reward_scale_by_name("contact_force_reward"),
        )
        # import ipdb; ipdb.set_trace()
        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras["step_all_reward"] = self.rew_buf.mean()
        self.extras["rotation_reward"] = iht_reward.mean()
        self.extras["undesired_rotation_penalty"] = unclip_rot_penalty.mean()
        self.extras["penalty/touch_finger_base"] = finger_base_penalty.mean()
        self.extras["penalty/dof_vel"] = dof_vel_penalty.mean()
        self.extras["penalty/dof_acc"] = dof_acc_penalty.mean()
        self.extras["penalty/position"] = position_penalty.mean()
        self.extras["penalty/finger_obj"] = finger_obj_penalty.mean()
        self.extras["object_linvel_penalty"] = object_linvel_penalty.mean()
        self.extras["pose_diff_penalty"] = pose_diff_penalty.mean()
        self.extras["work_done"] = work_penalty.mean()
        self.extras["torques"] = torque_penalty.mean()
        self.extras["roll"] = torch.abs(object_angvel[:, 0]).mean()
        self.extras["pitch"] = torch.abs(object_angvel[:, 1]).mean()
        self.extras["yaw"] = torch.abs(object_angvel[:, 2]).mean()
        self.extras["z_dist_penalty"] = z_dist_penalty.mean()
