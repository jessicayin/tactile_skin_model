# Isolated ReSkin model
# Does not run on its own, but highlights key functions and provides a skeleton for how to integrate into a typical RL training pipeline

from isaacgym import gymapi, gymtorch
import torch
import numpy as np
from isaacgym.torch_utils import (
    quat_apply,
    quat_conjugate,
    quat_mul,
)

class ReSkinSimulation():
    def __init__(self):
        self.num_envs = 1 # defined in config, but included here for completeness
        sensing_range = 0.01 #meters -> 1 cm
        self.penetration_distance_threshold = 0.0175 + sensing_range # 0.0175 = 1/2 cylinder length of reskin urdf [m]
        self.penetration_distance_threshold_noise = 0.15 #percentage -> 15%
        self.sensor_noise_percentage = 0.03 #percentage -> 3%
        self.obj_points = torch.tensor(np.load("./data/assets/obj_points_r0.002.npy").tolist()) #load pre-sampled points on object
        #NOTE: (EXPERIMENTAL / UNTESTED) for objects with non-uniform COM / density: points should be sampled according to mass distribution (i.e., more points on the surfaces where mass is higher) 
        #NOTE: the points should be sampled only on the collision surface of the object, which may require decimation of internal surfaces of the mesh

        #object points across number of environments
        self.obj_points_with_envs = self.obj_points.unsqueeze(0).expand(self.num_envs, self.obj_points.shape[0], 3) 
        self.obj_points_with_envs = self.obj_points_with_envs.to(self.device)
        self.obj_scales = torch.tensor(self.obj_scales).unsqueeze(1).unsqueeze(1)
        #for training with different scales of the canonical object, scale list defined in configs/task/AllegroHandIHT.yaml: env.randomization.randomizeScaleList
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

    def filter_close_contacts(self):

        #locations of the ReSkin sensors' origins in the world frame
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

        return dist, cylinder_linvel, cylinder_angvel
    
    def _get_palm_reskin_buffer(self, binary=False):
        dist, cylinder_linvel, cylinder_angvel = self.filter_close_contacts()
        dist = dist.reshape(self.num_envs, -1, 16)
        #mask by euclidean distance from cylinder coordinate point to each sensor, (num_envs, num_cylinder_points, sensor_num)
        close_contacts_euclidean = torch.where(dist < self.noisy_penetration_distance_threshold, torch.tensor(1), torch.tensor(0))
        #expand to x,y,z dimensions for masking, (num_envs, num_cylinder_points, sensor_num, 3) 
        close_contacts_xyz_mask = close_contacts_euclidean.unsqueeze(-1).expand(-1, -1, -1, 3)
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
    
    def compute_observations(self):
        #representative function for observation computation, HORA reference: https://github.com/HaozhiQi/hora/blob/aa4d654d17eedf53104c317aa5262088bf2d825c/hora/tasks/allegro_hand_hora.py#L294
        self._refresh_gym()

        palm_reskin_contact = self._get_palm_reskin_buffer(binary=False) #calculates sensor signals
        prev_palm_reskin_contact = self.palm_reskin_buffer[:, 1:].clone() 
        #add sensor signals to observation buffer, assuming a sliding window of sensor history
        cur_palm_reskin_contact = palm_reskin_contact[:, None]
        self.palm_reskin_buffer[:] = torch.cat(
            [prev_palm_reskin_contact, cur_palm_reskin_contact], dim=1
        )
        self.palm_reskin_buffer[at_reset_env_ids] = cur_palm_reskin_contact[
            at_reset_env_ids
        ]

    def _fill_obs_dict_from_bufs(self) -> Dict[str, torch.Tensor]:
        #HORA reference: https://github.com/HaozhiQi/hora/blob/aa4d654d17eedf53104c317aa5262088bf2d825c/hora/tasks/allegro_hand_hora.py#L424
        self.obs_dict["palm_reskin_info"] = self.palm_reskin_buffer.to(self.rl_device)
