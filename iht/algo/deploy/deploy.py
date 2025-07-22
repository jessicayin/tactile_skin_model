# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import threading
import time
import numpy as np
import pickle 

from gum.labs.dexit.algo.models.models import ActorCritic
from gum.labs.dexit.algo.models.running_mean_std import RunningMeanStd
from gum.labs.dexit.algo.padapt import proprio_adapt_net_config
from gum.labs.dexit.utils.misc import tprint
from gum.labs.dexit.algo.deploy.reskin import PalmBinary
import torch

def _obs_allegro2hora(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    obses = np.concatenate([obs_index, obs_thumb, obs_middle, obs_ring]).astype(
        np.float32
    )
    return obses


def _action_hora2allegro(actions):
    cmd_act = actions.copy()
    cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
    cmd_act[[12, 13, 14, 15]] = actions[[4, 5, 6, 7]]
    cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
    return cmd_act


class HardwarePlayer(object):
    def __init__(self, config):
        self.action_scale = 0.04167
        self.control_freq = 20
        self.device = "cuda:0"

        self.enable_palm_three_axis_binary = True #by default, assume S3-Axis. set to False for propio-only policy.
        self.unsigned_three_axis = False #U3-Axis
        # ReSkin input options, must match which model is running
        self.z_only_ablation = False
        self.signed_xy_only_ablation = False
        self.goal_pos = [0.09, 0.547, 0.02] #y and z are fixed in training. x range [0.03, 0.1]



        net_config = proprio_adapt_net_config(config)


        # Network configuration options
        self.config = config
        env_config = config.task.env
        self.enable_palm_binary = False

        obs_shape = (config.task.env.numObservations,)
        self.proprio_hist_dim = env_config.hora.propHistoryLen
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        torch.compile(self.model)
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
        self.running_mean_std.eval()
        torch.compile(self.running_mean_std)
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, 32)).to(self.device)
        self.sa_mean_std.eval()
        torch.compile(self.sa_mean_std)
        self.noisy_obj_quat_mean_std = RunningMeanStd((self.proprio_hist_dim, 7)).to(
            self.device
        )
        self.noisy_obj_quat_mean_std.eval()
        torch.compile(self.noisy_obj_quat_mean_std)

        # hand settings
        self.init_pose = config.deploy.init_pose
        self.allegro_dof_lower = torch.DoubleTensor(config.deploy.allegro_dof_lower).to(
            self.device
        )
        self.allegro_dof_upper = torch.DoubleTensor(config.deploy.allegro_dof_upper).to(
            self.device
        )
    
    def deploy(self, keyboard_interactive=False):
        import rclpy
        from allegro_hand_controllers.allegro_robot import AllegroRobot
        torch.backends.cudnn.benchmark = True

        # try to set up rospy
        rclpy.init()
        executor = rclpy.executors.SingleThreadedExecutor()
        allegro = AllegroRobot(hand_topic_prefix="allegroHand")
        executor.add_node(allegro)

        if self.enable_palm_three_axis_binary:
            self.palm_binary_node = PalmBinary()
            executor.add_node(self.palm_binary_node)

        threading.Thread(target=executor.spin, daemon=True).start()

        hz = self.control_freq
        ros_rate = allegro.create_rate(hz)

        if keyboard_interactive:
            input("presss to move to initial position")
        # command to the initial position
        for t in range(hz * 4):
            tprint(f"setup {t} / {hz * 4}")
            allegro_init_pose = _action_hora2allegro(np.asarray(self.init_pose, dtype=np.float64))
            allegro.command_joint_position(allegro_init_pose)
            ros_rate.sleep()
        if keyboard_interactive:
            input("presss to deploy policy")

        obses, _ = allegro.poll_joint_position(wait=True)
        obses = _obs_allegro2hora(obses)
        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 99)).astype(np.float32)).cuda() #96 joint angles + 3 goal pos
        proprio_hist_buf = torch.from_numpy(
            np.zeros((1, self.proprio_hist_dim, 16 * 2)).astype(np.float32)
        ).cuda()
        palm_reskin_buf = torch.from_numpy(np.ones((1, self.proprio_hist_dim, 48)).astype(np.float32)).cuda() #16 x 3 = 48
        palm_binary_buf = torch.from_numpy(np.ones((1, self.proprio_hist_dim, 16)).astype(np.float32)).cuda() 
        
       
        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = obses[None].clone() #may need to add 3 goal pos here

        for i in range(3):
            obs_buf[:, i * 16 + 0 : i * 16 + 16] = cur_obs_buf.clone()  # joint position
            obs_buf[
                :, i * 16 + 16 : i * 16 + 32
            ] = prev_target.clone()  # current target (obs_t-1 + s * act_t-1)
            #adds joint positions to obs_buf 3x
        obs_buf[:, 96:99] = torch.tensor(self.goal_pos, dtype=torch.float32).cuda()[None] #adds goal pos to obs_buf
      
        proprio_hist_buf[:, :, :16] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 16:32] = prev_target.clone()
        prev_obs_buf = torch.zeros(obs_buf[:, 32:96].shape)
        while True:
            obs = self.running_mean_std(obs_buf)
            input_dict = {
                "obs": obs,
                "proprio_hist": self.sa_mean_std(proprio_hist_buf),
            }
            input_dict["depth_buf"] = None

            if self.enable_palm_reskin:
                input_dict["palm_reskin_info"] = palm_reskin_buf
            if self.enable_palm_three_axis_binary:
                input_dict["palm_binary_info"] = palm_reskin_buf

            action, extrin = self.model.act_inference(input_dict)
            action = torch.clamp(action, -1.0, 1.0)
            target = prev_target + self.action_scale * action
            target = torch.clip(target, self.allegro_dof_lower, self.allegro_dof_upper)
            prev_target = target.clone()
            # interact with the hardware
            commands = target.cpu().numpy()[0]
            commands = _action_hora2allegro(commands)
            allegro.command_joint_position(commands)
            ros_rate.sleep()  # keep 20 Hz command
            # get o_{t+1}
            obses, torques = allegro.poll_joint_position(wait=False)
            obses = _obs_allegro2hora(obses)
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            cur_obs_buf = obses[None]
            prev_obs_buf = obs_buf[:, 32:96].clone() #up to 96 for the joint positions, 96:99 is goal pos
            obs_buf[:, :64] = prev_obs_buf
            obs_buf[:, 64:80] = cur_obs_buf
            obs_buf[:, 80:96] = target.clone()



            priv_proprio_buf = proprio_hist_buf[:, 1:30, :]
            cur_proprio_buf = torch.cat([cur_obs_buf, target.clone()], dim=-1)[:, None]
            proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_proprio_buf], dim=1)
            
            if self.enable_palm_three_axis_binary: #by default, assume s3-axis policy
                prev_palm_reskin = palm_reskin_buf[:, 1:, ...].clone() #shift the old values to the left
                cur_palm_reskin = self.palm_binary_node.poll_binary_data().reshape(1, 1, -1).to(self.device)
                # print(cur_palm_reskin)
                palm_reskin_buf = torch.cat([prev_palm_reskin, cur_palm_reskin], dim=1)
                if self.z_only_ablation:
                    cur_palm_reskin = cur_palm_reskin.reshape((1, 16, 3))
                    cur_palm_reskin[:, :, :2] = torch.zeros_like(cur_palm_reskin)[:, :, :2]
                    cur_palm_reskin = cur_palm_reskin.reshape((1, 1, -1))
                if self.signed_xy_only_ablation:
                    cur_palm_reskin = cur_palm_reskin.reshape((1, 16, 3))
                    cur_palm_reskin[:, :, 2] = torch.zeros_like(cur_palm_reskin)[:, :, 2]
                    cur_palm_reskin = cur_palm_reskin.reshape((1, 1, -1))
                if self.unsigned_three_axis:
                    cur_palm_reskin = torch.abs(cur_palm_reskin)
                palm_reskin_buf = torch.cat([prev_palm_reskin, cur_palm_reskin], dim=1)
            

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "point_mlp.mlp.0.weight" not in self.model.state_dict():
            # point_mlp is now optional in ProprioAdapt models, but old checkpoints
            # always have these (but they are not used)
            ckpt_model = {
                k: v for k, v in checkpoint["model"].items() if "point_mlp" not in k
            }
        else:
            ckpt_model = checkpoint["model"]
        self.model.load_state_dict(ckpt_model)
        self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
        if self.noisy_obj_quat:
            self.noisy_obj_quat_mean_std.load_state_dict(
                checkpoint["noisy_obj_quat_std"]
            )
       