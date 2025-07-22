#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from functools import partial
from .msg import PalmTaxel
import hydra
from omegaconf import DictConfig

from std_msgs.msg import Header
from sensor_msgs.msg import MagneticField
import numpy as np

class ReskinRawData(Node):

    def __init__(self, cfg: DictConfig):
        super().__init__('reskin_palm_binary')
        self.num_palm_sensors = 16
        self.threshold = cfg.palm.threshold
        self.subscription = {}
        self.data = {} # holds most recent MagneticField for each sensor

        # subscribe to all palm sensor topics and link to callback
        for i in range(1,self.num_palm_sensors+1):
            tname = f"/reskin/finger5/link4/sensor{i}"
            self.subscription[tname] = self.create_subscription(
                MagneticField, 
                tname, 
                partial(self.callback, tname), 18)
            self.data[tname] = np.zeros(3, dtype=np.float64)
        self.subscription  # prevent unused variable warning
    
        self.publisher_ = self.create_publisher(PalmTaxel, f'reskin/reskin_raw_data', 18)
        self.timer_period = 1.0/60.0 
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def callback(self, topic_name, msg):
        self.data[topic_name][0] = msg.magnetic_field.x
        self.data[topic_name][1] = msg.magnetic_field.y
        self.data[topic_name][2] = msg.magnetic_field.z

      
    def timer_callback(self):
        current_data = np.zeros((48), dtype=np.float64)

        # note that this will access most recent data per chip, not per hand
        # desired behavior for now, for most responsive tactile output
        for idx, key in enumerate(self.data):
            axes_idx = idx*3
            if self.data[key][0]:
                current_data[axes_idx] = self.data[key][0]
            else:
                current_data[axes_idx] = 0
            if self.data[key][1]:
                current_data[axes_idx+1] = self.data[key][1]
            else:
                current_data[axes_idx+1] = 0
            if self.data[key][2]:
                current_data[axes_idx+2] = self.data[key][2]
            else:
                current_data[axes_idx+2] = 0
                
        msg = PalmTaxel()
        self.publisher_.publish(PalmTaxel(
                header = Header(stamp=self.get_clock().now().to_msg()),
                threshold = self.threshold,
                palmdata = current_data
            ))
        
@hydra.main(version_base=None, config_path="config", config_name="metahand")
def main(cfg: DictConfig):
    rclpy.init()

    segd = ReskinRawData(cfg.reskin)
    rclpy.spin(segd)

    segd.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
