#!/usr/bin/env python
import rclpy 
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
import numpy as np
from .msg import PalmTaxel
import time
import threading
from collections import deque
import torch

class ReskinBinary(Node):
    def __init__(self):
        super().__init__('reskin_binary')
        
        subscriber = self.create_subscription(PalmTaxel, "reskin/reskin_raw_data", self._callback, 10)
        self._cur_data = None

        #hand tuned hyperparameters
        self.window_size = 10
        self.current_signal_size = 10
        self.x_threshold = 0.75
        self.y_threshold = 0.7
        self.z_threshold = 0.67
        self.binary_history_window_size = 50
        self.alpha = 0.4
        self.total_buffer_size = self.binary_history_window_size + self.current_signal_size

        self.x_raw_reskin_buffer = deque(maxlen=self.total_buffer_size)
        self.y_raw_reskin_buffer = deque(maxlen=self.total_buffer_size)
        self.z_raw_reskin_buffer = deque(maxlen=self.total_buffer_size)


        self.publisher = self.create_publisher(PalmTaxel, 'reskin/reskin_binary', 18)
        self.get_logger().info("Initialized reskin_binary node")


    def filter(self, data_buffer):
        all_filtered_data = np.zeros((16, self.total_buffer_size))  #(16, 30)
        data_buffer = np.transpose(np.asarray(data_buffer)) 
        for i in range(16): #num sensors
            data = data_buffer[i].flatten() #size (30,)
            filtered_data = np.zeros_like(data)
            filtered_data[:self.window_size] = data[:self.window_size]
            for j in range(self.window_size, len(data)):
                filtered_data[j] = self.alpha * np.mean(filtered_data[j - self.window_size: j]) + (1 - self.alpha) * np.mean(data[j - self.window_size: j])
            all_filtered_data[i, :] = filtered_data
        return all_filtered_data #output (16, 30)

    def signed_binary(self, data_buffer, threshold):
        binarized_data = np.zeros((16,1))
        for i in range(16):
            data = data_buffer[i].flatten() #(30,)
            current_signal = np.mean(data[-self.current_signal_size])
            prev_signal = np.mean(data[:-self.current_signal_size])
            if current_signal - prev_signal > threshold:
                binarized_data[i] = 1
            elif current_signal - prev_signal < -threshold:
                binarized_data[i] = -1
            else:
                binarized_data[i] = 0
        #output (16, 1)
        return binarized_data

    
    def _callback(self, data):
        self._cur_data = data
        reskin_arr = np.asarray(self._cur_data.palmdata).reshape(16, 3)
        self.x_raw_reskin_buffer.append(reskin_arr[:,0])
        self.y_raw_reskin_buffer.append(reskin_arr[:,1])
        self.z_raw_reskin_buffer.append(reskin_arr[:,2])
        if len(self.x_raw_reskin_buffer) < self.total_buffer_size:
            self.get_logger().info("Buffering reskin data...{}".format(len(self.x_raw_reskin_buffer)))
        else:

            self.x_filtered_reskin_buffer = self.filter(self.x_raw_reskin_buffer)
            self.y_filtered_reskin_buffer = self.filter(self.y_raw_reskin_buffer)
            self.z_filtered_reskin_buffer = self.filter(self.z_raw_reskin_buffer)


            x_binarized_reskin = self.signed_binary(self.x_filtered_reskin_buffer, self.x_threshold)
            y_binarized_reskin = self.signed_binary(self.y_filtered_reskin_buffer, self.y_threshold)
            z_binarized_reskin = self.signed_binary(self.z_filtered_reskin_buffer, self.z_threshold)
            z_binarized_reskin = np.where(z_binarized_reskin != 0, 1, z_binarized_reskin)

            
            current_output = np.hstack((x_binarized_reskin, y_binarized_reskin, z_binarized_reskin))

            self.publisher.publish(PalmTaxel(
                header = Header(stamp=self.get_clock().now().to_msg()), 
                palmdata = current_output.flatten().astype(np.float64),
            ))
            return current_output


def main(args=None):
    rclpy.init(args=args)

    reskin_binary = ReskinBinary()
    rclpy.spin(reskin_binary)
    threading.Thread(target=rclpy.spin, args=(reskin_binary,), daemon=True).start()

    reskin_binary.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()