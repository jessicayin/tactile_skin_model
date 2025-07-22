import torch
from sensor_msgs.msg import MagneticField
from gum_ros2.msg import PalmTaxel
from rclpy.node import Node

class PalmReskin(Node):
    def __init__(self, topic_prefix="/reskin/finger5/link4/sensor", num_sensors=16):
        super().__init__("reskin_buffer")

        self._cur_data = [None] * num_sensors
        self._subscribers = []
        

        for i in range(1, num_sensors+1):
            topic = f"{topic_prefix}{i}"
            subscriber = self.create_subscription(MagneticField, topic, self._callback_factory(i-1), 10) #msg type, topic, callback, queue size
            self._subscribers.append(subscriber)

        self.get_logger().info(f"Start reskin buffer with topics {topic_prefix}1 to {topic_prefix}_{num_sensors}")

    def _callback_factory(self, i):
        def _callback(data):
            self._cur_data[i] = data
        return _callback
    
    def _msg_to_three_axis_binary_tensor(self, msg):
        tensor = self._msg_to_tensor(msg)
        return (tensor > 0).float()

    def poll_binary_three_axis_data(self):
        """
        returns three axis binary data, e.g. [0, 1, 1], returns torch.Size([16, 3])
        """
        return torch.stack([self._msg_to_three_axis_binary_tensor(data) for data in self._cur_data if data is not None])
    
    def _msg_to_tensor(self, msg):
        return torch.tensor([msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z], dtype=torch.float)

    def poll_data(self):
        """
        returns three axis data, e.g. [0.1, 0.2, 0.3], returns torch.Size([16, 3])
        """
        return torch.stack([self._msg_to_tensor(data) for data in self._cur_data if data is not None])

class PalmBinary(Node):
    def __init__(self):
        super().__init__("palm_binary") 
        # subscriber = self.create_subscription(PalmTaxel, "reskin/palm_binary", self._callback, 10)
        subscriber = self.create_subscription(PalmTaxel, "reskin/real2sim", self._callback, 10)
        self._cur_data = None

        self.get_logger().info(f"Start reskin buffer with topic reskin/real2sim")
    
    def _callback(self, data):
        self._cur_data = data
        return self._cur_data
    
    def _msg_to_tensor(self, msg):
        # self.get_logger().info(f"msg.data: {msg.palmdata}") #print palmdata
        return torch.tensor([msg.palmdata], dtype=torch.float)

    def poll_binary_data(self):
        return self._msg_to_tensor(self._cur_data)