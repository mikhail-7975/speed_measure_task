#!/usr/bin/env python3
"""
Median filter node for float values with configurable type (float32/float64).
Supports runtime parameter reconfiguration.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import numpy as np
from std_msgs.msg import Float32, Float64
    
from speed_measure.utils.kalman_filter import KalmanFilter

class FilterNode(Node):
    def __init__(self):
        super().__init__('median_filter_node')

        # Declare parameters with defaults
        self.declare_parameter('input_topic', 'input')
        self.declare_parameter('output_topic', 'output')
        self.declare_parameter('value_type', 'float32')  # 'float32' or 'float64'

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.value_type = self.get_parameter('value_type').value.lower()
        if self.value_type == 'float64':
            self.msg_type = Float64
            self.buffer_dtype = np.float64
        elif self.value_type == 'float32':
            self.msg_type = Float32
            self.buffer_dtype = np.float32

        self.filter = KalmanFilter()

        self.subscriber = self.create_subscription(
            self.msg_type,
            self.input_topic,
            self._callback,
            10
        )

        self.publisher = self.create_publisher(
            Float32,
            self.output_topic,
            10
        )

        self.get_logger().info(
            f"MedianFilterNode initialized:\n"
            f"  Input:  {self.input_topic} ({self.value_type})\n"
            f"  Output: {self.output_topic}\n"
        )

    def _callback(self, msg):
        """Process incoming messages with median filter."""
        val = msg.data

        filtered_value = self.filter.update(val)

        # Create and publish output message
        out_msg = (self.msg_type)()
        out_msg.data = filtered_value
        self.publisher.publish(out_msg)



def main(args=None):
    rclpy.init(args=args)
    node = FilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()