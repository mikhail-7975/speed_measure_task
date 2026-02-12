#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Imu
from std_msgs.msg import Float32, Float64
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

import cv2
import numpy as np

from speed_measure.algorithms.optical_flow.optical_flow_estimator import OpticalFlowEstimator
from speed_measure.algorithms.optical_flow.utils import visualize_dense_flow
from speed_measure.utils.derivative_buffer import DerivativeBuffer
from speed_measure.utils.kalman_filter import KalmanFilter


def timestamp2float(ts):
    return ts.sec + ts.nanosec * 1e-9

class OpticalFlowEstimationNode(Node):
    def __init__(self):
        super().__init__('estimate_optical_flow')

        # Parameters (configurable via CLI/launch file)
        self.declare_parameter('input_topic', '/camera/image/raw')
        self.declare_parameter('alt_topic', '/alt')
        self.declare_parameter('compass_topic', '/compass')
        self.declare_parameter('output_topic', '/optical_flow')
        self.declare_parameter('output_velocity_topic', '/velocity')
        self.declare_parameter('input_encoding', 'bgr8')
        self.declare_parameter('vertical_speed_threshold', '0.4')

        # Bridge for OpenCV ↔ ROS conversion
        self.bridge = CvBridge()

        # Subscriber and Publisher
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter('input_topic').value,
            self.callback,
            10
        )

        self.alt_subscription = self.create_subscription(
            Float32,
            self.get_parameter('alt_topic').value,
            self.upd_alt_callback,
            10
        )

        self.alt_subscription = self.create_subscription(
            Float64,
            self.get_parameter('compass_topic').value,
            self.upd_compass_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            self.get_parameter('output_topic').value,
            10
        )

        self.vel_publisher = self.create_publisher(
            PointStamped,
            self.get_parameter("output_velocity_topic").get_parameter_value().string_value,
            10,
        )

        self.estimator = OpticalFlowEstimator()
        self.curr_frame = None
        self.prev_frame = None

        self.cur_ts = None
        self.prev_ts = None
        
        self.alt = DerivativeBuffer(np.float32)
        self.vertical_speed_threshold = float(self.get_parameter('vertical_speed_threshold').value)
        self.compass = DerivativeBuffer(np.float64)

        self.vx_kalman_filter = KalmanFilter()
        self.vy_kalman_filter = KalmanFilter()

        self.fx = 537.0
        self.fy = 532.9

    def get_time_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def upd_alt_callback(self, alt_msg: Float32):
        alt = np.float32(alt_msg.data)
        ts = np.float64(self.get_time_seconds())
        self.alt.append(alt, ts)

    def upd_compass_callback(self, compass_msg: Float64):
        compass = np.float64(compass_msg.data)
        ts = np.float64(self.get_time_seconds())
        self.compass.append(compass, ts)    

    def callback(self, msg: CompressedImage):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_image is None:
                self.get_logger().error('Failed to decode compressed image')
                return
            
            if self.curr_frame is not None:
                self.prev_frame = self.curr_frame
                self.prev_ts = self.cur_ts
            self.curr_frame = cv_image
            self.cur_ts = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
            if self.prev_frame is None:
                return
            
            flow = self.estimator.calculate_dense_flow(self.prev_frame, self.curr_frame)
            if flow is None:
                return

            vis = visualize_dense_flow(flow)
            image_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            image_msg.header = msg.header  # Preserve timestamp/frame_id
            self.publisher.publish(image_msg)

            dx, dy = self.estimator.get_average_flow(flow)

            cur_alt = self.alt.get_value()

            dt = self.cur_ts - self.prev_ts
            dw = self.compass.derivate()
            dz = self.alt.derivate()

            if abs(dz) > self.vertical_speed_threshold:
                self.get_logger().warning(f"Vertical speed > {self.vertical_speed_threshold} m/s, velocity incorrect")

            vx, vy = cur_alt * dx / self.fx / dt, cur_alt * dy / self.fy / dt
            filtered_vx, filtered_vy = self.vx_kalman_filter.update(vx), self.vy_kalman_filter.update(vy)

            out_msg = PointStamped()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.point.x = float(filtered_vx) 
            out_msg.point.y = float(filtered_vy)
            out_msg.point.z = float(dz)
            self.vel_publisher.publish(out_msg)


        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()