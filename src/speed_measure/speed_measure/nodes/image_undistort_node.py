#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from speed_measure.utils.image_undistortion import CameraUndistorter

class UndistortImageNode(Node):
    def __init__(self):
        super().__init__('compressed_to_image')

        # Parameters (configurable via CLI/launch file)
        self.declare_parameter('input_topic', '/camera/image/compressed')
        self.declare_parameter('output_topic', '/camera/image/raw')
        self.declare_parameter('output_encoding', 'bgr8')  # 'bgr8' for OpenCV, 'rgb8' for ROS convention

        # Bridge for OpenCV ↔ ROS conversion
        self.bridge = CvBridge()

        # Subscriber and Publisher
        self.subscription = self.create_subscription(
            CompressedImage,
            self.get_parameter('input_topic').value,
            self.compressed_image_callback,
            10
        )
        self.publisher = self.create_publisher(
            Image,
            self.get_parameter('output_topic').value,
            10
        )

        self.get_logger().info(
            f"Subscribed to '{self.subscription.topic_name}', "
            f"publishing to '{self.publisher.topic_name}'"
        )

        intrinsic_matrix = [
            537.0, 0.0, 476.30553261,
            0.0, 532.9, 272.7,
            0.0, 0.0, 1.0
        ]
        
        distortion_coeffs = [
            -0.08050122007374724,
            0.06195116455291804,
            -9.05187506022231e-05,
            0.0023246746346194213,
            0.0
        ]
        
        # Initialize undistorter (image size auto-detected on first call)
        self.undistorter = CameraUndistorter(
            intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=distortion_coeffs,
            optimize_fov=0.8  # Preserve more of the field of view
        )

    def compressed_image_callback(self, msg: CompressedImage):
        try:
            # Decompress JPEG/PNG image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error('Failed to decode compressed image')
                return

            undistorted = self.undistorter(cv_image)
            
            image_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
            image_msg.header = msg.header  # Preserve timestamp/frame_id

            # Publish
            self.publisher.publish(image_msg)
            # cv2.imwrite(f"/tmp/speed_measure_tmp/dataset/{datetime.datetime.now()}.png", cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = UndistortImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()