import os
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    current_directory = Path.cwd().parent
    print(current_directory)

    img_decompress_node = Node(
        package="speed_measure",
        name="image_undistort_node",
        executable="image_undistort_node",
        parameters=[
            {
                "input_topic": "/apelsin/compressed", 
                "output_topic": "/apelsin/raw", 
            }
        ]
    )

    # compass_filter_node = Node(
    #     package="speed_measure",
    #     name="compass_filter_node",
    #     executable="median_filter_node",
    #     parameters=[
    #         {
    #             "input_topic": "/mavros/global_position/compass_hdg", 
    #             "output_topic": "/mavros/global_position/compass_hdg_filtered",
    #             "value_type": "float64" 
    #         }
    #     ]
    # )
    
    alt_filter_node = Node(
        package="speed_measure",
        name="compass_filter_node",
        executable="median_filter_node",
        parameters=[
            {
                "input_topic": "/apelsin/altitude_estimation", 
                "output_topic": "/apelsin/altitude_estimation_filtered",
                "value_type": "float32",
                "window_size": 7
            }
        ]
    )

    optical_flow_node = Node(
        package="speed_measure",
        name="optical_flow_node",
        executable="optical_flow_node",
        parameters=[
            {
                "input_topic": "/apelsin/raw", 
                "alt_topic": "/apelsin/altitude_estimation_filtered",
                "compass_topic": "/mavros/global_position/compass_hdg",
                "output_topic": "/opt_flow", 
            }
        ]
    )

    return LaunchDescription([
        img_decompress_node, 
        optical_flow_node,
        alt_filter_node,
        # compass_filter_node
    ])
