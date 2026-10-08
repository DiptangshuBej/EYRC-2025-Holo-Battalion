#!/usr/bin/env python3
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    pkg_name = 'hb_description'
    pkg_share = get_package_share_directory(pkg_name)

    xacro_file = os.path.join(
        pkg_share,
        'models',
        'holonomic_bot',
        'hb_crystal.xacro'
    )

    robot_desc = xacro.process_file(xacro_file).toxml()

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true'
        ),

        # 1. Start Ignition Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'empty.sdf'],
            output='screen'
        ),

        # 2. Publish robot_description
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace='crystal',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc
            }]
        ),

        # 3. Spawn robot into Ignition
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'crystal_bot',
                '-topic', 'robot_description'
            ],
            output='screen'
        ),

        # 4. Bridge (example topics)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
            ],
            output='screen'
        ),
    ])
