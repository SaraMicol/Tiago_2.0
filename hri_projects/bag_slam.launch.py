import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    bag_path = LaunchConfiguration('bag_path')
    rviz_config = LaunchConfiguration('rviz_config')
    slam_params = LaunchConfiguration('slam_params')

    declare_bag_path = DeclareLaunchArgument(
        'bag_path',
        default_value='/root/exchange/exchange/sara_bag',
        description='Path to the ROS2 bag'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value='/root/exchange/exchange/rviz/default.rviz',
        description='Path to RViz config'
    )

    declare_slam_params = DeclareLaunchArgument(
        'slam_params',
        default_value='/root/tiago_public_ws/src/pmb2_navigation/pmb2_2dnav/config/nav_public_sim.yaml',
        description='Path to SLAM params'
    )

    play_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag_path,
            '--clock',
            '--rate', '0.3',
            '--topics',
            '/scan', '/tf', '/tf_static',
            '/head_front_camera/rgb/image_raw',
            '/head_front_camera/depth/image_raw',
            '/head_front_camera/depth/camera_info',
            '/head_front_camera/rgb/camera_info',
            '/joint_states',
            '/robot_description',
            '/mobile_base_controller/odom',
        ],
        output='screen'
    )

    slam_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[
                    slam_params,
                    {'use_sim_time': True},
                    {'scan_topic': '/scan'},
                ],
            )
        ]
    )

    rviz_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
                parameters=[{'use_sim_time': True}],
            )
        ]
    )

    return LaunchDescription([
        declare_bag_path,
        declare_rviz_config,
        declare_slam_params,
        play_bag,
        slam_node,
        rviz_node,
    ])
