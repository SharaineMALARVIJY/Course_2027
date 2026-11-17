import os

import launch
from launch import LaunchDescription
from launch.actions import (
    RegisterEventHandler,
    EmitEvent,
    DeclareLaunchArgument,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController


def make_webots_controller(robot_name, urdf_path, bridge_node_name,
                           params_path=None, condition=None):
    parameters = [
        {'robot_description': urdf_path},
    ]

    if params_path is not None:
        parameters.append(params_path)

    return WebotsController(
        robot_name=robot_name,
        parameters=parameters,
        remappings=[
            ('__node', bridge_node_name),
        ],
        respawn=True,
        condition=condition,
    )


def make_robot_state_publisher(robot_name, urdf_path, condition=None):
    with open(urdf_path, 'r') as file:
        robot_description = file.read()

    return Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name=f'{robot_name}_state_publisher',
        namespace=robot_name,
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': True},
        ],
        output='screen',
        condition=condition,
    )


def generate_launch_description():
    package_dir = get_package_share_directory('webots_sim')

    # ==================== ARGUMENTS ====================

    use_yellow = LaunchConfiguration('use_yellow')
    use_blue = LaunchConfiguration('use_blue')

    declare_use_yellow = DeclareLaunchArgument(
        'use_yellow',
        default_value='true',
        description='Lancer la voiture jaune'
    )

    declare_use_blue = DeclareLaunchArgument(
        'use_blue',
        default_value='false',
        description='Lancer la voiture bleue'
    )

    # ==================== PATHS ====================

    world_path = os.path.join(
        package_dir, 'worlds', 'Piste_CoVAPSy_2025a_rgbd.wbt'
    )

    yellow_urdf_path = os.path.join(
        package_dir, 'resource', 'TT02_jaune_rgbd.urdf'
    )

    yellow_driver_params_path = os.path.join(
        package_dir, 'config', 'yellow_driver_params.yaml'
    )

    blue_urdf_path = os.path.join(
        package_dir, 'resource', 'TT02_bleu.urdf'
    )

    # ==================== WEBOTS ====================

    webots = WebotsLauncher(
        world=world_path,
        ros2_supervisor=True,
        mode='realtime',
        gui=True,
    )

    # ==================== CONTROLLERS ====================

    yellow_webots_bridge = make_webots_controller(
        robot_name='yellow_car',
        urdf_path=yellow_urdf_path,
        bridge_node_name='yellow_webots_bridge',
        params_path=yellow_driver_params_path,
        condition=IfCondition(use_yellow),
    )

    blue_webots_bridge = make_webots_controller(
        robot_name='blue_car',
        urdf_path=blue_urdf_path,
        bridge_node_name='blue_webots_bridge',
        condition=IfCondition(use_blue),
    )

    # ==================== STATE PUBLISHERS ====================

    yellow_state_publisher = make_robot_state_publisher(
        robot_name='yellow_car',
        urdf_path=yellow_urdf_path,
        condition=IfCondition(use_yellow),
    )

    blue_state_publisher = make_robot_state_publisher(
        robot_name='blue_car',
        urdf_path=blue_urdf_path,
        condition=IfCondition(use_blue),
    )

    # ==================== AUTRES NŒUDS ====================

    front_scan_filter = Node(
        package='car_rosconfr',
        executable='front_scan_filter',
        name='front_scan_filter',
        output='screen',
    )

    # ==================== SHUTDOWN ====================

    shutdown_on_webots_exit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=webots,
            on_exit=[
                EmitEvent(event=launch.events.Shutdown())
            ],
        )
    )

    # ==================== LAUNCH ====================

    return LaunchDescription([
        declare_use_yellow,
        declare_use_blue,

        webots,
        webots._supervisor,

        yellow_webots_bridge,
        blue_webots_bridge,

        yellow_state_publisher,
        blue_state_publisher,

        front_scan_filter,

        shutdown_on_webots_exit,
    ])