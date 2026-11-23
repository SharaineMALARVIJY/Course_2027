import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Official Intel RealSense ROS 2 launch file
    realsense_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    # D435i configuration for debugging
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'enable_sync': 'true',
        }.items()
    )

    return LaunchDescription([realsense])