# RGB-D Perception

ROS 2 Jazzy package for processing RGB-D camera data.

The goal is to develop perception algorithms that work with both:

- **Intel RealSense D435i** — Real camera
- **Webots RGB-D camera** — Simulated camera

The same perception code should work with both cameras without modification.

## Installation

From the workspace root:

```bash
cd ~/course_2027_ws

source /opt/ros/jazzy/setup.bash

colcon build --symlink-install --packages-select rgbd_perception

source install/setup.bash
```

## Point Cloud Test

The `test_pointcloud` node is a simple diagnostic tool that estimates the distance to points detected in front of the camera.

It subscribes to a `sensor_msgs/msg/PointCloud2` topic and uses TF2 to transform the point cloud into the `camera_link` coordinate frame.

The node uses the same processing pipeline for real and simulated data.

### Test 1: Webots Simulation

Start the simulation:

```bash
ros2 launch webots_sim webots_sim_rgbd.launch.py \
    use_yellow:=true \
    use_blue:=false
```

Open another terminal and run:

```bash
source /opt/ros/jazzy/setup.bash
source ~/course_2027_ws/install/setup.bash

ros2 run rgbd_perception test_pointcloud \
    --ros-args \
    -p use_sim_time:=true \
    -p cloud_topic:=/yellow_car/camera/colored_points
```

The node should display distance estimates in meters.

### Test 2: Intel RealSense D435i

Stop the Webots simulation before starting the real camera.

Connect the D435i and launch its ROS 2 driver:

```bash
ros2 launch car_rosconfr realsense_debug.launch.py
```

Open another terminal and run:

```bash
source /opt/ros/jazzy/setup.bash
source ~/course_2027_ws/install/setup.bash

ros2 run rgbd_perception test_pointcloud \
    --ros-args \
    -p cloud_topic:=/camera/camera/depth/color/points
```

Place an object in front of the camera and observe the estimated distance.

## Coordinate Frames

The two cameras publish their point clouds in different coordinate frames.

**Webots:**

`camera_link`

- X: Forward
- Y: Left
- Z: Up

**Intel RealSense D435i:**

`camera_depth_optical_frame`

- X: Right
- Y: Down
- Z: Forward

The node uses TF2 to transform both point clouds into `camera_link`.

## Algorithm

The test performs the following operations:

1. Subscribe to a PointCloud2 topic.
2. Extract the XYZ coordinates.
3. Remove invalid points.
4. Transform points into `camera_link` using TF2.
5. Filter points within a predefined region in front of the camera.
6. Estimate the longitudinal distance using the 10th percentile.
7. Display the result in the terminal.

**Note:** This is a basic diagnostic tool, not a complete obstacle detection algorithm. It does not identify individual objects, and its reported distance is not necessarily the distance to the nearest obstacle.

## Objective

The purpose of this package is to centralize camera perception algorithms and simplify Sim-to-Real development.

Future algorithms should remain independent of the camera source whenever possible.