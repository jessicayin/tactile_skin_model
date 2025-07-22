## Initial ROS2 Setup

### Install ROS Conda packages
This install basic ROS and it's dependencies
```
./install_metahand_env.sh -e _metahand
mamba activate _metahand
```

### Build GUM ROS packages distributed in source
Pick a place for `ros_ws`, this can be anywhere
```
mkdir ros_ws
cd ros_ws
../colcon_build.sh
source ./install/setup.bash
ros2 launch gum_ros2 metahand.py
```
## Development
### Making changes to the gum_ros2 package

If you make changes to `CMakeLists.txt`, you will need to rebuild with colcon:
```
cd $HOME/tactile-iht/
.scripts/ros2/colcon_build.sh
```
## ReSkin ROS2 Nodes

```/reskin/reskin_raw_data``` outputs the raw magnetometer values in a 48-dim array representing 16 taxels, each with 3-axes. 
```/reskin/reskin_binary``` outputs the filtered binary ReSkin values used by the policy. Here, we implement a sliding window filter based on the time-derivative of the sensor values.

```
ros2 launch gum_ros2 metahand.py
```
