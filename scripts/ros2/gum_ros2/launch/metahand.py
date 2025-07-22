import os
import sys
import pathlib
from hydra import compose, initialize

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.descriptions import ParameterValue
from launch.substitutions import Command
from launch_ros.actions import Node

from ament_index_python import get_package_share_directory

pkg_root = pathlib.Path(__file__).resolve().parents[1]

def generate_launch_description():

    # config_file = None
    # for arg in sys.argv:
    #     if arg.startswith("metahand_config_file:="):
    #         config_file = arg.split(":=")[1]
    # if config_file is None:
    #     raise Exception("No config file specified.")

    # # Load metahand config    
    # initialize(config_path="../config")
    # cfg = compose(config_name=config_file, overrides=['hydra.output_subdir=null','hydra.run.dir=.'])

    reskin_raw_data = ExecuteProcess(
        cmd=[
            f"python {pkg_root}/reskin_raw_data.py --config-name {config_file}",
        ],
        shell=True,
        output='screen',
    )

    config = os.path.dirname(os.path.abspath(__file__)) + '/allegro_node_params.yaml'
    allegro = Node(
        package='allegro_hand_controllers',
        executable='allegro_node_pd',
        name='allegro',
        parameters = [config]
    )

    metahand = Node(
        package='gum_ros2',
        executable='metahand.py',
        name='metahand'
    )

    meta_hand_desc_dir = get_package_share_directory('meta_hand_description')

    asset_server = ExecuteProcess(
        cmd=[
            f"python -m http.server 8766 --directory {meta_hand_desc_dir}",
        ],
        shell=True,
        output='screen',
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', f"{meta_hand_desc_dir}/robots/meta_hand.urdf.xacro side:=right fingertip:=digitv1 sphere_fingertip_coll:=true joint_postfix:='.0' asset_prefix:=http://localhost:8766"]), value_type=str

            )
        }],
        remappings=[
            ("joint_states", "allegroHand/joint_states")
        ],
    )

    reskin_binary = Node(
        package = "gum_ros2",
        executable = "reskin_binary.py",
        name = "reskin_binary"
    )


    return LaunchDescription([
        reskin_raw_data,
        allegro,
        metahand,
        asset_server,
        robot_state_publisher_node,
        reskin_binary,
    ])
