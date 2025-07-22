import glob
import pathlib
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

import git

repo_root = pathlib.Path(git.Repo(".", search_parent_directories=True).working_tree_dir)

data_root = repo_root / "data"

robot_desc_root = (
    repo_root / "gum" / "devices" / "metahand" / "ros" / "meta_hand_description"
)


def priv_info_dict_from_config(
    priv_config: Dict[str, Any]
) -> Dict[str, Tuple[int, int]]:
    priv_dims = OrderedDict()
    priv_dims["net_contact"] = priv_config["contact_input_dim"]
    priv_dims["obj_orientation"] = 4
    priv_dims["obj_linvel"] = 3
    priv_dims["obj_angvel"] = 3
    priv_dims["fingertip_position"] = 3 * 4
    priv_dims["fingertip_orientation"] = 4 * 4
    priv_dims["fingertip_linvel"] = 4 * 3
    priv_dims["fingertip_angvel"] = 4 * 3
    priv_dims["hand_scale"] = 1
    priv_dims["obj_restitution"] = 1

    priv_info_dict = {
        "obj_position": (0, 3),
        "obj_scale": (3, 4),
        "obj_mass": (4, 5),
        "obj_friction": (5, 6),
        "obj_com": (6, 9),
        "obj_end_point": (10, 16)
    }
    start_index = 9
    for name, dim in priv_dims.items():
        # (lep) Address naming incosistencies w/o changing the rest of the code
        config_key = f"enable_{name}".replace("fingertip", "ft")
        config_key = config_key.replace("position", "pos")
        config_key = config_key.replace("_net_contact", "NetContactF")
        if priv_config[config_key]:
            priv_info_dict[name] = (start_index, start_index + dim)
            start_index += dim
    return priv_info_dict


def priv_info_dim_from_dict(priv_info_dict: Dict[str, Tuple[int, int]]) -> int:
    return max([v[1] for k, v in priv_info_dict.items()])


def get_object_subtype_info(
    data_root: pathlib.Path,
    object_type: str,
    raw_prob: Sequence[float],
    max_num_objects: Optional[int] = None,
):
    assert sum(raw_prob) == 1

    primitive_list = object_type.split("+")
    object_type_prob = []
    object_subtype_list = []
    asset_files_dict = {
        "simple_tennis_ball": "ball.urdf",
        "simple_cube": "cube.urdf",
        "simple_cylin4cube": "cylinder4cube.urdf",
        "rolling_pin": "rolling_pin.urdf",
        "65mm_cylin": "65mm_cylinder.urdf",
        "cylinder": "cylinder.urdf",
    }
    assets_root = data_root / "assets"
    for p_id, prim in enumerate(primitive_list):
        if "cuboid" in prim or "hora_cylinder" in prim or "realshape" in prim:
            prim_name, subset_name = prim.split("/")
            urdf_pattern = str(
                assets_root / f"{prim_name}" / f"{subset_name}" / "*.urdf"
            )
            urdf_filenames = sorted(glob.glob(urdf_pattern))
            object_subtype_list_for_prim = []
            for i, filepath in enumerate(urdf_filenames):
                if i == max_num_objects:
                    break
                # given filepath = "/path/to/file/my_object.urdf"
                # then fname = "my_object"
                object_name = filepath.split("/")[-1].split(".")[0]
                asset_files_dict[object_name] = filepath.split("assets/")[1]
                object_subtype_list_for_prim.append(object_name)
            object_subtype_list += object_subtype_list_for_prim
            object_type_prob += [
                raw_prob[p_id] / len(object_subtype_list_for_prim)
                for _ in object_subtype_list_for_prim
            ]
        else:
            object_subtype_list += [prim]
            object_type_prob += [raw_prob[p_id]]

    return object_subtype_list, object_type_prob, asset_files_dict
