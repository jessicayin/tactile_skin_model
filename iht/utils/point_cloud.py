import os
import pathlib
import random
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import trimesh
from sklearn.neighbors import KDTree

if TYPE_CHECKING:
    import torch


def sample_cylinder(h, num_points=100, num_circle_points=15, side_points=70, noise=0.0):
    # h = length/height of cylinder
    # assume that radius of cylinder is 1
    if num_points != 100:
        num_circle_points = int(0.15 * num_points)
        side_points = int(0.7 * num_points)
    assert num_points == num_circle_points * 2 + side_points
    pcs = np.zeros((num_points, 3))
    # sample 100 points from top and bottom surface
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[:num_circle_points, 0] = r * np.cos(theta) * 0.5
    pcs[:num_circle_points, 1] = r * np.sin(theta) * 0.5
    pcs[:num_circle_points, 2] = 0.5 * h
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[num_circle_points : num_circle_points * 2, 0] = r * np.cos(theta) * 0.5
    pcs[num_circle_points : num_circle_points * 2, 1] = r * np.sin(theta) * 0.5
    pcs[num_circle_points : num_circle_points * 2, 2] = -0.5 * h
    # sample 400 points from the side surface
    vec = np.random.random((side_points, 2)) - 0.5
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    vec *= 0.5
    pcs[num_circle_points * 2 :, :2] = vec
    pcs[num_circle_points * 2 :, 2] = h * (np.random.random(side_points) - 0.5)
    if noise:
        noise_for_pcs = np.random.normal(scale=noise, size=pcs.shape)
        noisy_pcs = pcs + noise_for_pcs
        return noisy_pcs
    return pcs


def sample_cuboid(s_x, s_y, s_z, num_points=100):
    # this function makes a few assumptions: center at (0, 0, 0)
    # side length is s_x, s_y, s_z
    pcs = np.zeros((num_points, 3))
    # assign number of points for each side because it may not divides 6
    idx = np.random.randint(0, 6, size=(num_points,))

    num_points_0 = sum(idx == 0)
    xs = np.random.uniform(0, s_x, size=(num_points_0, 1))
    ys = np.random.uniform(0, s_y, size=(num_points_0, 1))
    zs = np.zeros((num_points_0, 1))
    pcs_0 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_1 = sum(idx == 1)
    xs = np.random.uniform(0, s_x, size=(num_points_1, 1))
    ys = np.random.uniform(0, s_y, size=(num_points_1, 1))
    zs = np.ones((num_points_1, 1)) * s_z
    pcs_1 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_2 = sum(idx == 2)
    xs = np.random.uniform(0, s_x, size=(num_points_2, 1))
    ys = np.zeros((num_points_2, 1))
    zs = np.random.uniform(0, s_z, size=(num_points_2, 1))
    pcs_2 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_3 = sum(idx == 3)
    xs = np.random.uniform(0, s_x, size=(num_points_3, 1))
    ys = np.ones((num_points_3, 1)) * s_z
    zs = np.random.uniform(0, s_z, size=(num_points_3, 1))
    pcs_3 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_4 = sum(idx == 4)
    xs = np.zeros((num_points_4, 1))
    ys = np.random.uniform(0, s_y, size=(num_points_4, 1))
    zs = np.random.uniform(0, s_z, size=(num_points_4, 1))
    pcs_4 = np.concatenate([xs, ys, zs], axis=-1)

    num_points_5 = sum(idx == 5)
    xs = np.ones((num_points_5, 1)) * s_x
    ys = np.random.uniform(0, s_y, size=(num_points_5, 1))
    zs = np.random.uniform(0, s_z, size=(num_points_5, 1))
    pcs_5 = np.concatenate([xs, ys, zs], axis=-1)

    pcs[idx == 0] = pcs_0
    pcs[idx == 1] = pcs_1
    pcs[idx == 2] = pcs_2
    pcs[idx == 3] = pcs_3
    pcs[idx == 4] = pcs_4
    pcs[idx == 5] = pcs_5

    pcs[:, 0] -= s_x / 2
    pcs[:, 1] -= s_y / 2
    pcs[:, 2] -= s_z / 2

    return pcs


def sample_point_cloud_from_urdf(
    assets_root: Union[pathlib.Path, str],
    urdf_fname: str,
    sampled_dim: int,
    noise: float = 0.0,
) -> np.ndarray:
    """Computes a point cloud from mesh at the given folder."""

    assets_root = pathlib.Path(assets_root)
    urdf_path = pathlib.Path(urdf_fname)

    def read_xml(filename):
        import xml.etree.ElementTree as Et

        root = Et.parse(filename).getroot()
        return root
    import pdb; pdb.set_trace()
    src_urdf = read_xml(str(assets_root / urdf_fname))

    col_el = src_urdf.findall(f".//collision/geometry/")[0]
    if col_el.tag == "sphere":
        # Sample points uniformly from a sphere
        # (X, Y, Z) ~ N(0, 1) ---> (X/norm, Y/norm, Z/norm) ~ U(S^2)
        # see  https://mathworld.wolfram.com/SpherePointPicking.html
        # (and Muller 1959, Marsaglia 1972).

        radius = float(col_el.attrib["radius"].split(" ")[0])
        mesh = np.zeros((sampled_dim, 3), dtype=float)
        mesh_norm = np.linalg.norm(mesh, axis=1, keepdims=True)
        while True:
            resample = mesh_norm[:, 0] < 1e-8
            if not resample.any():
                break
            mesh[resample] = np.random.normal(
                loc=0.0, scale=1.0, size=(resample.shape[0], 3)
            )
            mesh_norm = np.linalg.norm(mesh, axis=1, keepdims=True)
        mesh = radius * mesh / mesh_norm
    else:
        scale = float(col_el.attrib["scale"].split(" ")[0])
        mesh_file = urdf_path.parent / col_el.attrib["filename"]
        mesh = trimesh.load(assets_root / mesh_file, force="mesh")
        mesh = np.array(trimesh.sample.sample_surface(mesh, sampled_dim)[0]) * scale
    if noise > 0.0:
        mesh += np.random.normal(scale=noise, size=mesh.shape)
    return mesh


def sample_point_cloud_regions_from_urdf(
    assets_root: Union[str, pathlib.Path],
    urdf_fname: str,
    sampled_dim: int,
    num_clouds: int,
    cloud_size_limits: List[int],
    noise: float = 0.0,
) -> List[np.ndarray]:
    import torch  # lazy to prevent isaacgym import errors

    mesh = sample_point_cloud_from_urdf(
        assets_root=assets_root,
        urdf_fname=urdf_fname,
        sampled_dim=sampled_dim,
        noise=noise,
    )
    assert (
        len(cloud_size_limits) == 2
        and cloud_size_limits[0] > 0
        and cloud_size_limits[1] < mesh.shape[0]
    )

    sample_clouds = []
    tree = KDTree(mesh)

    for _ in range(num_clouds):
        query_idx = np.random.randint(mesh.shape[0])
        query = mesh[query_idx : query_idx + 1]
        cloud_size = np.random.randint(cloud_size_limits[0], cloud_size_limits[1] + 1)
        _, ind = tree.query(query, k=cloud_size)
        sample_clouds.append(torch.from_numpy(mesh[ind].copy()))
    return sample_clouds


def load_point_clouds(
    point_cloud_dir: Union[str, pathlib.Path],
    sampled_dim: int,
    max: Optional[int] = None,
    order: str = "sorted",
    scale: float = 1.0,
) -> List["torch.Tensor"]:
    import torch  # lazy to prevent isaacgym import errors

    point_cloud_dir = pathlib.Path(point_cloud_dir)
    pcs = []
    files = os.listdir(point_cloud_dir)
    if order == "sorted":
        files = sorted(files)
    elif order == "shuffle":
        random.shuffle(files)
    else:
        raise ValueError("Order must be one of 'sorted' or 'shuffle'.")
    if len(files) == 0:
        raise RuntimeError(f"No point cloud files in directory {point_cloud_dir}.")
    for i, f in enumerate(files):
        if i == max:
            break
        pc = torch.load(point_cloud_dir / f)
        sz = min(sampled_dim, pc.shape[0])
        if sz < pc.shape[0]:
            pc = pc[np.random.choice(pc.shape[0], sz, replace=False), :]
        pcs.append(pc * scale)
    return pcs


def _chamfer_base(x: np.ndarray, y: np.ndarray) -> float:
    kd_tree_y = KDTree(y, p=1)
    return np.mean(kd_tree_y.query(x)[0])


# TODO(lep): This is based on part of
# gum.labs.tac_neural.eval.chamfer.compute_trimesh_chamfer().
# We should extract this to a common util.
def chamfer(
    x: np.ndarray, y: np.ndarray, reverse: bool = False, bidirectional: bool = True
) -> float:
    if bidirectional:
        return _chamfer_base(x, y) + _chamfer_base(y, x)
    return _chamfer_base(y, x) if reverse else _chamfer_base(x, y)


# TODO(lep, joe, suddhu): Try to unify into a shared util with tac_neural
def depth_to_cloud(
    depth,
    width: float,
    height: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device,
    fourth_dim=False,
):
    import torch

    # Adapted from https://github.com/facebookresearch/GUM/blob/main/gum/labs/tac_neural/geometry/transform.py
    u = torch.arange(0, width, device=device)
    v = torch.arange(0, height, device=device)
    v2, u2 = torch.meshgrid(v, u, indexing="ij")

    Z = depth
    # print('u {} c {} f {} Z {}'.format(v2.shape,cy.shape,fy.shape,Z.shape))

    X = -(u2[None, :, :] - cx[:, None, None]) / fx[:, None, None] * Z
    Y = (v2[None, :, :] - cy[:, None, None]) / fy[:, None, None] * Z

    X = torch.flatten(X, start_dim=1)
    Y = torch.flatten(Y, start_dim=1)
    Z = torch.flatten(Z, start_dim=1)

    sample_prob = (Z <= -1e-8).float() + 1e-8
    sample_prob = sample_prob / (sample_prob.sum(dim=-1).unsqueeze(1))

    if fourth_dim:
        ones = torch.ones_like(X)
        points = torch.stack((X, Y, Z, ones), dim=-1)
    else:
        points = torch.stack((X, Y, Z), dim=-1)

    return points, sample_prob
