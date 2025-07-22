from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from isaacgym.torch_utils import (
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_unit,
)

# isort: off
import torch

# isort: on

TensorContainer = TypeVar(
    "TensorContainer",
    torch.Tensor,
    Sequence[Optional[torch.Tensor]],
    Dict[Any, Optional[torch.Tensor]],
)


def random_choose(
    tensor: torch.Tensor, dim: int = 0, rng: Optional[np.random.Generator] = None
) -> Tuple[torch.Tensor, int]:
    sz = tensor.shape[dim]
    idx = np.random.randint(sz) if rng is None else rng.integers(0, sz)
    return tensor.select(dim, idx), idx


# Typing will probably complain about nested containers, and I'm not even
# sure if mypy supports this
# type ignore is your friend if you hit this case
def all_to(tensors: TensorContainer, *args, **kwargs) -> TensorContainer:
    """Recursively applies tensor.to() to all tensors in a container.

    Args:
        tensors (TensorContainer): a (nested) list/tuple/dict of
            tensors or None.

    Raises:
        ValueError: if not a list/tuple/dict and if leaves are
            not None or tensor.

    Returns:
        TensorContainer: the same container replacing leaf tensors with
            tensor.to(*args, **kwargs).
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(*args, **kwargs)

    def _check_sequence(t_seq: Sequence[Optional[torch.Tensor]]) -> bool:
        return all(
            isinstance(t, (torch.Tensor, list, tuple, dict))
            for t in t_seq
            if t is not None
        )

    if isinstance(tensors, (list, tuple)):
        good = _check_sequence(tensors)
    elif isinstance(tensors, dict):
        good = _check_sequence(tensors.values())
    else:
        good = False

    if not good:
        raise ValueError(
            "Incorrect input to all_to(). "
            "Expected a tensor, a (nested) list/tuple of optional(tensor), "
            "or a (nested) dict[any, optional(tensor)]"
        )

    def _to_helper(
        t: Optional[torch.Tensor], *args, **kwargs
    ) -> Optional[torch.Tensor]:
        assert t is None or isinstance(t, (torch.Tensor, list, tuple, dict))
        return t if t is None else all_to(t, *args, **kwargs)

    if isinstance(tensors, (list, tuple)):
        return type(tensors)(_to_helper(t, *args, **kwargs) for t in tensors)
    # otherwise a dict
    return {k: _to_helper(t, *args, **kwargs) for k, t in tensors.items()}


def noisy_pos_rot(
    pos: torch.Tensor,
    rot: torch.Tensor,
    noise_pos_scale: float = 0.0,
    noise_rpy_scale: float = 0.0,
    noise_pos_bias: Union[float, torch.Tensor] = 0.0,
    noise_rpy_bias: Union[float, torch.Tensor] = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert pos.ndim == 2 and rot.ndim == 2 and pos.shape[1] == 3 and rot.shape[1] == 4
    size = (pos.shape[0], 3)
    rand_rpy = noise_rpy_bias + torch.normal(
        0, noise_rpy_scale, size=size, device=rot.device, dtype=rot.dtype
    )
    rand_quat = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
    noisy_rot = quat_mul(rand_quat, rot)
    noisy_pos = (
        torch.normal(0, noise_pos_scale, size=size, device=pos.device, dtype=pos.dtype)
        + noise_pos_bias
        + pos
    )
    return noisy_pos, noisy_rot


def noisy_quat_from_pos_rot(
    pos: torch.Tensor,
    rot: torch.Tensor,
    noise_pos_bias: Union[float, torch.Tensor] = 0.0,
    noise_rpy_bias: Union[float, torch.Tensor] = 0.0,
    noise_pos_scale: float = 0.0,
    noise_rpy_scale: float = 0.0,
) -> torch.Tensor:
    """Converts position and rotation tensors to a 7D pose array with optional noise.

    Unconventionally, returns rotation first then xyz position (as used in the
    AllegroHandHora environment).

    Args:
        pos (torch.Tensor): position tensor.
        rot (torch.Tensor): rotation tensor.
        noise_pos_bias (Union[float, torch.Tensor], optional):
            position noise bias. Defaults to 0.0.
        noise_rpy_bias (Union[float, torch.Tensor], optional):
            rotation noise bias. Defaults to 0.0.
        noise_pos_scale (float, optional): position noise scale. Defaults to 0.0.
        noise_rpy_scale (float, optional): rotation noise scale. Defaults to 0.0.

    Returns:
        torch.Tensor: a (N, 7) pose array in the format [xr, yr, zr, wr, xt, yt, zt].
    """
    noisy_pos, noisy_rot = noisy_pos_rot(
        pos,
        rot,
        noise_pos_scale=noise_pos_scale,
        noise_rpy_scale=noise_rpy_scale,
        noise_pos_bias=noise_pos_bias,
        noise_rpy_bias=noise_rpy_bias,
    )
    return torch.cat([noisy_rot, noisy_pos], dim=-1)


def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_axis_angle

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., 3:])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., :3] / sin_half_angles_over_angles


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Adapted from PyTorch3D:
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_matrix

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def to_world_frame(
    points: torch.Tensor, object_rot: torch.Tensor, object_pos: torch.Tensor
) -> torch.Tensor:
    assert points.ndim == 3, "points tensor must be (B, nPoints, 3)"
    assert object_rot.ndim == 2, "object rot must be (B, 4)"
    assert object_pos.ndim == 2, "object pos must be (B, 3)"
    return (
        quat_apply(
            object_rot[:, None].repeat(1, points.shape[1], 1),
            points,
        )
        + object_pos[:, None]
    )


def from_world_frame(
    points: torch.Tensor,
    frame_rot: torch.Tensor,
    frame_pos: torch.Tensor,
    quaternions: Optional[torch.Tensor] = None,
    quat_first: bool = False,
) -> torch.Tensor:
    """Converts points and (optional) quaternions from world frame to a given frame.

    Args:
        points (torch.Tensor): a (B, nPoints, 3)-shaped tensor with xyz.
        frame_rot (torch.Tensor): the rotation quaternion for the reference frame,
            with shape (B, 4).
        frame_pos (torch.Tensor): the xyzposition of the reference frame, with
            shape (B, 3).
        quaternions (Optional[torch.Tensor], optional): (B, nPoints, 4)-shaped
            tensor with rotations to convert. Defaults to None.
        quat_first (bool): if True and quaternions is given, the return tensor
            has xyzw quaternion before xyz position. If false, xyz then xyzw.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: the transformed
            points and (optional) quaternions. If only points, the return has shape
            (B, nPoints, 3). Otherwise it is (B, nPoints, 7).
    """
    assert points.ndim == 3, "points tensor must be (B, nPoints, 3)"
    assert frame_rot.ndim == 2, "object rot must be (B, 4)"
    assert frame_pos.ndim == 2, "object pos must be (B, 3)"
    quat_conj = quat_conjugate(frame_rot)
    inv_quat = quat_unit(quat_conj)[:, None].repeat(1, points.shape[1], 1)
    points__frame = quat_apply(
        inv_quat,
        points - frame_pos[:, None],
    )
    if quaternions is None:
        return points__frame

    assert quaternions.ndim == 3, "points tensor must be (B, nPoints, 4)"
    quat__frame = quat_mul(
        inv_quat,
        quaternions,
    )
    ret_list = (
        [quat__frame, points__frame] if quat_first else [points__frame, quat__frame]
    )
    return torch.cat(ret_list, dim=-1)
