import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from arguments import ModelParams, get_combined_args

QUATERNION_NORM_EPSILON = 1e-12
COVARIANCE_REGULARIZATION = 1e-8


def collect_mask_files(mask_path: Path):
    if mask_path.is_file():
        return [mask_path]
    if mask_path.is_dir():
        return sorted(mask_path.glob("*_gaus_mask.pt"))
    raise FileNotFoundError(f"Mask path does not exist: {mask_path}")


def as_float_list(arr):
    if torch.is_tensor(arr):
        return [float(x) for x in arr.reshape(-1).tolist()]
    return [float(x) for x in arr]


def matrix3_to_list(mat):
    if torch.is_tensor(mat):
        mat = mat.reshape(3, 3).tolist()
    return [[float(x) for x in row] for row in mat]


def rotation_matrix_to_quaternion_xyzw(rot):
    if not torch.is_tensor(rot):
        rot = torch.tensor(rot, dtype=torch.float64)
    r = rot.to(dtype=torch.float64).reshape(3, 3)
    trace = float(torch.trace(r).item())
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = float((r[2, 1] - r[1, 2]).item()) / s
        y = float((r[0, 2] - r[2, 0]).item()) / s
        z = float((r[1, 0] - r[0, 1]).item()) / s
    elif float(r[0, 0].item()) > float(r[1, 1].item()) and float(r[0, 0].item()) > float(r[2, 2].item()):
        s = math.sqrt(1.0 + float(r[0, 0].item()) - float(r[1, 1].item()) - float(r[2, 2].item())) * 2.0
        w = float((r[2, 1] - r[1, 2]).item()) / s
        x = 0.25 * s
        y = float((r[0, 1] + r[1, 0]).item()) / s
        z = float((r[0, 2] + r[2, 0]).item()) / s
    elif float(r[1, 1].item()) > float(r[2, 2].item()):
        s = math.sqrt(1.0 + float(r[1, 1].item()) - float(r[0, 0].item()) - float(r[2, 2].item())) * 2.0
        w = float((r[0, 2] - r[2, 0]).item()) / s
        x = float((r[0, 1] + r[1, 0]).item()) / s
        y = 0.25 * s
        z = float((r[1, 2] + r[2, 1]).item()) / s
    else:
        s = math.sqrt(1.0 + float(r[2, 2].item()) - float(r[0, 0].item()) - float(r[1, 1].item())) * 2.0
        w = float((r[1, 0] - r[0, 1]).item()) / s
        x = float((r[0, 2] + r[2, 0]).item()) / s
        y = float((r[1, 2] + r[2, 1]).item()) / s
        z = 0.25 * s
    quaternion = torch.tensor([x, y, z, w], dtype=torch.float64)
    n = float(torch.linalg.norm(quaternion).item())
    if n < QUATERNION_NORM_EPSILON:
        return [0.0, 0.0, 0.0, 1.0]
    quaternion = quaternion / n
    return as_float_list(quaternion)


def compute_aabb(points_xyz):
    mins = points_xyz.min(dim=0).values
    maxs = points_xyz.max(dim=0).values
    center = (mins + maxs) / 2.0
    size = maxs - mins
    return {
        "min": as_float_list(mins),
        "max": as_float_list(maxs),
        "center": as_float_list(center),
        "size": as_float_list(size),
    }


def compute_obb(points_xyz):
    mean = points_xyz.mean(dim=0)
    centered = points_xyz - mean

    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    cov = cov + torch.eye(3, dtype=torch.float64) * COVARIANCE_REGULARIZATION
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    rot = eigvecs[:, order]

    if torch.det(rot).item() < 0:
        rot[:, 2] *= -1.0

    local = centered @ rot
    local_min = local.min(dim=0).values
    local_max = local.max(dim=0).values
    local_center = (local_min + local_max) / 2.0
    extents = (local_max - local_min) / 2.0

    center_world = mean + local_center @ rot.T
    size = extents * 2.0

    return {
        "center": as_float_list(center_world),
        "extents": as_float_list(extents),
        "size": as_float_list(size),
        "rotation_matrix": matrix3_to_list(rot),
        "quaternion_xyzw": rotation_matrix_to_quaternion_xyzw(rot),
    }


def to_unity_left_handed(aabb, obb):
    coordinate_flip = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))

    aabb_min = torch.tensor(aabb["min"], dtype=torch.float64)
    aabb_max = torch.tensor(aabb["max"], dtype=torch.float64)
    min_lhs = torch.tensor([aabb_min[0], aabb_min[1], -aabb_max[2]], dtype=torch.float64)
    max_lhs = torch.tensor([aabb_max[0], aabb_max[1], -aabb_min[2]], dtype=torch.float64)
    center_lhs = (min_lhs + max_lhs) / 2.0
    size_lhs = max_lhs - min_lhs

    center = torch.tensor(obb["center"], dtype=torch.float64)
    rot = torch.tensor(obb["rotation_matrix"], dtype=torch.float64)
    extents = torch.tensor(obb["extents"], dtype=torch.float64)

    center_lhs_obb = coordinate_flip @ center
    rot_lhs = coordinate_flip @ rot @ coordinate_flip

    return {
        "aabb": {
            "min": as_float_list(min_lhs),
            "max": as_float_list(max_lhs),
            "center": as_float_list(center_lhs),
            "size": as_float_list(size_lhs),
        },
        "obb": {
            "center": as_float_list(center_lhs_obb),
            "extents": as_float_list(extents),
            "size": as_float_list(extents * 2.0),
            "rotation_matrix": matrix3_to_list(rot_lhs),
            "quaternion_xyzw": rotation_matrix_to_quaternion_xyzw(rot_lhs),
        },
    }


def extract_xyz_from_checkpoint_model(model_params):
    if not isinstance(model_params, (tuple, list)) or len(model_params) < 2:
        raise ValueError(
            "Invalid checkpoint model params format: expected tuple/list with xyz at index 1."
        )
    xyz = model_params[1]
    if not torch.is_tensor(xyz):
        raise ValueError(
            f"Invalid xyz type in checkpoint model params: expected torch.Tensor, got {type(xyz)}."
        )
    xyz = xyz.detach().cpu().to(dtype=torch.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid xyz shape in checkpoint model params: got {tuple(xyz.shape)}.")
    return xyz


def main():
    parser = ArgumentParser(description="Export object-level AABB/OBB from Gaussian masks")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--start_checkpoint", required=True, type=str)
    parser.add_argument("--gaus_mask_path", default=None, type=str,
                        help="Path to one *_gaus_mask.pt file or a directory containing mask files")
    parser.add_argument("--save_path", default=None, type=str,
                        help="Output bbox JSON path, default: ${model_path}/objects_bbox.json")
    parser.add_argument("--min_points", default=16, type=int,
                        help="Skip exporting objects with fewer gaussian points than this threshold")
    args = get_combined_args(parser)

    dataset = model.extract(args)

    if args.gaus_mask_path is None and not dataset.model_path:
        raise ValueError("model_path is required when --gaus_mask_path is not provided")

    checkpoint = torch.load(args.start_checkpoint, map_location="cpu")
    model_params, _iter_step = checkpoint
    xyz = extract_xyz_from_checkpoint_model(model_params)
    gaus_num = xyz.shape[0]

    mask_path = Path(args.gaus_mask_path or os.path.join(dataset.model_path, "objects"))
    save_path = Path(args.save_path or os.path.join(dataset.model_path, "objects_bbox.json"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mask_files = collect_mask_files(mask_path)
    if len(mask_files) == 0:
        raise FileNotFoundError(
            f"No '*_gaus_mask.pt' files found in directory: {mask_path}. "
            "Please run eval_3d with --save_gaus_mask first, or pass --gaus_mask_path."
        )

    output = {
        "version": 1,
        "model_path": dataset.model_path,
        "source_path": dataset.source_path,
        "start_checkpoint": args.start_checkpoint,
        "coordinate": {
            "raw": "right_handed",
            "unity": "left_handed_z_flipped"
        },
        "objects": [],
    }

    for mask_file in mask_files:
        gaus_mask = torch.load(mask_file, map_location="cpu")
        if not torch.is_tensor(gaus_mask):
            print(f"[Skip] {mask_file}: loaded object is not a tensor (type={type(gaus_mask)})")
            continue

        gaus_mask = gaus_mask.bool().reshape(-1)
        if gaus_mask.numel() != gaus_num:
            print(f"[Skip] {mask_file}: mask size {gaus_mask.numel()} != gaussian size {gaus_num}")
            continue

        points = xyz[gaus_mask]
        if points.shape[0] < args.min_points:
            print(f"[Skip] {mask_file}: too few points ({points.shape[0]} < {args.min_points})")
            continue

        aabb = compute_aabb(points)
        obb = compute_obb(points)
        unity = to_unity_left_handed(aabb, obb)

        object_name = mask_file.name.replace("_gaus_mask.pt", "")
        output["objects"].append({
            "object_name": object_name,
            "mask_file": str(mask_file),
            "gaussian_count": int(points.shape[0]),
            "aabb": aabb,
            "obb": obb,
            "unity": unity,
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. Exported {len(output['objects'])} object bbox entries to: {save_path}")


if __name__ == "__main__":
    main()
