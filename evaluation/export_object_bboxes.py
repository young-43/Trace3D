import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from arguments import ModelParams, get_combined_args
from scene import GaussianModel


def collect_mask_files(mask_path: Path):
    if mask_path.is_file():
        return [mask_path]
    if mask_path.is_dir():
        return sorted(mask_path.glob("*_gaus_mask.pt"))
    raise FileNotFoundError(f"Mask path does not exist: {mask_path}")


def as_float_list(arr):
    return [float(x) for x in np.asarray(arr).reshape(-1).tolist()]


def matrix3_to_list(mat):
    return [[float(x) for x in row] for row in np.asarray(mat).reshape(3, 3).tolist()]


def rotation_matrix_to_quaternion_xyzw(rot):
    r = np.asarray(rot, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    q = q / n
    return as_float_list(q)


def compute_aabb(points_xyz):
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    size = maxs - mins
    return {
        "min": as_float_list(mins),
        "max": as_float_list(maxs),
        "center": as_float_list(center),
        "size": as_float_list(size),
    }


def compute_obb(points_xyz):
    mean = points_xyz.mean(axis=0)
    centered = points_xyz - mean

    cov = np.cov(centered, rowvar=False)
    cov = cov + np.eye(3, dtype=np.float64) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rot = eigvecs[:, order]

    if np.linalg.det(rot) < 0:
        rot[:, 2] *= -1.0

    local = centered @ rot
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
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
    s = np.diag([1.0, 1.0, -1.0])

    aabb_min = np.array(aabb["min"], dtype=np.float64)
    aabb_max = np.array(aabb["max"], dtype=np.float64)
    min_lhs = np.array([aabb_min[0], aabb_min[1], -aabb_max[2]], dtype=np.float64)
    max_lhs = np.array([aabb_max[0], aabb_max[1], -aabb_min[2]], dtype=np.float64)
    center_lhs = (min_lhs + max_lhs) / 2.0
    size_lhs = max_lhs - min_lhs

    center = np.array(obb["center"], dtype=np.float64)
    rot = np.array(obb["rotation_matrix"], dtype=np.float64)
    extents = np.array(obb["extents"], dtype=np.float64)

    center_lhs_obb = s @ center
    rot_lhs = s @ rot @ s

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


if __name__ == "__main__":
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

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.restore(model_params, mode="render")

    xyz = gaussians.get_xyz.detach().cpu().numpy()
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

        points = xyz[gaus_mask.numpy()]
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
