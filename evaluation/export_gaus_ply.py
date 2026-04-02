import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from arguments import ModelParams, get_combined_args
from scene import GaussianModel
from utils.metric_utils import get_obj_by_mask


def collect_mask_files(mask_path: Path):
    """Return mask files from one file path or from a directory of *_gaus_mask.pt files."""
    if mask_path.is_file():
        return [mask_path]
    if mask_path.is_dir():
        return sorted(mask_path.glob("*_gaus_mask.pt"))
    raise FileNotFoundError(f"Mask path does not exist: {mask_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export gaussian object subsets to PLY")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument(
        "--start_checkpoint",
        required=True,
        type=str,
        help="Checkpoint used by eval_3d (e.g. output/.../sp_20000.pth)",
    )
    parser.add_argument(
        "--gaus_mask_path",
        default=None,
        type=str,
        help="Path to one *_gaus_mask.pt file or a directory containing them",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        help="Directory to save exported PLY files",
    )
    parser.add_argument(
        "--unity_compatible",
        action="store_true",
        help="Export with activated opacity/scale/rotation and scale_2=1 for Unity loaders",
    )
    args = get_combined_args(parser)

    dataset = model.extract(args)
    checkpoint = torch.load(args.start_checkpoint, map_location="cpu")
    model_params, _iter_step = checkpoint

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.restore(model_params, mode="render")

    if args.gaus_mask_path is None and not dataset.model_path:
        raise ValueError("model_path is required when --gaus_mask_path is not provided")

    mask_path = Path(args.gaus_mask_path or os.path.join(dataset.model_path, "objects"))
    save_path = Path(args.save_path or os.path.join(dataset.model_path, "objects_ply"))
    save_path.mkdir(parents=True, exist_ok=True)

    mask_files = collect_mask_files(mask_path)
    if len(mask_files) == 0:
        raise FileNotFoundError(
            f"No '*_gaus_mask.pt' files found in directory: {mask_path}. "
            "Please run eval_3d with --save_gaus_mask first, or pass --gaus_mask_path."
        )

    model_device = gaussians.get_xyz.device
    gaus_num = gaussians.get_xyz.shape[0]
    exported = 0
    for mask_file in mask_files:
        gaus_mask = torch.load(mask_file, map_location="cpu")
        if not torch.is_tensor(gaus_mask):
            print(f"[Skip] {mask_file}: loaded object is not a tensor (type={type(gaus_mask)})")
            continue

        gaus_mask = gaus_mask.bool().reshape(-1)
        if gaus_mask.numel() != gaus_num:
            print(f"[Skip] {mask_file}: mask size {gaus_mask.numel()} != gaussian size {gaus_num}")
            continue

        object_gaussians = get_obj_by_mask(gaussians, gaus_mask.to(model_device))
        object_name = mask_file.name.replace("_gaus_mask.pt", "")
        ply_path = save_path / f"{object_name}.ply"
        object_gaussians.save_ply(str(ply_path), for_unity=args.unity_compatible)
        exported += 1
        print(f"[OK] Exported {ply_path}")

    print(f"Done. Exported {exported} object PLY file(s) to: {save_path}")
