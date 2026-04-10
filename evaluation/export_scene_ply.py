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


if __name__ == "__main__":
    parser = ArgumentParser(description="Export full-scene Gaussian model to PLY")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument(
        "--start_checkpoint",
        required=True,
        type=str,
        help="Checkpoint path (e.g. output/.../sp_20000.pth)",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        help="Output PLY file path (default: ${model_path}/scene.ply)",
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

    if args.save_path is None and not dataset.model_path:
        raise ValueError("model_path is required when --save_path is not provided")

    save_path = Path(args.save_path or os.path.join(dataset.model_path, "scene.ply"))
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gaussians.save_ply(str(save_path), for_unity=args.unity_compatible)
    print(f"[OK] Exported scene PLY to: {save_path}")
