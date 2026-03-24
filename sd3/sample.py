#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from sd3_pipeline import StableDiffusion3Pipeline


DEFAULT_SKIP_GUIDANCE_LAYERS = [7, 8, 9]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_int_list(csv_values: str) -> list[int]:
    if not csv_values.strip():
        return []
    return [int(part.strip()) for part in csv_values.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample images with SD3/SD3.5 segmented guidance."
    )
    parser.add_argument(
        "--model",
        choices=["sd3", "sd35"],
        required=True,
        help="Model family selector used for naming only.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local pretrained SD3/SD3.5 pipeline.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for generated images.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (for example: cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Inference dtype.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=28, help="Number of denoising steps."
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=4.5, help="CFG guidance scale."
    )
    parser.add_argument(
        "--cfg-guidance-start",
        type=int,
        default=1,
        help="CFG start step (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--cfg-guidance-end",
        type=int,
        default=28,
        help="CFG end step (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--use-segmented-guidance",
        action="store_true",
        help="Enable segmented guidance logic in the pipeline.",
    )
    parser.add_argument(
        "--skip-guidance-layers",
        type=str,
        default="7,8,9",
        help="Comma-separated layer ids for skip guidance.",
    )
    parser.add_argument(
        "--skip-layer-guidance-scale",
        type=float,
        default=3.0,
        help="SLG guidance scale.",
    )
    parser.add_argument(
        "--skip-layer-guidance-start",
        type=int,
        default=1,
        help="SLG start step (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--skip-layer-guidance-end",
        type=int,
        default=28,
        help="SLG end step (1-indexed, inclusive).",
    )
    parser.add_argument(
        "--cfg-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="CFG schedule policy in segmented pipeline.",
    )
    parser.add_argument(
        "--cfg-d-w",
        type=float,
        default=0.0,
        help="CFG linear schedule delta.",
    )
    parser.add_argument(
        "--slg-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="SLG schedule policy in segmented pipeline.",
    )
    parser.add_argument(
        "--slg-d-w",
        type=float,
        default=0.0,
        help="SLG linear schedule delta.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional output file name. Default is auto-generated.",
    )
    return parser.parse_args()


def to_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=to_dtype(args.dtype),
    )
    pipe = pipe.to(args.device)

    skip_guidance_layers = parse_int_list(args.skip_guidance_layers)
    if not skip_guidance_layers:
        skip_guidance_layers = DEFAULT_SKIP_GUIDANCE_LAYERS

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        cfg_guidance_start=args.cfg_guidance_start,
        cfg_guidance_end=args.cfg_guidance_end,
        use_segmented_guidance=args.use_segmented_guidance,
        skip_guidance_layers=skip_guidance_layers,
        skip_layer_guidance_scale=args.skip_layer_guidance_scale,
        skip_layer_guidance_start=args.skip_layer_guidance_start,
        skip_layer_guidance_end=args.skip_layer_guidance_end,
        cfg_schedule=args.cfg_schedule,
        cfg_d_w=args.cfg_d_w,
        slg_schedule=args.slg_schedule,
        slg_d_w=args.slg_d_w,
    ).images[0]

    if args.filename:
        filename = args.filename
    else:
        mode = "sgg" if args.use_segmented_guidance else "cfg"
        filename = (
            f"{args.model}_{mode}_seed{args.seed}_steps{args.num_inference_steps}"
            f"_cfg{args.guidance_scale}.png"
        )
    out_path = output_dir / filename
    image.save(out_path)
    print(f"Saved image to: {out_path}")


if __name__ == "__main__":
    main()
