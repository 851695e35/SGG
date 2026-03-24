#!/usr/bin/env python3
"""Load SD3 once, then interactively sample CFG and SGG images per iteration."""
from __future__ import annotations

import argparse
import random
import sys
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
        description="Interactive loop: load model once, then CFG + SGG per iteration."
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
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt (same for every iteration).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for generated images.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
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
        "--skip-guidance-layers",
        type=str,
        default="7,8,9",
        help="Comma-separated layer ids for skip guidance (SGG only).",
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
    return parser.parse_args()


def to_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def read_line(prompt: str) -> str | None:
    """Return stripped line, or None on EOF / empty / q|quit|exit."""
    try:
        line = input(prompt)
    except EOFError:
        return None
    stripped = line.strip()
    if stripped == "" or stripped.lower() in ("q", "quit", "exit"):
        return None
    return stripped


def read_float(prompt: str) -> float | None:
    raw = read_line(prompt)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        print("Invalid number.", file=sys.stderr)
        return None


def read_int(prompt: str) -> int | None:
    raw = read_line(prompt)
    if raw is None:
        return None
    try:
        return int(raw, 10)
    except ValueError:
        print("Invalid integer.", file=sys.stderr)
        return None


def main() -> None:
    args = parse_args()
    n = args.num_inference_steps

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_guidance_layers = parse_int_list(args.skip_guidance_layers)
    if not skip_guidance_layers:
        skip_guidance_layers = DEFAULT_SKIP_GUIDANCE_LAYERS

    print("Loading pipeline (one-time)...", flush=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=to_dtype(args.dtype),
    )
    pipe = pipe.to(args.device)
    print(
        "Ready. Each iteration: prompt, CFG scale, SLG scale, segmented step "
        f"(1..{n - 1}: CFG on steps 1..s, SLG on s+1..{n}).\n"
        "Empty line, q, or quit on any prompt exits.\n",
        flush=True,
    )

    iteration = 0
    while True:
        prompt = read_line("Prompt (empty or q to quit): ")
        if prompt is None:
            print("Exiting.", flush=True)
            break

        cfg_w = read_float("CFG guidance scale (w): ")
        if cfg_w is None:
            print("Exiting.", flush=True)
            break

        slg_w = read_float("SLG guidance scale (w): ")
        if slg_w is None:
            print("Exiting.", flush=True)
            break

        seg_step = read_int(
            f"Segmented step s — CFG on 1..s, SLG on s+1..{n} (s in 1..{n - 1}): "
        )
        if seg_step is None:
            print("Exiting.", flush=True)
            break
        if not (1 <= seg_step <= n - 1):
            print(
                f"Segmented step must be between 1 and {n - 1} (inclusive). Skipping.",
                file=sys.stderr,
            )
            continue

        cfg_start, cfg_end = 1, seg_step
        slg_start, slg_end = seg_step + 1, n
        loop_seed = args.seed + iteration
        set_seed(loop_seed)

        path_cfg = output_dir / "cfg.png"
        path_sgg = output_dir / "sgg.png"

        print(f"  [{iteration}] CFG (steps 1..{n}, full CFG)...", flush=True)
        img_cfg = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=n,
            guidance_scale=cfg_w,
            cfg_guidance_start=1,
            cfg_guidance_end=n,
            use_segmented_guidance=False,
            skip_guidance_layers=skip_guidance_layers,
            skip_layer_guidance_scale=slg_w,
            skip_layer_guidance_start=1,
            skip_layer_guidance_end=n,
            cfg_schedule=args.cfg_schedule,
            cfg_d_w=args.cfg_d_w,
            slg_schedule=args.slg_schedule,
            slg_d_w=args.slg_d_w,
        ).images[0]
        img_cfg.save(path_cfg)
        print(f"  Saved CFG: {path_cfg}", flush=True)

        print(
            f"  [{iteration}] SGG (CFG {cfg_start}..{cfg_end}, SLG {slg_start}..{slg_end})...",
            flush=True,
        )
        
        set_seed(loop_seed)
        img_sgg = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=n,
            guidance_scale=cfg_w,
            cfg_guidance_start=cfg_start,
            cfg_guidance_end=cfg_end,
            use_segmented_guidance=True,
            skip_guidance_layers=skip_guidance_layers,
            skip_layer_guidance_scale=slg_w,
            skip_layer_guidance_start=slg_start,
            skip_layer_guidance_end=slg_end,
            cfg_schedule=args.cfg_schedule,
            cfg_d_w=args.cfg_d_w,
            slg_schedule=args.slg_schedule,
            slg_d_w=args.slg_d_w,
        ).images[0]
        img_sgg.save(path_sgg)
        print(f"  Saved SGG: {path_sgg}\n", flush=True)

        iteration += 1


if __name__ == "__main__":
    main()
