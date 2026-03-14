import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
try:
    import wandb
except ImportError:
    wandb = None
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    # Handle log_with parameter properly
    log_with = None if args.report_to == "none" else args.report_to
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        weak_type=args.weak_type,
        weak_layer=args.weak_layer,
        condition=args.condition,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    
    # Create separate weak model for "Separate" weak type
    weak_model = None
    weak_ema = None
    if args.weak_type == "Separate":
        weak_model_name ='SiT-S/2' if args.model == 'SiT-B/2' else 'UNet-Base'
        weak_model = SiT_models[weak_model_name](
            input_size=latent_size,
            num_classes=args.num_classes,
            use_cfg = (args.cfg_prob > 0),
            z_dims = z_dims,
            encoder_depth=args.encoder_depth,
            weak_type="None",  # The weak model itself doesn't use weak-to-strong
            weak_layer=args.weak_layer,
            condition=args.condition,
            **block_kwargs
        )
        weak_model = weak_model.to(device)
        weak_ema = deepcopy(weak_model).to(device)
        requires_grad(weak_ema, False)
    
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=args.vae_model,
        local_files_only=args.vae_local_files_only,
    ).to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        weak_type=args.weak_type,
        w_scale=args.w_scale,
        weak_loss_ratio=args.weak_loss_ratio,
        weak_alpha=args.weak_alpha,
        weak_model=weak_model,
        num_classes=args.num_classes,
        guidance_low=args.guidance_low,
        guidance_high=args.guidance_high,
    )
    
    # create separate loss function for weak model
    weak_loss_fn = None
    if args.weak_type == "Separate":
        weak_loss_fn = SILoss(
            prediction=args.prediction,
            path_type=args.path_type, 
            encoders=encoders,
            accelerator=accelerator,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            weighting=args.weighting,
            weak_type="None",  # Weak model doesn't use weak-to-strong
            w_scale=args.w_scale,
            weak_loss_ratio=args.weak_loss_ratio,
            weak_model=None,  # No weak model for the weak model itself
            num_classes=args.num_classes,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
        )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Debug mode adjustments
    if args.debug:
        if accelerator.is_main_process:
            logger.info("DEBUG MODE: Reducing training steps and checkpointing frequency")
        args.max_train_steps = min(args.max_train_steps, 100)  # Limit to 100 steps in debug mode
        args.checkpointing_steps = 50  # Save checkpoint every 50 steps in debug mode
        args.sampling_steps = 25  # Sample every 25 steps in debug mode
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Create separate optimizer for weak model
    weak_optimizer = None
    if args.weak_type == "Separate":
        weak_optimizer = torch.optim.AdamW(
            weak_model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )    
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir, num_classes=args.num_classes, max_images_per_class=args.max_images_per_class)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    if args.weak_type == "Separate":
        update_ema(weak_ema, weak_model, decay=0)  # Ensure weak EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    if args.weak_type == "Separate":
        weak_model.train()
        weak_ema.eval()
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        if args.weak_type == "Separate" and 'weak_model' in ckpt:
            weak_model.load_state_dict(ckpt['weak_model'])
            weak_ema.load_state_dict(ckpt['weak_ema'])
            weak_optimizer.load_state_dict(ckpt['weak_opt'])
        global_step = ckpt['steps']

    # Prepare models for accelerator
    if args.weak_type == "Separate":
        model, optimizer, weak_model, weak_optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, weak_model, weak_optimizer, train_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

    if accelerator.is_main_process and log_with is not None:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name=args.project_name, 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
    
    # Counter for weak model updates
    weak_update_counter = 0
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:        
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None

            labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)
            duration = global_step / args.max_train_steps
            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs, training_duration=duration)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean + proj_loss_mean * args.proj_coeff
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
                    
                    # Update weak model every weak_update steps
                    if args.weak_type == "Separate":
                        weak_update_counter += 1
                        if weak_update_counter >= args.weak_update:
                            weak_update_counter = 0
                            # Compute weak model loss and update
                            with accelerator.accumulate(weak_model):
                                weak_model_kwargs = dict(y=labels)
                                weak_loss, weak_proj_loss = weak_loss_fn(weak_model, x, weak_model_kwargs, zs=zs)
                                weak_loss_mean = weak_loss.mean()
                                weak_proj_loss_mean = weak_proj_loss.mean()
                                weak_total_loss = weak_loss_mean + weak_proj_loss_mean * args.proj_coeff
                                accelerator.backward(weak_total_loss)
                                if accelerator.sync_gradients:
                                    weak_params_to_clip = weak_model.parameters()
                                    weak_grad_norm = accelerator.clip_grad_norm_(weak_params_to_clip, args.max_grad_norm)
                                weak_optimizer.step()
                                weak_optimizer.zero_grad(set_to_none=True)
                                update_ema(weak_ema, weak_model)
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # Handle both single GPU and multi-GPU cases
                    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    checkpoint = {
                        "model": model_state_dict,
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    if args.weak_type == "Separate":
                        weak_model_state_dict = weak_model.module.state_dict() if hasattr(weak_model, 'module') else weak_model.state_dict()
                        checkpoint.update({
                            "weak_model": weak_model_state_dict,
                            "weak_ema": weak_ema.state_dict(),
                            "weak_opt": weak_optimizer.state_dict(),
                        })
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # In debug mode, save only the first checkpoint and exit
                    if args.debug and global_step == args.checkpointing_steps:
                        logger.info("DEBUG MODE: Saved first checkpoint, exiting training")
                        break
            
            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=1.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                if wandb is not None and args.report_to != "none":
                    accelerator.log(
                        {
                            "samples": wandb.Image(array2grid(out_samples)),
                            "gt_samples": wandb.Image(array2grid(gt_samples)),
                        }
                    )
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-images-per-class", type=int, default=-1, help="Maximum images per class (-1 for all)")

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization # epoch need to be set larger when training on smaller scale.
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    
    parser.add_argument("--project-name", type=str, default="REPA")


    # weak to strong
    parser.add_argument("--weak-type", type=str, default="None", choices=["None", "LayerSkip", "Branch", "Separate", "Uncond", "Segmented"])
    parser.add_argument("--weak-layer", type=int, default=-1)
    parser.add_argument("--w-scale", type=float, default=0.0)
    parser.add_argument("--weak-alpha", type=float, default=0.0001)
    parser.add_argument("--weak-loss-ratio", type=float, default=0.2)
    parser.add_argument("--weak-update", type=int, default=1, help="Update weak model every weak-update steps")

    # guidance high and guidance low for training
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)

    # condition
    parser.add_argument("--condition", type=str, default="cond", choices=["cond", "uncond"])
    
    # debug mode
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: save only first checkpoint and reduce training steps")
    parser.add_argument(
        "--vae-model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE checkpoint name/path used for training-time sample visualization.",
    )
    parser.add_argument(
        "--vae-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load VAE with local_files_only for offline environments.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
