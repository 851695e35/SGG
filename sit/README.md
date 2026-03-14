# SiT Training (Public Release)

This folder is a compact training/sampling release for SiT weak-to-strong guidance.

## Core files

- `train.py`: main training entrypoint.
- `loss.py`: weak/segmented guidance loss definitions.
- `dataset.py`: dataset loading utilities.
- `samplers.py`: sampling methods used in training and generation.
- `generate.py`: checkpoint sampling script.
- `scripts/`: simplified runnable training examples.

## Supported guidance types

- `None`
- `Uncond`
- `Segmented`
- `LayerSkip`
- `Branch`
- `Separate`

## Quick start

```bash
cd sit
DATA_DIR=/path/to/imagenet256 \
OUTPUT_DIR=./exps \
MAX_TRAIN_STEPS=2000 \
./scripts/train_example.sh None
```

Run other guidance types by changing the first argument:

- `./scripts/train_example.sh Uncond`
- `./scripts/train_example.sh Segmented`
- `./scripts/train_example.sh LayerSkip`
- `./scripts/train_example.sh Branch`
- `./scripts/train_example.sh Separate`
