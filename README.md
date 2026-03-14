# Segmented Guidance


### Note
We thank a reviewer for pointing out a naming overlap with another interesting work in diffusion guidance research, [SEG](https://arxiv.org/abs/2408.00760). Since changing the conference title was not feasible, we use **SGG** to avoid confusion. Thank you for your understanding.

Public release for segmented guidance:

- `sd3`: SD3.5 sampling (and SD3-compatible pipeline/CLI).
- `sit`: SiT training with multiple guidance variants.

Install dependencies from repo root:

```bash
pip install -r requirements.txt
```

## SD3.5 Sampling

`sd3` contains:

- `sd3_pipeline.py`: segmented-guidance pipeline implementation.
- `sample.py`: unified CLI sampler for both SD3 and SD3.5.

Use the unified sampler in `sd3/sample.py`.

```bash
python sd3/sample.py \
  --model sd35 \
  --model-path /path/to/sd35 \
  --prompt "a cinematic photo of a lighthouse in stormy sea" \
  --use-segmented-guidance \
  --guidance-scale 4.5 \
  --cfg-guidance-start 1 \
  --cfg-guidance-end 14 \
  --skip-layer-guidance-scale 3.0 \
  --skip-layer-guidance-start 15 \
  --skip-layer-guidance-end 28 \
  --output-dir outputs/sd35
```

CFG-only baseline example:

```bash
python sd3/sample.py \
  --model sd3 \
  --model-path /path/to/sd3 \
  --prompt "a watercolor painting of a red fox in snow" \
  --guidance-scale 3.5 \
  --output-dir outputs/sd3
```

For SD3, keep the same command and switch `--model sd3` with an SD3 model path.

### Qualitative results


![SD3 qualitative comparison](asset/qualitative.png)

## SiT Training

### Training

Use the compact scripts under `sit/scripts`:

```bash
cd sit
DATA_DIR=/path/to/imagenet256 \
OUTPUT_DIR=./exps \
MAX_TRAIN_STEPS=400000 \
./scripts/train_segmented.sh
```

Supported guidance types:

- `None` (Baseline)
- `Uncond` (Model Guidance)
- `Segmented` (SGG)
- `LayerSkip` (SLG)
- `Branch ` (BR)
- `Separate` (AG)

Examples:

```bash
./scripts/train_none.sh
./scripts/train_uncond.sh
./scripts/train_segmented.sh
./scripts/train_layerskip.sh
./scripts/train_branch.sh
./scripts/train_separate.sh
```







### Evaluation

Following REPA and ADM, you can first generate samples with `sample.sh`, then refer to [ADM evaluation](https://github.com/openai/guided-diffusion/tree/main/evaluations) for FID/Inception Score.

```bash
cd sit
OUTPUT_DIR=./exps \
EXP_NAME=XXXXX \
CKPT_STEP=latest \
./scripts/sample.sh
```





### Reference links

For running details one can refer to wandb project for references.

https://wandb.ai/liangyuy/w2sseg/reports/w2s-seg---VmlldzoxNjA2Nzk0Mg?accessToken=hu0pq3um4hgqge00uhxmx65hhsxi1ik2nda9obmp7ut941hrphindexsflvdi8li





## Contact

If you have any problem or suggestions, feel free to contact liangyuy001@gmail.com.
