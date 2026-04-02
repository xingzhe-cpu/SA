# Local Usage Guide

This guide is for running the project locally without Docker.

## 1. Environment

The project keeps a local conda environment at:

```bash
/data0/user/yejiawei/code/SA/diffusion-for-shared-autonomy/.conda/env
```

Activate it with:

```bash
cd /data0/user/yejiawei/code/SA/diffusion-for-shared-autonomy
conda activate /data0/user/yejiawei/code/SA/diffusion-for-shared-autonomy/.conda/env
```

## 2. Required environment variables

Use these defaults for local runs:

```bash
export DATA_DIR=$PWD/data-dir
export OUT_DIR=$PWD/output-dir
export WANDB_MODE=disabled
export MPLCONFIGDIR=$PWD/.mplconfig
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Create the output directory once:

```bash
mkdir -p "$OUT_DIR"
```

## 3. Quickstart evaluation

Run pretrained evaluation:

```bash
python -m diffusha.diffusion.evaluation.eval_assistance --env-name LunarLander-v1 --out-dir "$OUT_DIR" --num-episodes 1 --num-envs 1
python -m diffusha.diffusion.evaluation.eval_assistance --env-name LunarLander-v5 --out-dir "$OUT_DIR" --num-episodes 1 --num-envs 1
python -m diffusha.diffusion.evaluation.eval_assistance --env-name BlockPushMultimodal-v1 --out-dir "$OUT_DIR" --num-episodes 1 --num-envs 1
```

## 4. Full pipeline

### 4.1 Train expert policies

```bash
python -m diffusha.data_collection.train_sac --env-name LunarLander-v1 --steps 3000000
python -m diffusha.data_collection.train_sac --env-name LunarLander-v5 --steps 3000000
python -m diffusha.data_collection.train_sac --env-name BlockPushMultimodal-v1 --steps 1000000
```

Expert checkpoints are written under:

```bash
$OUT_DIR/sac/<env-name-lower>/...
```

Move or copy the final checkpoint directories to the locations expected by data collection:

```bash
$DATA_DIR/experts/lunarlander/v1
$DATA_DIR/experts/lunarlander/v5
$DATA_DIR/experts/blockpush
```

### 4.2 Collect demonstrations

```bash
python -m diffusha.data_collection.generate_data -l 0 --sweep-file diffusha/data_collection/config/sweep/sweep_lander-v1.jsonl
python -m diffusha.data_collection.generate_data -l 0 --sweep-file diffusha/data_collection/config/sweep/sweep_lander-v5.jsonl
python -m diffusha.data_collection.generate_data -l 0 --sweep-file diffusha/data_collection/config/sweep/sweep-blockpush.jsonl
```

For BlockPush, also run:

```bash
python -m diffusha.data_collection.flip_replay "$DATA_DIR/replay/blockpush/target/randp_0.0" "$DATA_DIR/replay/blockpush/target-flipped/randp_0.0"
```

### 4.3 Train diffusion models

```bash
python -m diffusha.diffusion.train --sweep-file diffusha/config/sweep/sweep-lunarlander.jsonl -l 0
python -m diffusha.diffusion.train --sweep-file diffusha/config/sweep/sweep-lunarlander.jsonl -l 1
python -m diffusha.diffusion.train --sweep-file diffusha/config/sweep/sweep-blockpush.jsonl -l 0
```

Diffusion checkpoints are written under:

```bash
$DATA_DIR/ddpm/<env-name-lower>/
```

## 5. Helper script

For smoke tests and quick checks, you can use:

```bash
bash scripts/local_pipeline.sh env-info
bash scripts/local_pipeline.sh eval-pretrained
bash scripts/local_pipeline.sh full-smoke
ENV_NAME=BlockPushMultimodal-v1 bash scripts/local_pipeline.sh eval-pretrained
```

## 6. Notes

- The project still prints some harmless warnings from `gym`, `pygame`, `pybullet`, and optional environments not used by the current task.
- `WANDB_MODE=disable` from the original upstream README is also accepted, but `WANDB_MODE=disabled` is the safer value for current wandb versions.
- For local runs, `BlockPush` requires `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
