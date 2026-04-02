#!/usr/bin/env python3
import os
from pathlib import Path

from params_proto import ParamsProto


proj_root = Path(
    os.environ.get("RMX_CODE_DIR", Path(__file__).resolve().parents[2])
).resolve()
data_root = Path(os.environ.get("DATA_DIR", proj_root / "data-dir")).resolve()
out_root = Path(os.environ.get("OUT_DIR", proj_root / "output-dir")).resolve()

# The upstream README uses WANDB_MODE=disable, but modern wandb expects disabled.
if os.environ.get("WANDB_MODE", "").lower() == "disable":
    os.environ["WANDB_MODE"] = "disabled"

class Args(ParamsProto):
    env_name = 'LunarLander-v1'
    dataset_envs = None
    data_dir = str(data_root)
    out_dir = str(out_root)

    # DDPM configuration
    num_diffusion_steps = 50
    beta_min = 1e-4
    beta_max = 0.26
    beta_schedule = 'sigmoid'
    ddpm_model_path = os.environ.get("DDPM_MODEL_PATH", str(data_root / "ddpm"))

    num_training_steps = 100_000
    eval_every = 2000
    save_every = 2000

    # Data directories
    lunarlander_data_dir = str(data_root / "replay" / "lunarlander")
    lunar_data_dir = lunarlander_data_dir  # Backward compatible alias.
    pointmaze_data_dir = str(data_root / "replay" / "pointmaze")
    blockpush_data_dir = str(data_root / "replay" / "blockpush")

    # Stores evaluation results
    results_dir = os.environ.get("RESULTS_DIR", str(out_root))
    pt_dir = str(data_root)

    randp = 0.
    seed = 0

    # Used in evaluation
    fwd_diff_ratio = 0.4
    laggy_actor_repeat_prob = 0.8
    noisy_actor_eps = 0.8

    batch_size = 4096

    # Temporary directory that stores SAC model checkpoints
    sac_model_dir = os.environ.get("SAC_MODEL_DIR", str(out_root / "sac"))
