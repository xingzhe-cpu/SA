#!/usr/bin/env python3
import os
from pathlib import Path

from params_proto import PrefixProto


proj_root = Path(
    os.environ.get("RMX_CODE_DIR", Path(__file__).resolve().parents[3])
).resolve()
data_root = Path(os.environ.get("DATA_DIR", proj_root / "data-dir")).resolve()


class DCArgs(PrefixProto):
    seed = 0
    env_name = 'LunarLander-v1'  # Custom Env
    num_transitions = 10_000_000

    # Lunar Lander
    lunarlander_sac_model_dir = str(data_root / "experts" / "lunarlander")
    lunar_sac_model_dir = lunarlander_sac_model_dir  # Backward compatible alias.
    lunarlander_data_dir = str(data_root / "replay" / "lunarlander")
    lunar_data_dir = lunarlander_data_dir  # Backward compatible alias.

    pointmaze_data_dir = str(data_root / "replay" / "pointmaze")
    valid_return_threshold = -50
    randp = 0.

    # Block Push
    blockpush_sac_model_dir = str(data_root / "experts" / "blockpush")
    blockpush_data_dir = str(data_root / "replay" / "blockpush")
    blockpush_user_goal = 'target'  # 'target' or 'target2'
    pushenv_user_goal = blockpush_user_goal  # Backward compatible alias.
