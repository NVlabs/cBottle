#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pickle
from pathlib import Path

from cbottle import checkpointing


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a network-snapshot-*.pkl EMA artifact into a "
            "training-state-style .checkpoint file for inference."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="Path to network-snapshot-*.pkl",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=str,
        required=True,
        help=(
            "Path to a training-state-*.checkpoint from the same run. "
            "Used to copy batch_info and model_config."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write converted EMA checkpoint.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    snapshot_path = Path(args.snapshot).expanduser().resolve()
    reference_ckpt_path = Path(args.reference_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not snapshot_path.is_file():
        raise FileNotFoundError(str(snapshot_path))
    if not reference_ckpt_path.is_file():
        raise FileNotFoundError(str(reference_ckpt_path))

    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)
    if "ema" not in snapshot:
        raise KeyError(f"{snapshot_path} does not contain key 'ema'")
    ema_model = snapshot["ema"]

    with checkpointing.Checkpoint(str(reference_ckpt_path), "r") as ref_ckpt:
        batch_info = ref_ckpt.read_batch_info()
        model_config = ref_ckpt.read_model_config()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpointing.Checkpoint(str(output_path), "w") as out_ckpt:
        out_ckpt.write_model(ema_model)
        out_ckpt.write_batch_info(batch_info)
        out_ckpt.write_model_config(model_config)

    print(f"Wrote EMA checkpoint: {output_path}")


if __name__ == "__main__":
    main()
