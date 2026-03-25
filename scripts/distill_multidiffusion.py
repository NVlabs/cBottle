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
from cbottle.training import distill_super_resolution


def parse_args():
    parser = argparse.ArgumentParser(description="global CorrDiff")
    parser.add_argument(
        "--output-path", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "--log-freq", type=int, default=100, help="Log every N steps (default: 100)"
    )
    parser.add_argument(
        "--lr-level", type=int, default=6, help="HPX level of the low-resolution map"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=15, help="training batch size per GPU"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=30, help="validation batch size per GPU"
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=3,
        help="number of workers for training dataloader",
    )
    parser.add_argument(
        "--bf16", action="store_true", help="use bfloat16 precision for training"
    )
    parser.add_argument(
        "--torch-compile", action="store_true", help="enable torch.compile for training"
    )
    parser.add_argument(
        "--window-function",
        type=str,
        default=None,
        help="Which window smoothing function to use in superpatch training",
    )
    parser.add_argument(
        "--window-alpha", type=float, default=1, help="window function alpha"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    distill_super_resolution.train(
        output_path=args.output_path,
        num_steps=1e6,
        log_freq=args.log_freq,
        lr_level=args.lr_level,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        torch_compile=args.torch_compile,
        window_function=args.window_function,
        window_alpha=args.window_alpha,
    )
