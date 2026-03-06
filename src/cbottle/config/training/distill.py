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

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PatchingConfig:
    subpatch_num: int = 2
    overlap_pixel: int = 32


@dataclass
class OptConfig:
    lr: float = 1e-7
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.99])
    eps: float = 1e-11


@dataclass
class SampleTConfig:
    """Config for sampling t from a time distribution."""

    # time distribution (currently supporting: uniform, lognormal, polynomial, logitnormal, shift, and log_t)
    time_dist_type: str = "lognormal"
    # mu in lognormal, logitnormal, and log_t distributions
    train_p_mean: float = 0.0
    # sigma in lognormal, logitnormal, and log_t distributions
    train_p_std: float = 1.6
    # lowest value in truncated range
    min_t: float = 0.002
    # highest value in truncated range
    max_t: float = 180.0
    # If provided, it is in the form [t_max, ..., 0] where len(t_list) needs to equal student_sample_steps + 1
    t_list: Optional[List[float]] = None
    min_r: float = 0.0
    quantize: bool = False
    sigma_data: float = 0.5


@dataclass
class LossConfig:
    use_cd: bool = False
    huber_const: Optional[float] = None
    use_squared_l2: Optional[bool] = None
    weighting_ct_loss: Optional[str] = None
    # SCM-specific extensions:
    tagent_warmup_steps: Optional[int] = None
    tangent_warmup_const: Optional[float] = None
    prior_weighting_enabled: Optional[bool] = None
    g_norm_spatial_invariance: Optional[bool] = None
    divide_x_0_spatial_dim: Optional[bool] = None
    use_jvp_finite_diff: Optional[bool] = None


@dataclass
class BlockKwargs:
    dropout: float = 0.0  # Default 0.0 but overridden in cm and scm


@dataclass
class CtScheduleConfig:
    q: float = 2.0
    ratio_limit: float = 0.999
    kimg_per_stage: int = 2000


@dataclass
class CallbacksConfig:
    ct_schedule: Optional[CtScheduleConfig] = None
    # ema: Optional[EmaConfig] = None  # Placeholder for possible extension


@dataclass
class CmModelConfig:
    use_ema: bool = False
    student_sample_steps: int = 1
    student_sample_type: str = "sde"
    precision: str = "float32"
    precision_amp: Optional[str] = None
    precision_amp_infer: Optional[str] = None
    precision_amp_enc: Optional[str] = None
    precision_fsdp: Optional[str] = None
    add_teacher_to_fsdp_dict: bool = True

    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(
            use_cd=False,
            huber_const=0.06,
            use_squared_l2=False,
            weighting_ct_loss="c_out_sq",
        )
    )

    sample_t_cfg: SampleTConfig = field(
        default_factory=lambda: SampleTConfig(
            train_p_mean=0.0,
            train_p_std=1.6,
            min_t=0.002,
            t_list=None,
            max_t=180,
            min_r=0.0,
            quantize=False,
            sigma_data=0.5,
            time_dist_type="lognormal",
        )
    )
    block_kwargs: BlockKwargs = field(default_factory=lambda: BlockKwargs(dropout=0.2))


@dataclass
class CmConfig:
    model: CmModelConfig = field(default_factory=CmModelConfig)
    callbacks: CallbacksConfig = field(
        default_factory=lambda: CallbacksConfig(
            ct_schedule=CtScheduleConfig(
                q=4.0, ratio_limit=0.9961, kimg_per_stage=128000
            )
        )
    )


@dataclass
class ScmModelConfig:
    use_ema: bool = False
    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(
            use_cd=False,
            tagent_warmup_steps=10000,
            tangent_warmup_const=0.1,
            prior_weighting_enabled=True,
            g_norm_spatial_invariance=True,
            divide_x_0_spatial_dim=True,
            use_jvp_finite_diff=False,
        )
    )
    sample_t_cfg: SampleTConfig = field(
        default_factory=lambda: SampleTConfig(
            train_p_mean=0.0,
            train_p_std=1.2,
            min_t=0.002,
            t_list=None,
            max_t=180,
            sigma_data=0.5,
        )
    )
    block_kwargs: BlockKwargs = field(default_factory=lambda: BlockKwargs(dropout=0.2))


@dataclass
class ScmConfig:
    model: ScmModelConfig = field(default_factory=ScmModelConfig)
    # callbacks: Optional[CallbacksConfig] = None  # Can be added if needed


@dataclass
class Dmd2ModelConfig:
    use_ema: bool = False
    sample_t_cfg: SampleTConfig = field(
        default_factory=lambda: SampleTConfig(
            train_p_mean=0.0,
            train_p_std=1.2,
            min_t=0.002,
            t_list=None,
            max_t=180,
            sigma_data=0.5,
        )
    )
    student_update_freq: int = 5
    guidance_scale: float = 1.0
    gan_loss_weight_gen: float = 0.001
    block_kwargs: BlockKwargs = field(default_factory=lambda: BlockKwargs(dropout=0.0))


@dataclass
class Dmd2Config:
    model: Dmd2ModelConfig = field(default_factory=Dmd2ModelConfig)


@dataclass
class ModulusDefaultSchedulerConfig:
    lr_decay: float = 1.0
    lr_rampup: int = 0


@dataclass
class LambdaInverseSquareRootSchedulerConfig:
    warm_up_steps: int = 0
    decay_steps: int = 10000


@dataclass
class LambdaLinearSchedulerConfig:
    warm_up_steps: List[int] = field(default_factory=lambda: [0])
    cycle_lengths: List[int] = field(default_factory=lambda: [10000000000])
    f_start: List[float] = field(default_factory=lambda: [1.0e-6])
    f_max: List[float] = field(default_factory=lambda: [1.0])
    f_min: List[float] = field(default_factory=lambda: [1.0])


@dataclass
class SchedulerConfig:
    modulus_default: ModulusDefaultSchedulerConfig = field(
        default_factory=ModulusDefaultSchedulerConfig
    )
    LambdaInverseSquareRootScheduler: LambdaInverseSquareRootSchedulerConfig = field(
        default_factory=LambdaInverseSquareRootSchedulerConfig
    )
    LambdaLinearScheduler: LambdaLinearSchedulerConfig = field(
        default_factory=LambdaLinearSchedulerConfig
    )


@dataclass
class DistillConfig:
    teacher_ckp_path: str = "/lustre/fsw/portfolios/coreai/projects/coreai_climate_earth2/asui/.cache/cbottle/cBottle-SR.zip"
    total_batch_size: int = 1024
    training_duration: int = 512000000
    grad_clip_threshold: float = 1000000
    patching: Optional[PatchingConfig] = None  # disable superpatch training
    scheduler_name: str = "LambdaInverseSquareRootScheduler"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    opt_name: str = "Adam"
    opt: OptConfig = field(
        default_factory=lambda: OptConfig(
            lr=1e-7, weight_decay=0.0, betas=[0.9, 0.99], eps=1e-11
        )
    )
    mode: str = "cm"
    cm: CmConfig = field(default_factory=CmConfig)
    scm: ScmConfig = field(default_factory=ScmConfig)
    dmd2: Dmd2Config = field(default_factory=Dmd2Config)
