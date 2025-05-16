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
"""File formats for checkpointing


# Version 1:
format = zip file
contains

```
net_state.pth
batch_info.json
model.json: str # json-serialized ModelConfigV1 (if version == 1)
metadata.json
    version: int
```

# other files are not interpreted by cbottle but can be used for other purposes
# (like in custom TrainingLoops)
"""

import torch
import zipfile
import json
from typing import Literal
import numpy as np
import cbottle.config.models
import cbottle.models
import cbottle.datasets.base
import warnings

current_version = 1


class Checkpoint:
    """A checkpoint object

    This is similar to ZipFile, but with convenience methods for reading and
    writing models.
    """

    def __init__(self, f, mode: Literal["w", "r"] = "r"):
        # Set zip to None in case the zipfile fails to open
        # this will be used in __del__
        self._zip = None
        self._zip = zipfile.ZipFile(f, mode)

    def write_model(self, net: torch.nn.Module):
        with self._zip.open("net_state.pth", "w", force_zip64=True) as f:
            torch.save(net.state_dict(), f)

    def read_model(self, net=None, map_location=None, transfer_learning=False, load_weights=None) -> torch.nn.Module:
        """Read the model from the checkpoint

        Args:
            net: If provided, the state dict will be loaded into this net.
                Otherwise, a new model will be created.
            map_location: Map location for torch.load
            transfer_learning: If True, allow partial loading (non-strict) and log unmatched keys
        """
        try:
            metadata = json.loads(self._zip.read("metadata.json").decode())
        except KeyError:
            warnings.warn("Old checkpoint format detected. Falling back to old format.")
            with self._zip.open("net_state.pth", "r") as f:
                return torch.load(f, weights_only=False)["net"]

        # new checkpoint format
        if metadata["version"] != current_version:
            raise ValueError(f"Unsupported checkpoint version: {metadata['version']}")

        if net is None:
            net = cbottle.models.get_model(self.read_model_config())

        with self._zip.open("net_state.pth", "r") as f:
            state_dict = torch.load(f, weights_only=True, map_location=map_location)

            if transfer_learning:
                missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
                loaded_keys = [k for k in state_dict.keys() if k not in unexpected_keys]
                if missing_keys:
                    print("Missing keys:")
                    for k in missing_keys:
                        print(f"  - {k}")
                if unexpected_keys:
                    print("Unexpected keys:")
                    for k in unexpected_keys:
                        print(f"  - {k}")
                print(f"Loaded {len(loaded_keys)} layers from checkpoint.")
                return net, bool(missing_keys or unexpected_keys)
            else:
                net.load_state_dict(state_dict, strict=True)

        return net


    def read_model_config(self) -> cbottle.config.models.ModelConfigV1:
        return cbottle.config.models.ModelConfigV1.loads(
            self._zip.open("model.json").read()
        )

    def read_model_state_dict(self) -> dict:
        with self._zip.open("net_state.pth", "r") as f:
            return torch.load(f, weights_only=True)

    def write_batch_info(self, batch_info: cbottle.datasets.base.BatchInfo):
        d = {
            "channels": batch_info.channels,
            "time_step": batch_info.time_step,
            "time_unit": batch_info.time_unit.name,  # enums don't serialize nicely
        }
        if batch_info.scales is not None:
            d["scales"] = list(batch_info.scales)
        if batch_info.center is not None:
            d["center"] = list(batch_info.center)
        self._zip.writestr("batch_info.json", json.dumps(d))

    def read_batch_info(self) -> cbottle.datasets.base.BatchInfo:
        with self._zip.open("batch_info.json", "r") as f:
            d = json.loads(f.read())
            d["scales"] = np.asarray(d["scales"]) if d["scales"] is not None else None
            d["center"] = np.asarray(d["center"]) if d["center"] is not None else None
            if d["time_unit"] == "":  # backwards compatibility
                d["time_unit"] = "h"
            d["time_unit"] = cbottle.datasets.base.TimeUnit[d["time_unit"]]
            return cbottle.datasets.base.BatchInfo(**d)

    def write_model_config(self, model_config: cbottle.config.models.ModelConfigV1):
        self._zip.writestr("model.json", model_config.dumps())

    def open(self, name, mode: Literal["w", "r"] = "r"):
        return self._zip.open(name, mode, force_zip64=True)

    def close(self):
        if self._zip.mode == "w":
            self._zip.writestr(
                "metadata.json", json.dumps({"version": current_version})
            )
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        if self._zip is not None and self._zip.fp:
            self.close()
