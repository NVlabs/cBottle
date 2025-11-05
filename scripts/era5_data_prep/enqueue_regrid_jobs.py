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
import pandas as pd
import tqdm
import os
import tasks
from cbottle.storage import get_filesystem
import config

start_time = "1980"
stop_time = "2025"
variables = tasks.sl_variables + tasks.pl_variables

if __name__ == "__main__":
    df = pd.read_csv("index.csv")
    n = df.set_index(["start_date", "name"]).sort_index()
    selected = n.loc[pd.IndexSlice[start_time:stop_time, variables], :]

    fs = get_filesystem(config.OUTPUT_PROFILE)
    done_files = fs.ls(config.OUTPUT_BUCKET)
    done_files = set([os.path.basename(f) for f in done_files])
    todo = [f for f in selected.file if os.path.basename(f) not in done_files]
    print("Enqueing", len(todo), "files", "of", len(selected.file))
    for f in tqdm.tqdm(todo):
        tasks.regrid_file.delay(f)
