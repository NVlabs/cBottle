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
"""Create an index of the ERA5 dataset"""

# %%
import datetime
import re
import pandas as pd
import cbottle.storage
import os
import joblib
import config

FILE_PATTERN = re.compile(
    r"""
(?P<path>[^\n]*?  # Capture the directory name (any character except newline, followed by a forward slash)
    (   
        (?P<file>e5\.oper\.an\.(sfc|pl)\.
        (?P<table>\d+)_
        (?P<paramID>\d+)_
        (?P<variable_name>[\w\d]+)\.
        ll025(sc|uv)\.
        (?P<start_date>\d{10})_
        (?P<end_date>\d{10})\.
        nc)|
        (?P<precipFile>e5\.accumulated_tp_1h\.(?P<precipMonth>\d{6})\.nc)
    )
)
""",
    re.VERBOSE,
)


mem = joblib.Memory("/tmp/joblib")


@mem.cache
def ls(f, profile):
    fs = cbottle.storage.get_filesystem(profile)
    return fs.ls(f)


def parse_time(s):
    # parse 10 digit time string
    year = s[:4]
    month = s[4:6]
    day = s[6:8]
    hour = s[8:10]
    return datetime.datetime(int(year), int(month), int(day), int(hour))


def get_index(path):
    print("Indexing", path)
    files = ls(path)
    files = [os.path.join(path, file) for file in files]
    return index_from_files(path)


def precip_start_time(d):
    year = int(d[:4])
    month = int(d[4:6])
    return pd.Timestamp(year, month, 1)


def precip_stop_time(d):
    date = precip_start_time(d)
    stop = date + pd.DateOffset(months=1)
    return stop - pd.Timedelta(hours=1)


def index_from_files(files):
    all_matches = FILE_PATTERN.findall("\n".join(files))

    records = []
    for m in all_matches:
        m = {k: m[i - 1] for k, i in FILE_PATTERN.groupindex.items()}
        record = {}
        record["file"] = m["path"]
        if m["precipFile"]:
            record["table"] = -1
            record["param"] = -1
            record["name"] = "tp"
            record["start_date"] = precip_start_time(m["precipMonth"])
            record["end_date"] = precip_stop_time(m["precipMonth"])
        else:
            record["table"] = m["table"]
            record["param"] = m["paramID"]
            record["name"] = m["variable_name"]
            record["start_date"] = parse_time(m["start_date"])
            record["end_date"] = parse_time(m["end_date"])

        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df


if __name__ == "__main__":
    import sys

    files = []
    for root in config.ERA5_ROOTS:
        files.extend(ls(root, config.ERA5_PROFILE))

    index = index_from_files(files)
    index.to_csv(sys.argv[1], index=False)
