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
import configparser
from typing import Any
import os
import shutil
import tempfile
import sys

import fsspec

DEFAULT_PATH = os.path.expanduser("~/.config/rclone/rclone.conf")


class StorageConfigError(Exception):
    pass


def get_filesystem(remote_name, config_path=DEFAULT_PATH) -> fsspec.AbstractFileSystem:
    return fsspec.filesystem(
        "s3", **get_storage_options(remote_name, config_path=config_path)
    )


def get_storage_options(remote_name, config_path=DEFAULT_PATH) -> dict[str, Any]:
    if not remote_name:
        return None
    # Parse the rclone config file
    config = configparser.ConfigParser()
    config.read(config_path)

    # Ensure the remote exists in the config
    if remote_name not in config:
        raise StorageConfigError(f"Remote '{remote_name}' not found in rclone config.")

    # Extract credentials from the config
    remote_config = config[remote_name]

    if remote_config.get("type") != "s3":
        raise StorageConfigError(f"Remote '{remote_name}' is not an S3 remote.")

    access_key = remote_config.get("access_key_id")
    secret_key = remote_config.get("secret_access_key")
    endpoint_url = remote_config.get("endpoint", None)  # Optional endpoint

    if not access_key or not secret_key:
        raise StorageConfigError(
            f"Access key or secret key missing for remote '{remote_name}'."
        )

    # Instantiate and return the S3FileSystem object
    return dict(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": endpoint_url} if endpoint_url else None,
    )


def ensure_downloaded(url, local):
    if os.path.exists(local):
        return

    fs = fsspec.filesystem("http")
    print(f"Downloading from {url} to {local}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as d:
        tmpfile = os.path.join(d, "file")
        fs.get(url, tmpfile)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        shutil.move(tmpfile, local)
