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
import subprocess
import os
import sys

import argparse

license_header = """# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# limitations under the License."""

parser = argparse.ArgumentParser()
parser.add_argument("--fix", action="store_true")
args = parser.parse_args()

expected = "SPDX-License-Identifier: Apache-2.0"
exts = (
    ".sh",
    ".py",
    ".yaml",
    ".yml",
    ".sh",
)

files = subprocess.check_output(["git", "ls-files", "."])

failed = []
for line in files.split():
    file = line.decode().strip()
    _, ext = os.path.splitext(file)
    if ext not in exts:
        continue

    with open(file) as f:
        buf = f.read(len(license_header))
        if buf != license_header:
            failed.append(file)

if failed:
    print("Missing license headers found for files:")
    print("----------------------------------------")
    for file in failed:
        if not args.fix:
            print(file)
            continue

        print(f"Fixing {file}")
        with open(file, "r") as f:
            text = ""
            for line in f:
                # strip incorrect SPDX lines
                text += line

        with open(file, "w") as f:
            f.write(license_header)
            f.write("\n")
            f.write(text)

    print()
    print("Run python ci/check_licenses.py --fix to fix the license headers")
    sys.exit(1)
