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
# python3 scripts/xarray_server.py --sigma-max 200 --bf16  --state-path /scratch/icon_hpx_hack/ngc-versions/v1/cBottle-3d.zip  --start-time 2000-01-01 --end-time 2001-12-31  --batch-size 4 --freq 3h

docker run  --rm --name nginx --net host \
-ti \
-v $PWD/nginx.conf:/etc/nginx/nginx.conf \
-v nginx_cache:/tmp/cache nginx

