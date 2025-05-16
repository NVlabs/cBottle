#!/bin/bash
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

# shifter --image=nginx:stable --volume $PWD/scripts:/opt/scripts \
# ls /opt/scripts
# exit 1

if [ -f "scripts/nginx.pid" ]; then
    echo "Nginx is already running"
    shifter --image=nginx:stable --volume "$PWD/scripts:/opt/scripts;$SCRATCH/nginx:/nginx" bash -c 'nginx -c /opt/scripts/nginx/xarray.conf -s reload'
else
    mkdir -p $SCRATCH/nginx
    shifter --image=nginx:stable --volume "$PWD/scripts:/opt/scripts;$SCRATCH/nginx:/nginx" bash -c '
        mkdir -p /nginx/cache/{client_temp,proxy_temp,fastcgi_temp,uwsgi_temp,scgi_temp}
        nginx -c /opt/scripts/nginx/xarray.conf -g "daemon off;" &
    '
fi
