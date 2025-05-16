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
#!/bin/bash
# trouble shooting
# if you see this  2025-05-15T23:04:51.249-0700 [ERROR] agent: startup error: error="error reading server metadata: unexpected end of JSON input"
# then delete the consul_data directory, kill any lingering processes (ps aux | grep consul) and try again
set -ex

port="$1"

# Create necessary directories
mkdir -p $SCRATCH/consul_data
mkdir -p $SCRATCH/nginx $SCRATCH/consul-template

# download consul-template
mkdir -p $PWD/.bin
if [ ! -f $PWD/.bin/consul-template ]; then
    curl -O https://releases.hashicorp.com/consul-template/0.31.0/consul-template_0.31.0_linux_amd64.zip
    unzip consul-template_0.31.0_linux_amd64.zip
    mv consul-template $PWD/.bin/consul-template
    
    rm -f consul-template_0.31.0_linux_amd64.zip
fi
export PATH=$PWD/.bin:$PATH

# Pull required Shifter images
# shifterimg pull consul:1.15
# shifterimg pull nginx:stable
# shifterimg pull hashicorp/consul-template:latest

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    pkill -f "consul"
    pkill -f "nginx"
    pkill -f "consul-template"
    rm -f .load-balancer
}


trap cleanup EXIT 

# Start Consul in Shifter
MY_IP=$(hostname -I | awk '{print $1}')
shifter --image=consul:1.15 --volume $SCRATCH/consul_data:/consul/data \
    consul agent -server -bootstrap-expect=1 -ui -http-port="$CONSUL_PORT" -bind=$MY_IP -advertise=$MY_IP -client 0.0.0.0 -data-dir=/consul/data &

echo $MY_IP > .load-balancer

# Wait for Consul to start
sleep 10
# fill in template

consul-template -config scripts/consul-template/config.hcl

echo "Load balancer setup complete. Services are running."
echo "You can check the status of services with:"
echo "  - Consul: http://localhost:8500"
echo "  - Nginx: http://localhost"
