#!/usr/bin/env python

import os
import configparser
from pathlib import Path

# Environment variables
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
pdx_secret = os.getenv("PDX_SECRET_ACCESS_KEY", "")

# Target config path
config_path = Path.home() / ".config" / "rclone" / "rclone.conf"
config_path.parent.mkdir(parents=True, exist_ok=True)

# Read existing config
config = configparser.ConfigParser()
config.read(config_path)

# Add [pdx] section if missing
if "pdx" not in config:
    config["pdx"] = {
        "type": "s3",
        "provider": "Other",
        "access_key_id": "team-earth2-datasets",
        "secret_access_key": pdx_secret,
        "endpoint": "https://pdx.s8k.io",
    }

# Add [pbss] section if missing
if "pbss" not in config:
    config["pbss"] = {
        "type": "s3",
        "provider": "Other",
        "access_key_id": "team-earth2-datasets",
        "secret_access_key": aws_secret,
        "endpoint": "https://pbss.s8k.io",
    }

# Write updated config
with open(config_path, "w") as f:
    config.write(f)
