#!/usr/bin/env python3
"""Download the LoRA adapter/metadata from S3, then fetch Whisper large-v3.

Default source: s3://ait-ai-storage-prod/indonesian_data/model/
Files are placed in this script's ``model/`` directory unless --destination is
specified. Requires the configured AWS CLI and GetObject permission.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from s3_env import aws_environment


DEFAULT_BUCKET = "ait-ai-storage-prod"
DEFAULT_PREFIX = "indonesian_data/model/"
ALLOWED_FILENAMES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "whisper_token_lid_metadata.json",
)
MODEL_DIR = Path(__file__).resolve().parent
WHISPER_DOWNLOADER = MODEL_DIR / "download_whisper_large_v3.py"
DEFAULT_ENV_FILE = MODEL_DIR.parent / ".env"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--destination", type=Path, default=MODEL_DIR)
    parser.add_argument("--profile", help="Optional AWS shared-credentials profile.")
    parser.add_argument("--region", help="Optional AWS region.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="Credentials file with AWS_* or S3_AWS_* values.")
    args = parser.parse_args()

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip("/") + "/"
    if shutil.which("aws") is None:
        parser.error("AWS CLI was not found on PATH. Install/configure aws before running this script.")
    try:
        environment = aws_environment(args.env_file.resolve(), args.region)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    command_prefix = ["aws"]
    if args.profile:
        command_prefix += ["--profile", args.profile]
    if args.region:
        command_prefix += ["--region", args.region]
    for filename in ALLOWED_FILENAMES:
        # ``sync`` needs ListBucket; direct cp only needs GetObject for this
        # known allowlisted key.
        command = command_prefix + ["s3", "cp", f"s3://{args.bucket}/{prefix}{filename}", str(destination / filename)]
        print(f"Downloading {filename}")
        subprocess.run(command, check=True, env=environment)

    if not WHISPER_DOWNLOADER.is_file():
        parser.error(f"Missing Whisper downloader script: {WHISPER_DOWNLOADER}")
    subprocess.run(
        [sys.executable, str(WHISPER_DOWNLOADER), "--output", str(destination / "large-v3.pt")],
        check=True,
    )


if __name__ == "__main__":
    main()
