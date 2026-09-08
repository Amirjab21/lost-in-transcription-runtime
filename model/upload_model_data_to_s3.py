#!/usr/bin/env python3
"""Upload only the LoRA adapter and metadata required from this model folder.

Default destination: s3://ait-ai-storage-prod/indonesian_data/model/

Requires the configured AWS CLI and credentials with PutObject permission.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
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
DEFAULT_ENV_FILE = MODEL_DIR.parent / ".env"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--profile", help="Optional AWS shared-credentials profile.")
    parser.add_argument("--region", help="Optional AWS region.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="Credentials file with AWS_* or S3_AWS_* values.")
    args = parser.parse_args()

    prefix = args.prefix.strip("/") + "/"
    sources = [MODEL_DIR / filename for filename in ALLOWED_FILENAMES]
    missing = [path.name for path in sources if not path.is_file()]
    if missing:
        parser.error(f"Missing required model files: {', '.join(missing)}")

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
    for source in sources:
        # ``sync`` calls ListObjectsV2 before upload. Explicit cp commands do
        # not, so a tightly scoped PutObject-only IAM policy is sufficient.
        command = command_prefix + ["s3", "cp", str(source), f"s3://{args.bucket}/{prefix}{source.name}"]
        print(f"Uploading {source.name}")
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
