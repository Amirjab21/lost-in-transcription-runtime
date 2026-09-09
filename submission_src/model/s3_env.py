"""Load S3 credentials from a local .env file for AWS CLI subprocesses."""

from __future__ import annotations

import os
from pathlib import Path


def read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE dotenv entries without executing file contents."""
    values: dict[str, str] = {}
    for line_number, source_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def aws_environment(env_file: Path, region_override: str | None = None) -> dict[str, str]:
    if not env_file.is_file():
        raise FileNotFoundError(f"Credentials file does not exist: {env_file}")
    values = read_dotenv(env_file)
    access_key = values.get("AWS_ACCESS_KEY_ID") or values.get("S3_AWS_ACCESS_KEY_ID")
    secret_key = values.get("AWS_SECRET_ACCESS_KEY") or values.get("S3_AWS_SECRET_ACCESS_KEY")
    missing = []
    if not access_key:
        missing.append("AWS_ACCESS_KEY_ID (or S3_AWS_ACCESS_KEY_ID)")
    if not secret_key:
        missing.append("AWS_SECRET_ACCESS_KEY (or S3_AWS_SECRET_ACCESS_KEY)")
    if missing:
        raise ValueError(f"Missing required credential value(s) in {env_file}: {', '.join(missing)}")
    environment = os.environ.copy()
    environment["AWS_ACCESS_KEY_ID"] = access_key
    environment["AWS_SECRET_ACCESS_KEY"] = secret_key
    session_token = values.get("AWS_SESSION_TOKEN") or values.get("S3_AWS_SESSION_TOKEN")
    if session_token:
        environment["AWS_SESSION_TOKEN"] = session_token
    region = region_override or values.get("AWS_REGION") or values.get("AWS_DEFAULT_REGION") or values.get("S3_AWS_REGION")
    if region:
        environment["AWS_DEFAULT_REGION"] = region
    return environment
