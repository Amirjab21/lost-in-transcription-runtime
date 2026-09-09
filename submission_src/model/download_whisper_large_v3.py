#!/usr/bin/env python3
"""Download and verify the official OpenAI Whisper large-v3 checkpoint.

Run from any directory:

    python model/download_whisper_large_v3.py

The checkpoint is saved as ``model/large-v3.pt``. Interrupted downloads can be
resumed safely from ``model/large-v3.pt.part``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path


CHECKPOINT_SHA256 = "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"
CHECKPOINT_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    f"{CHECKPOINT_SHA256}/large-v3.pt"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "large-v3.pt"
CHUNK_SIZE = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and sha256(output) == CHECKPOINT_SHA256:
        print(f"Verified checkpoint already exists: {output}")
        return

    partial = output.with_suffix(output.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(CHECKPOINT_URL)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
        print(f"Resuming download from byte {offset:,}")
    else:
        print(f"Downloading Whisper large-v3 to {partial}")

    try:
        with urllib.request.urlopen(request) as response:
            status = getattr(response, "status", None)
            # A server that ignores Range must not append a full response to a
            # partial checkpoint, because that would silently corrupt it.
            if offset and status != 206:
                print("Server did not accept resume request; restarting download.")
                partial.unlink(missing_ok=True)
                return download(output)
            expected = response.headers.get("Content-Length")
            total = offset + int(expected) if expected else None
            with partial.open("ab" if offset else "wb") as handle:
                received = offset
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
                    received += len(chunk)
                    if total:
                        print(f"\rDownloaded {received / 2**30:.2f}/{total / 2**30:.2f} GiB", end="", flush=True)
    except urllib.error.URLError as error:
        raise RuntimeError(f"Checkpoint download failed: {error}") from error
    print()

    actual = sha256(partial)
    if actual != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {partial}: expected {CHECKPOINT_SHA256}, got {actual}. "
            "Delete the .part file and retry."
        )
    partial.replace(output)
    print(f"Downloaded and verified: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Checkpoint destination path.")
    args = parser.parse_args()
    try:
        download(args.output.resolve())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
