#!/usr/bin/env python3
"""Offline Whisper large-v3 + token-LID LoRA inference entrypoint."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch

from model.whisper_lid import load_token_lid_lora_adapter


ROOT = Path(__file__).resolve().parent
# Defaults are the evaluator's fixed paths. The environment overrides are only
# for direct, no-Docker local/GPU smoke tests of this exact entrypoint.
DATA_DIR = Path(os.environ.get("CODE_EXECUTION_DATA_DIR", "/code_execution/data"))
OUTPUT_PATH = Path(
    os.environ.get("CODE_EXECUTION_SUBMISSION_PATH", "/code_execution/submission/submission.csv")
)
MODEL_DIR = ROOT / "model"
BASE_CHECKPOINT = MODEL_DIR / "large-v3.pt"
ADAPTER_DIR = MODEL_DIR


def read_manifest() -> list[dict[str, str]]:
    """Read the evaluator manifest, with the development TSV as a local fallback."""
    for filename, delimiter in (
        ("test_metadata.csv", ","),
        ("submission_format.csv", ","),
        ("metadata.tsv", "\t"),
    ):
        path = DATA_DIR / filename
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
        if not rows or "audio_filename" not in rows[0]:
            raise ValueError(f"{path} must contain an audio_filename column")
        return rows
    raise FileNotFoundError("No test_metadata.csv, submission_format.csv, or metadata.tsv in /code_execution/data")


def load_model() -> object:
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing vendored Whisper checkpoint: {BASE_CHECKPOINT}")
    if not (ADAPTER_DIR / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter: {ADAPTER_DIR / 'adapter_model.safetensors'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Passing a checkpoint *path* prevents openai-whisper from downloading at
    # runtime. Merge LoRA weights so Whisper's native transcribe method sees
    # the adapted projections instead of the unwrapped base model.
    adapter = load_token_lid_lora_adapter(
        ADAPTER_DIR,
        device=device,
        base_model_path=BASE_CHECKPOINT,
    )
    model = adapter.merge_and_unload()
    model.eval()
    return model


def main() -> None:
    rows = read_manifest()
    model = load_model()
    language = os.environ.get("WHISPER_LANGUAGE", "id")
    output_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        filename = row["audio_filename"]
        audio_path = DATA_DIR / "clips" / filename
        if not audio_path.is_file():
            raise FileNotFoundError(f"Manifest clip does not exist: {audio_path}")
        result = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            temperature=0,
            beam_size=5,
            condition_on_previous_text=True,
            fp16=torch.cuda.is_available(),
            verbose=False,
        )
        output_rows.append({"audio_filename": filename, "transcript": str(result["text"]).strip()})
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"Transcribed {index}/{len(rows)} clips")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio_filename", "transcript"))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} predictions")


if __name__ == "__main__":
    main()
