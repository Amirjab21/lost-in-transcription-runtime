"""Offline Whisper + LoRA competition submission."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import torch


SOURCE_DIR = Path(__file__).resolve().parent
MODEL_DIR = SOURCE_DIR / "model"
DATA_DIR = Path("/code_execution/data")
SUBMISSION_FORMAT_CSV = DATA_DIR / "submission_format.csv"
SUBMISSION_PATH = Path("/code_execution/submission/submission.csv")
MAX_CUDA_WORKERS = 2
CHECKPOINT_INTERVAL = 80
MAX_TRANSCRIPTION_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1


def log_timing(event: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    print(f"{timestamp} | TIMING | {event}", flush=True)


# Prefer the vendored Whisper implementation over any installed package.
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@cache
def load_model(requested_device: str = "auto") -> tuple[Any, str]:
    metadata_path = MODEL_DIR / "whisper_token_lid_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base_model = str(metadata["base_model"])
    checkpoint = MODEL_DIR / f"{base_model}.pt"
    adapter = MODEL_DIR / "adapter_model.safetensors"
    required = (
        checkpoint,
        adapter,
        MODEL_DIR / "adapter_config.json",
        metadata_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Submission model bundle is incomplete; missing: " + ", ".join(missing)
        )

    device = select_device(requested_device)
    print(f"Loading offline Whisper {base_model} + LoRA on {device}", flush=True)
    load_token_lid_lora_adapter = importlib.import_module(
        "whisper_lid"
    ).load_token_lid_lora_adapter
    peft_model = load_token_lid_lora_adapter(
        MODEL_DIR,
        device=device,
        base_model_path=checkpoint,
    )
    model = peft_model.merge_and_unload().eval()
    return model, device


def transcribe_with_model(model: Any, audio_path: Path, device: str) -> str:
    result = model.transcribe(
        str(audio_path),
        language="id",
        task="transcribe",
        temperature=0,
        beam_size=None,
        condition_on_previous_text=True,
        fp16=device == "cuda",
        # None disables Whisper's per-segment text and tqdm progress output.
        verbose=None,
    )
    return str(result["text"]).strip()


def transcribe(audio_path: Path, requested_device: str = "auto") -> str:
    model, device = load_model(requested_device)
    with torch.inference_mode():
        return transcribe_with_model(model, audio_path, device)


class TranscriptionWorker:
    """One model replica and, on CUDA, its dedicated execution stream."""

    def __init__(
        self,
        model: Any,
        device: str,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.stream = stream

    @torch.inference_mode()
    def _transcribe_once(self, audio_path: Path) -> str:
        if self.stream is None:
            return transcribe_with_model(self.model, audio_path, self.device)
        with torch.cuda.stream(self.stream):
            transcript = transcribe_with_model(self.model, audio_path, self.device)
        self.stream.synchronize()
        return transcript

    def transcribe(self, audio_path: Path) -> str | None:
        for attempt in range(1, MAX_TRANSCRIPTION_ATTEMPTS + 1):
            try:
                return self._transcribe_once(audio_path)
            except Exception as error:  # noqa: BLE001 - isolate per-clip failures
                print(
                    f"Attempt {attempt}/{MAX_TRANSCRIPTION_ATTEMPTS} failed for "
                    f"{audio_path.name}: {type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt == MAX_TRANSCRIPTION_ATTEMPTS:
                    print(
                        f"Giving up on {audio_path.name}; leaving transcript blank",
                        flush=True,
                    )
                    return None
                if self.device == "cuda" and isinstance(error, torch.OutOfMemoryError):
                    torch.cuda.empty_cache()
                time.sleep(RETRY_DELAY_SECONDS)
        return None


def create_workers(model: Any, device: str, count: int) -> list[TranscriptionWorker]:
    if count == 1:
        stream = torch.cuda.Stream(device=device) if device == "cuda" else None
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(device))
        return [TranscriptionWorker(model, device, stream)]

    if device != "cuda":
        raise ValueError("Multiple transcription workers require CUDA")

    models = [model]
    for _ in range(1, count):
        models.append(copy.deepcopy(model).eval())
    # Deep copies are enqueued on CUDA's current stream. Complete them before
    # any worker starts reading weights on its own stream.
    torch.cuda.synchronize(device)
    return [
        TranscriptionWorker(replica, device, torch.cuda.Stream(device=device))
        for replica in models
    ]


def transcribe_rows(
    rows: list[dict[str, str]],
    clips_dir: Path,
    requested_device: str,
    checkpoint: Callable[[int], None] | None = None,
) -> None:
    if not rows:
        return

    model, device = load_model(requested_device)
    is_smoke_test = os.getenv("LOST_IN_TRANSCRIPTION_IS_SMOKE", "0") == "1"
    worker_count = (
        min(MAX_CUDA_WORKERS, len(rows))
        if device == "cuda" and not is_smoke_test
        else 1
    )
    print(f"Using {worker_count} transcription worker(s)", flush=True)
    workers = create_workers(model, device, worker_count)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for batch_start in range(0, len(rows), worker_count):
            batch = rows[batch_start : batch_start + worker_count]
            futures = {}
            for offset, (worker, row) in enumerate(zip(workers, batch)):
                audio_path = clips_dir / row["audio_filename"]
                if not audio_path.is_file():
                    raise FileNotFoundError(f"Audio clip does not exist: {audio_path}")
                future = executor.submit(worker.transcribe, audio_path)
                futures[future] = batch_start + offset

            for future in as_completed(futures):
                row_index = futures[future]
                prediction = future.result()
                if prediction is not None:
                    rows[row_index]["transcript"] = prediction
                completed_count += 1
                if completed_count == 0:
                    log_timing("First transcription started")
                if completed_count == len(rows):
                    log_timing("Last transcription finished")

            completed = batch_start + len(batch)
            if checkpoint is not None and (
                completed % CHECKPOINT_INTERVAL == 0 or completed == len(rows)
            ):
                checkpoint(completed)


def write_submission(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    output: Path,
) -> None:
    """Atomically replace the output so interruption cannot leave a partial CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    with temporary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_output.replace(output)


def main() -> None:
    log_timing("Script started")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--input", type=Path, default=SUBMISSION_FORMAT_CSV)
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(f"Missing submission input: {args.input}")

    delimiter = "\t" if args.input.suffix.casefold() == ".tsv" else ","
    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or "audio_filename" not in reader.fieldnames:
            raise ValueError("Submission input must contain an audio_filename column")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if args.limit is not None:
        rows = rows[: args.limit]

    if "transcript" not in fieldnames:
        fieldnames.append("transcript")
    for row in rows:
        row["transcript"] = ""

    # Create a valid, complete output before expensive model loading begins.
    write_submission(rows, fieldnames, args.output)
    print(f"Wrote blank submission to {args.output}", flush=True)

    def checkpoint(completed: int) -> None:
        write_submission(rows, fieldnames, args.output)
        print(
            f"Checkpointed {completed}/{len(rows)} predictions to {args.output}",
            flush=True,
        )

    transcribe_rows(
        rows,
        args.data_dir / "clips",
        args.device,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main()
