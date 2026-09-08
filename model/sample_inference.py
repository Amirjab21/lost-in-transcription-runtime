#!/usr/bin/env python3
"""Run a small, inspectable Whisper + LID evaluation sample locally.

Example:
    python model/sample_inference.py --limit 10 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_DIR.parent
# Import the OpenAI Whisper implementation vendored beside this script. This
# avoids both a Hugging Face implementation and an installed-package/network
# dependency at inference time.
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import whisper

from whisper_lid import decode_with_token_language, load_token_lid_lora_adapter


def normalise(text: str) -> list[str]:
    """Apply the competition's relevant text normalization before WER."""
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\(\?+\)", " ", text)
    text = re.sub(r"\(([^()]*)\)", r"\1", text)
    text = text.replace("~", "").replace("—", " ")
    return re.sub(r'[¿¡";:,.!?]+', " ", text.casefold()).split()


def word_error_rate(reference: str, prediction: str) -> float:
    """Compute token-level Levenshtein WER without extra dependencies."""
    reference_words, prediction_words = normalise(reference), normalise(prediction)
    previous = list(range(len(prediction_words) + 1))
    for row, reference_word in enumerate(reference_words, start=1):
        current = [row]
        for column, prediction_word in enumerate(prediction_words, start=1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (reference_word != prediction_word),
            ))
        previous = current
    return previous[-1] / len(reference_words) if reference_words else 0.0


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def first_window_language_labels(model: object, audio_path: Path, language: str) -> list[str]:
    """Return token-LID predictions for the first 30-second Whisper window."""
    audio = whisper.load_audio(str(audio_path))
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
    )
    result = decode_with_token_language(model, mel, language=language)
    return result.token_languages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Number of metadata rows to evaluate.")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "data")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "sample_inference_results.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--language", default="id", help="Whisper language code used for transcription.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    metadata_path = args.data_dir / "metadata.tsv"
    clips_dir = args.data_dir / "clips"
    if not metadata_path.is_file():
        parser.error(f"Missing development metadata: {metadata_path}")
    if not clips_dir.is_dir():
        parser.error(f"Missing clips directory: {clips_dir}")
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))[: args.limit]
    if not rows:
        parser.error(f"No rows found in {metadata_path}")

    base_checkpoint = MODEL_DIR / "large-v3.pt"
    adapter_path = MODEL_DIR / "adapter_model.safetensors"
    if not base_checkpoint.is_file() or not adapter_path.is_file():
        parser.error("model/ must contain large-v3.pt and adapter_model.safetensors")

    device = select_device(args.device)
    print(f"Loading local Whisper large-v3 + LoRA on {device}")
    adapter = load_token_lid_lora_adapter(
        MODEL_DIR, device=device, base_model_path=base_checkpoint
    )
    # merge_and_unload is essential: it makes Whisper's native transcribe()
    # method use the adapted projection weights.
    model = adapter.merge_and_unload().eval()

    results: list[dict[str, str | float]] = []
    for index, row in enumerate(rows, start=1):
        audio_path = clips_dir / row["audio_filename"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Metadata clip does not exist: {audio_path}")
        predicted = str(model.transcribe(
            str(audio_path), language=args.language, task="transcribe", temperature=0,
            beam_size=5, condition_on_previous_text=True,
            fp16=device == "cuda", verbose=False,
        )["text"]).strip()
        results.append({
            "audio_file_path": str(audio_path),
            "ground_truth_transcript": row["transcript"],
            "language_labels": row.get("language", ""),
            "predicted_transcript": predicted,
            "predicted_language_labels": json.dumps(
                first_window_language_labels(model, audio_path, args.language)
            ),
            "wer": word_error_rate(row["transcript"], predicted),
        })
        print(f"Processed {index}/{len(rows)} clips")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "audio_file_path", "ground_truth_transcript", "language_labels",
            "predicted_transcript", "predicted_language_labels", "wer",
        ))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
