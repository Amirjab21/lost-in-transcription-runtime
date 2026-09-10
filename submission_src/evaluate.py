#!/usr/bin/env python3
"""Create a detailed local evaluation CSV from development metadata.tsv.

This is deliberately separate from main.py: competition submissions must emit
only audio_filename and transcript, whereas local model review benefits from
references, WER, and token-level language-ID output.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import torch
import whisper

from main import DATA_DIR, load_model
from model.whisper_lid import decode_with_token_language


OUTPUT_PATH = Path("/code_execution/submission/local_evaluation.csv")


def normalise(text: str) -> list[str]:
    """Use the competition's relevant punctuation/case normalization."""
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\(\?+\)", " ", text)
    text = re.sub(r"\(([^()]*)\)", r"\1", text)
    text = text.replace("~", "").replace("—", " ")
    text = re.sub(r'[¿¡";:,.!?]+', " ", text.casefold())
    return text.split()


def word_error_rate(reference: str, prediction: str) -> float:
    """Token Levenshtein WER, avoiding an extra runtime dependency."""
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


def predict_language_labels(model: object, audio_path: Path) -> list[str]:
    """Return the LID labels for the first Whisper-sized (30-second) window.

    The LID decoder is intentionally greedy and supports one 30-second mel at
    a time.  Transcription below still uses Whisper's long-audio beam-search
    path, so the labels are auxiliary diagnostics rather than an alignment to
    every beam-search token.
    """
    audio = whisper.load_audio(str(audio_path))
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=model.dims.n_mels)
    labels = decode_with_token_language(model, mel, language=os.environ.get("WHISPER_LANGUAGE", "id"))
    return labels.token_languages


def main() -> None:
    metadata_path = DATA_DIR / "metadata.tsv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Expected development metadata at {metadata_path}")
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    model = load_model()
    language = os.environ.get("WHISPER_LANGUAGE", "id")
    results: list[dict[str, str | float]] = []

    for index, row in enumerate(rows, start=1):
        filename = row["audio_filename"]
        audio_path = DATA_DIR / "clips" / filename
        prediction = str(model.transcribe(
            str(audio_path), language=language, task="transcribe", temperature=0,
            beam_size=5, condition_on_previous_text=True,
            fp16=torch.cuda.is_available(), verbose=False,
        )["text"]).strip()
        results.append({
            "audio_file_path": str(audio_path),
            "ground_truth_transcript": row["transcript"],
            "language_labels": row.get("language", ""),
            "predicted_transcript": prediction,
            "predicted_language_labels": json.dumps(predict_language_labels(model, audio_path)),
            "wer": word_error_rate(row["transcript"], prediction),
        })
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"Evaluated {index}/{len(rows)} clips")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "audio_file_path", "ground_truth_transcript", "language_labels",
            "predicted_transcript", "predicted_language_labels", "wer",
        ))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
