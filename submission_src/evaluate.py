"""Evaluate a small number of development clips with the submission model."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from main import DATA_DIR, transcribe


DEFAULT_OUTPUT = Path("/code_execution/submission/local_evaluation.csv")


def normalise(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\(\?+\)", " ", text)
    text = re.sub(r"\(([^()]*)\)", r"\1", text)
    text = text.replace("~", "").replace("—", " ")
    return re.sub(r'[¿¡";:,.!?]+', " ", text.casefold()).split()


def word_error_rate(reference: str, prediction: str) -> float:
    reference_words = normalise(reference)
    prediction_words = normalise(prediction)
    previous = list(range(len(prediction_words) + 1))
    for row_number, reference_word in enumerate(reference_words, start=1):
        current = [row_number]
        for column, prediction_word in enumerate(prediction_words, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_word != prediction_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words) if reference_words else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    metadata_path = DATA_DIR / "metadata.tsv"
    clips_dir = DATA_DIR / "clips"
    if not metadata_path.is_file():
        parser.error(f"Missing development metadata: {metadata_path}")
    if not clips_dir.is_dir():
        parser.error(f"Missing clips directory: {clips_dir}")

    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))[: args.limit]
    if not rows:
        parser.error(f"No rows found in {metadata_path}")

    results: list[dict[str, str | float]] = []
    for index, row in enumerate(rows, start=1):
        audio_path = clips_dir / row["audio_filename"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio clip does not exist: {audio_path}")
        prediction = transcribe(audio_path)
        reference = row["transcript"]
        results.append(
            {
                "audio_filename": row["audio_filename"],
                "reference": reference,
                "prediction": prediction,
                "language": row.get("language", ""),
                "wer": word_error_rate(reference, prediction),
            }
        )
        print(f"Processed {index}/{len(rows)} clips", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audio_filename", "reference", "prediction", "language", "wer"),
        )
        writer.writeheader()
        writer.writerows(results)
    mean_wer = sum(float(row["wer"]) for row in results) / len(results)
    print(f"Wrote {len(results)} rows to {args.output}; mean WER={mean_wer:.4f}")


if __name__ == "__main__":
    main()
