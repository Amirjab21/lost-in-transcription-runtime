#!/usr/bin/env python3
"""Score a standard competition submission CSV against development metadata.

This post-processing tool does not participate in model inference. It writes
per-clip WER and reports corpus WER using the runtime repository's scorer
normalization rules.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def normalise(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\(\?+\)", " ", text)
    text = re.sub(r"\(([^()]*)\)", r"\1", text)
    text = text.replace("~", "").replace("#x27;", "'")
    text = re.sub(r'[¿¡";:]+', " ", text)

    def lowercase_initial(match: re.Match[str]) -> str:
        delimiter, first, second = match.group(1), match.group(2), match.group(3)
        return delimiter + (first.lower() if first.isupper() and not (second and second.isupper()) else first) + second

    text = re.sub(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)", lowercase_initial, text)
    text = re.sub(r",+", " ", text.replace("—", ", "))
    text = re.sub(r"[!?]+", " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    return re.sub(r"  +", " ", text).split()


def error_counts(reference: str, prediction: str) -> tuple[int, int]:
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
    return previous[-1], len(reference_words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REPOSITORY_ROOT / "data" / "data" / "metadata.tsv")
    parser.add_argument("--submission", type=Path, default=REPOSITORY_ROOT / "submission" / "submission.csv")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "model" / "submission_evaluation.csv")
    args = parser.parse_args()

    with args.metadata.open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle, delimiter="\t"))
    with args.submission.open(encoding="utf-8", newline="") as handle:
        predictions = {row["audio_filename"]: row["transcript"] for row in csv.DictReader(handle)}

    results, total_errors, total_words = [], 0, 0
    for row in metadata:
        filename, reference = row["audio_filename"], row["transcript"]
        if filename not in predictions:
            raise ValueError(f"Missing prediction for {filename}")
        prediction = predictions[filename]
        errors, words = error_counts(reference, prediction)
        total_errors += errors
        total_words += words
        results.append({
            "audio_file_path": str(args.metadata.parent / "clips" / filename),
            "ground_truth_transcript": reference,
            "language_labels": row.get("language", ""),
            "predicted_transcript": prediction,
            "wer": errors / words if words else 0.0,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "audio_file_path", "ground_truth_transcript", "language_labels", "predicted_transcript", "wer",
        ))
        writer.writeheader()
        writer.writerows(results)
    print(f"Corpus WER: {total_errors / total_words:.6f} ({total_errors}/{total_words} word errors)")
    print(f"Wrote per-clip WER to {args.output}")


if __name__ == "__main__":
    main()
