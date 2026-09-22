"""Download OpenAI's SimpleQA CSV into the local dataset format."""

import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen


SOURCE = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def main() -> None:
    with urlopen(SOURCE, timeout=30) as response:
        rows = list(csv.DictReader(io.TextIOWrapper(response, encoding="utf-8")))
    if len(rows) < 100 or not {"metadata", "problem", "answer"} <= rows[0].keys():
        raise ValueError("Unexpected SimpleQA dataset format")

    DATASET_DIR.mkdir(exist_ok=True)
    for name, data in (("simpleqa.json", rows), ("simpleqa_small.json", rows[:100])):
        with (DATASET_DIR / name).open("w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False)
    print(f"Saved {len(rows)} SimpleQA rows to {DATASET_DIR}")


if __name__ == "__main__":
    main()
