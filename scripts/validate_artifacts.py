import json
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "model_output")
METRICS_PATH = os.getenv("METRICS_PATH", "results/metrics.json")
RUN_SUMMARY_PATH = os.getenv("RUN_SUMMARY_PATH", "results/run_summary.json")
TRAIN_CSV_PATH = os.getenv("TRAIN_CSV_PATH", "data/processed/train.csv")
TEST_CSV_PATH = os.getenv("TEST_CSV_PATH", "data/processed/test.csv")


def _exists_model_weights(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()


def validate() -> None:
    required_paths = [
        Path(TRAIN_CSV_PATH),
        Path(TEST_CSV_PATH),
        Path(METRICS_PATH),
        Path(RUN_SUMMARY_PATH),
        Path(MODEL_OUTPUT_DIR) / "config.json",
        Path(MODEL_OUTPUT_DIR) / "tokenizer_config.json",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if not _exists_model_weights(Path(MODEL_OUTPUT_DIR)):
        missing.append(
            f"{Path(MODEL_OUTPUT_DIR) / 'model.safetensors'} or {Path(MODEL_OUTPUT_DIR) / 'pytorch_model.bin'}"
        )

    if missing:
        print("Validation failed. Missing artifacts:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
        json.load(metrics_file)

    with open(RUN_SUMMARY_PATH, "r", encoding="utf-8") as summary_file:
        json.load(summary_file)

    print("Validation passed. All required artifacts exist and JSON files are valid.")


if __name__ == "__main__":
    validate()
