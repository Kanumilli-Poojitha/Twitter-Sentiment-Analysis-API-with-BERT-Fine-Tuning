import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("preprocess")


def _get_int_env(name: str, default: int) -> int:
	value = os.getenv(name, str(default))
	try:
		parsed = int(value)
		if parsed <= 0:
			raise ValueError
		return parsed
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be a positive integer. Got: {value}") from exc


@dataclass
class PreprocessConfig:
	train_csv_path: str = os.getenv("TRAIN_CSV_PATH", "data/processed/train.csv")
	test_csv_path: str = os.getenv("TEST_CSV_PATH", "data/processed/test.csv")
	train_sample_size: int = _get_int_env("TRAIN_SAMPLE_SIZE", 10000)
	test_sample_size: int = _get_int_env("TEST_SAMPLE_SIZE", 2000)
	random_seed: int = _get_int_env("RANDOM_SEED", 42)
	text_column: str = os.getenv("TEXT_COLUMN", "text")
	label_column: str = os.getenv("LABEL_COLUMN", "label")


def _ensure_parent_dir(file_path: str) -> None:
	Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def _select_subset(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
	if sample_size >= len(df):
		return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
	return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def _validate_columns(df: pd.DataFrame, text_column: str, label_column: str) -> None:
	required = {text_column, label_column}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def build_processed_csvs(config: PreprocessConfig) -> None:
	LOGGER.info("Loading IMDB dataset from Hugging Face Hub")
	dataset: DatasetDict = load_dataset("imdb")

	train_df = dataset["train"].to_pandas()
	test_df = dataset["test"].to_pandas()
	_validate_columns(train_df, config.text_column, config.label_column)
	_validate_columns(test_df, config.text_column, config.label_column)

	processed_train = _select_subset(train_df, config.train_sample_size, config.random_seed)
	processed_test = _select_subset(test_df, config.test_sample_size, config.random_seed)

	_ensure_parent_dir(config.train_csv_path)
	_ensure_parent_dir(config.test_csv_path)

	processed_train[[config.text_column, config.label_column]].to_csv(config.train_csv_path, index=False)
	processed_test[[config.text_column, config.label_column]].to_csv(config.test_csv_path, index=False)

	LOGGER.info("Saved train CSV: %s (rows=%d)", config.train_csv_path, len(processed_train))
	LOGGER.info("Saved test CSV: %s (rows=%d)", config.test_csv_path, len(processed_test))


def main() -> None:
	try:
		config = PreprocessConfig()
		build_processed_csvs(config)
		LOGGER.info("Preprocessing completed successfully")
	except Exception as exc:
		LOGGER.exception("Preprocessing failed: %s", exc)
		raise


if __name__ == "__main__":
	main()
