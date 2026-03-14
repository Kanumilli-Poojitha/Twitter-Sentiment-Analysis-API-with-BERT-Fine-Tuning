import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()


logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("batch_predict")


def _get_int_env(name: str, default: int) -> int:
	value = os.getenv(name, str(default))
	try:
		parsed = int(value)
		if parsed <= 0:
			raise ValueError
		return parsed
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be a positive integer. Got: {value}") from exc


def _model_source() -> str:
	output_dir = os.getenv("MODEL_OUTPUT_DIR", "model_output")
	model_name = os.getenv("MODEL_NAME", "distilbert-base-uncased")
	return output_dir if Path(output_dir).exists() else model_name


def _normalize_sentiment(label: str) -> str:
	label_upper = label.upper()
	if label_upper in {"POSITIVE", "LABEL_1"}:
		return "positive"
	if label_upper in {"NEGATIVE", "LABEL_0"}:
		return "negative"
	return label.lower()


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run batch sentiment prediction on a CSV file.")
	parser.add_argument(
		"--input-file",
		dest="input_file",
		default=os.getenv("PREDICTION_INPUT_CSV", "data/unseen/input.csv"),
		help="Path to the input CSV file.",
	)
	parser.add_argument(
		"--output-file",
		dest="output_file",
		default=os.getenv("PREDICTION_OUTPUT_CSV", "results/predictions.csv"),
		help="Path to the output CSV file.",
	)
	parser.add_argument(
		"--text-column",
		dest="text_column",
		default=os.getenv("PREDICTION_TEXT_COLUMN", os.getenv("TEXT_COLUMN", "text")),
		help="Column name containing input text.",
	)
	parser.add_argument(
		"--max-length",
		dest="max_length",
		type=int,
		default=_get_int_env("MAX_LENGTH", 256),
		help="Maximum token length per example.",
	)
	parser.add_argument(
		"--batch-size",
		dest="batch_size",
		type=int,
		default=_get_int_env("PREDICTION_BATCH_SIZE", 32),
		help="Batch size for inference.",
	)
	return parser.parse_args()


def predict_from_csv(input_csv: str, output_csv: str, text_column: str, max_length: int, batch_size: int) -> None:
	if max_length <= 0:
		raise ValueError(f"max_length must be a positive integer. Got: {max_length}")
	if batch_size <= 0:
		raise ValueError(f"batch_size must be a positive integer. Got: {batch_size}")

	if not Path(input_csv).exists():
		raise FileNotFoundError(f"Input CSV not found: {input_csv}")

	df = pd.read_csv(input_csv)
	if text_column not in df.columns:
		raise ValueError(f"Input CSV must include text column '{text_column}'. Found columns: {list(df.columns)}")
	if df.empty:
		raise ValueError(f"Input CSV is empty: {input_csv}")

	model_source = _model_source()
	LOGGER.info("Loading model for batch inference: %s", model_source)
	classifier = pipeline("text-classification", model=model_source, tokenizer=model_source)

	texts: List[str] = df[text_column].fillna("").astype(str).tolist()
	all_predictions: List[Dict[str, float]] = []

	for start_idx in range(0, len(texts), batch_size):
		end_idx = min(start_idx + batch_size, len(texts))
		batch_texts = texts[start_idx:end_idx]
		LOGGER.info("Scoring rows %d to %d", start_idx, end_idx - 1)
		predictions = classifier(batch_texts, truncation=True, max_length=max_length)
		all_predictions.extend(predictions)

	output_df = df.copy()
	output_df["predicted_label"] = [pred.get("label", "unknown") for pred in all_predictions]
	output_df["predicted_sentiment"] = [
		_normalize_sentiment(str(pred.get("label", "unknown"))) for pred in all_predictions
	]
	output_df["confidence"] = [float(pred.get("score", 0.0)) for pred in all_predictions]

	Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
	output_df.to_csv(output_csv, index=False)
	LOGGER.info("Saved batch predictions: %s", output_csv)


def main() -> None:
	try:
		args = _parse_args()
		predict_from_csv(
			input_csv=args.input_file,
			output_csv=args.output_file,
			text_column=args.text_column,
			max_length=args.max_length,
			batch_size=args.batch_size,
		)
	except Exception as exc:
		LOGGER.exception("Batch prediction failed: %s", exc)
		raise


if __name__ == "__main__":
	main()
