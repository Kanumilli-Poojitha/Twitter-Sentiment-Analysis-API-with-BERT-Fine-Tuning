import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	DataCollatorWithPadding,
	Trainer,
	TrainingArguments,
)


load_dotenv()


logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("train")


def _get_int_env(name: str, default: int) -> int:
	value = os.getenv(name, str(default))
	try:
		return int(value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be an integer. Got: {value}") from exc


def _get_float_env(name: str, default: float) -> float:
	value = os.getenv(name, str(default))
	try:
		return float(value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be a float. Got: {value}") from exc


@dataclass
class TrainConfig:
	model_name: str = os.getenv("MODEL_NAME", "distilbert-base-uncased")
	train_csv_path: str = os.getenv("TRAIN_CSV_PATH", "data/processed/train.csv")
	test_csv_path: str = os.getenv("TEST_CSV_PATH", "data/processed/test.csv")
	model_output_dir: str = os.getenv("MODEL_OUTPUT_DIR", "model_output")
	results_dir: str = os.getenv("RESULTS_DIR", "results")
	metrics_path: str = os.getenv("METRICS_PATH", "results/metrics.json")
	run_summary_path: str = os.getenv("RUN_SUMMARY_PATH", "results/run_summary.json")
	text_column: str = os.getenv("TEXT_COLUMN", "text")
	label_column: str = os.getenv("LABEL_COLUMN", "label")
	max_length: int = _get_int_env("MAX_LENGTH", 256)
	num_epochs: int = _get_int_env("NUM_EPOCHS", 1)
	train_batch_size: int = _get_int_env("TRAIN_BATCH_SIZE", 8)
	eval_batch_size: int = _get_int_env("EVAL_BATCH_SIZE", 8)
	learning_rate: float = _get_float_env("LEARNING_RATE", 2e-5)
	weight_decay: float = _get_float_env("WEIGHT_DECAY", 0.01)
	logging_steps: int = _get_int_env("LOGGING_STEPS", 50)
	save_total_limit: int = _get_int_env("SAVE_TOTAL_LIMIT", 1)
	random_seed: int = _get_int_env("RANDOM_SEED", 42)


def _ensure_parent_dir(file_path: str) -> None:
	Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(dir_path: str) -> None:
	Path(dir_path).mkdir(parents=True, exist_ok=True)


def _load_csv(path: str, text_column: str, label_column: str) -> pd.DataFrame:
	if not Path(path).exists():
		raise FileNotFoundError(f"Input CSV not found: {path}")

	df = pd.read_csv(path)
	missing = {text_column, label_column} - set(df.columns)
	if missing:
		raise ValueError(f"CSV is missing required columns: {sorted(missing)} at {path}")
	if df.empty:
		raise ValueError(f"CSV is empty: {path}")

	return df[[text_column, label_column]].dropna().reset_index(drop=True)


def _compute_metrics(eval_pred: Any) -> Dict[str, float]:
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=-1)
	return {
		"accuracy": float(accuracy_score(labels, preds)),
		"precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
		"recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
		"f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
	}


def _build_hf_dataset(df: pd.DataFrame, text_column: str, label_column: str) -> Dataset:
	mapped_df = df.rename(columns={text_column: "text", label_column: "label"})
	return Dataset.from_pandas(mapped_df, preserve_index=False)


def _validate_training_outputs(config: TrainConfig) -> None:
	model_dir = Path(config.model_output_dir)
	metrics_file = Path(config.metrics_path)
	summary_file = Path(config.run_summary_path)

	model_files = [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"]
	required_exists = [
		model_dir / "config.json",
		model_dir / "tokenizer_config.json",
		metrics_file,
		summary_file,
	]

	missing = [str(path) for path in required_exists if not path.exists()]
	if not any(path.exists() for path in model_files):
		missing.append(f"{model_dir / 'model.safetensors'} or {model_dir / 'pytorch_model.bin'}")

	if missing:
		raise RuntimeError(f"Training finished but required artifacts are missing: {missing}")


def fine_tune_and_save(config: TrainConfig) -> None:
	_ensure_dir(config.model_output_dir)
	_ensure_dir(config.results_dir)
	_ensure_parent_dir(config.metrics_path)
	_ensure_parent_dir(config.run_summary_path)

	train_df = _load_csv(config.train_csv_path, config.text_column, config.label_column)
	test_df = _load_csv(config.test_csv_path, config.text_column, config.label_column)

	LOGGER.info("Loading tokenizer and model: %s", config.model_name)
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)
	model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)

	train_ds = _build_hf_dataset(train_df, config.text_column, config.label_column)
	test_ds = _build_hf_dataset(test_df, config.text_column, config.label_column)

	def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
		return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

	train_ds = train_ds.map(tokenize_fn, batched=True)
	test_ds = test_ds.map(tokenize_fn, batched=True)
	train_ds = train_ds.remove_columns(["text"])  # Keep only model inputs and labels.
	test_ds = test_ds.remove_columns(["text"])

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	training_args = TrainingArguments(
		output_dir=config.model_output_dir,
		eval_strategy="epoch",
		save_strategy="epoch",
		logging_strategy="steps",
		logging_steps=config.logging_steps,
		learning_rate=config.learning_rate,
		per_device_train_batch_size=config.train_batch_size,
		per_device_eval_batch_size=config.eval_batch_size,
		num_train_epochs=config.num_epochs,
		weight_decay=config.weight_decay,
		save_total_limit=config.save_total_limit,
		load_best_model_at_end=True,
		metric_for_best_model="f1",
		seed=config.random_seed,
		report_to=[],
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_ds,
		eval_dataset=test_ds,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=_compute_metrics,
	)

	LOGGER.info("Starting fine-tuning")
	train_result = trainer.train()
	eval_metrics = trainer.evaluate()

	LOGGER.info("Saving model artifacts to %s", config.model_output_dir)
	trainer.save_model(config.model_output_dir)
	tokenizer.save_pretrained(config.model_output_dir)

	metrics_payload = {
		"model_name": config.model_name,
		"train_metrics": {k: float(v) for k, v in train_result.metrics.items() if isinstance(v, (int, float))},
		"eval_metrics": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
	}

	run_summary_payload = {
		"timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"config": asdict(config),
		"num_train_rows": int(len(train_df)),
		"num_test_rows": int(len(test_df)),
		"artifacts": {
			"model_output_dir": config.model_output_dir,
			"metrics_path": config.metrics_path,
			"run_summary_path": config.run_summary_path,
		},
	}

	with open(config.metrics_path, "w", encoding="utf-8") as metrics_file:
		json.dump(metrics_payload, metrics_file, indent=2)

	with open(config.run_summary_path, "w", encoding="utf-8") as summary_file:
		json.dump(run_summary_payload, summary_file, indent=2)

	_validate_training_outputs(config)
	LOGGER.info("Training completed and required artifacts verified")


def main() -> None:
	try:
		config = TrainConfig()
		fine_tune_and_save(config)
	except Exception as exc:
		LOGGER.exception("Training failed: %s", exc)
		raise


if __name__ == "__main__":
	main()
