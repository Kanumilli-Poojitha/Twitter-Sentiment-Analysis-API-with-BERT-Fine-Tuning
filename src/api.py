import logging
import os
import threading
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline


load_dotenv()


logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO"),
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("api")


MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "model_output")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))


class PredictRequest(BaseModel):
	text: str = Field(..., min_length=1, description="Input text for sentiment prediction")


class PredictResponse(BaseModel):
	label: str
	sentiment: str
	score: float


app = FastAPI(title="Sentiment Analysis API", version="1.0.0")
_model_lock = threading.Lock()
_classifier = None


def _resolve_model_source() -> str:
	return MODEL_OUTPUT_DIR if os.path.exists(MODEL_OUTPUT_DIR) else MODEL_NAME


def _normalize_sentiment(label: str) -> str:
	label_upper = label.upper()
	if label_upper in {"POSITIVE", "LABEL_1"}:
		return "positive"
	if label_upper in {"NEGATIVE", "LABEL_0"}:
		return "negative"
	return label.lower()


def _load_model() -> None:
	global _classifier
	model_source = _resolve_model_source()
	LOGGER.info("Loading sentiment model from: %s", model_source)
	_classifier = pipeline("text-classification", model=model_source, tokenizer=model_source)


@app.on_event("startup")
def startup_event() -> None:
	try:
		_load_model()
	except Exception as exc:
		LOGGER.exception("Failed to load model at startup: %s", exc)


@app.get("/health")
def health() -> Dict[str, Any]:
	return {
		"status": "ok" if _classifier is not None else "degraded",
		"model_loaded": _classifier is not None,
		"model_source": _resolve_model_source(),
	}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
	global _classifier

	text = payload.text.strip()
	if not text:
		raise HTTPException(status_code=400, detail="Input text must not be empty")

	if _classifier is None:
		try:
			_load_model()
		except Exception as exc:
			LOGGER.exception("Model unavailable: %s", exc)
			raise HTTPException(status_code=503, detail="Model is not available") from exc

	try:
		with _model_lock:
			result = _classifier(text, truncation=True, max_length=MAX_LENGTH)[0]

		label = str(result.get("label", "unknown"))
		score = float(result.get("score", 0.0))
		sentiment = _normalize_sentiment(label)
		return PredictResponse(label=label, sentiment=sentiment, score=score)
	except HTTPException:
		raise
	except Exception as exc:
		LOGGER.exception("Prediction failed: %s", exc)
		raise HTTPException(status_code=500, detail="Prediction failed") from exc
