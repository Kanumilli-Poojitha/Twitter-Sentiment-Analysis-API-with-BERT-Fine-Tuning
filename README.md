# Twitter/Review Sentiment Analysis with BERT (Production-Style)

This project provides an end-to-end sentiment analysis system using Hugging Face Transformers and a production-style MLOps layout.

## Features

- Dataset: IMDB from Hugging Face (`datasets`)
- Model: configurable BERT family model (default: `distilbert-base-uncased`)
- Preprocessing pipeline that generates:
	- `data/processed/train.csv`
	- `data/processed/test.csv`
- Training pipeline that:
	- fine-tunes the model
	- saves model/tokenizer artifacts to `model_output/`
	- writes `results/metrics.json`
	- writes `results/run_summary.json`
- FastAPI backend:
	- `GET /health`
	- `POST /predict`
- Batch prediction script that reads input CSV and writes predictions CSV
- Streamlit UI that sends requests to FastAPI
- Fully containerized deployment:
	- `Dockerfile.api`
	- `Dockerfile.ui`
	- `docker-compose.yml` with health checks
- Environment variable-driven configuration (`.env`)

## Project Structure

```text
.
|-- data/
|   |-- processed/
|   |-- raw/
|   `-- unseen/
|-- model_output/
|-- results/
|-- scripts/
|   |-- preprocess.py
|   |-- train.py
|   |-- batch_predict.py
|   `-- validate_artifacts.py
|-- src/
|   |-- api.py
|   `-- ui.py
|-- tests/
|-- Dockerfile.api
|-- Dockerfile.ui
|-- docker-compose.yml
|-- requirements.txt
|-- .env.example
|-- VALIDATION_CHECKLIST.md
`-- README.md
```

## Environment Configuration

1. Copy `.env.example` to `.env`.
2. Adjust values as needed.

Key variables:

- `MODEL_NAME` (default `distilbert-base-uncased`)
- `TRAIN_CSV_PATH`, `TEST_CSV_PATH`
- `MODEL_OUTPUT_DIR`
- `METRICS_PATH`, `RUN_SUMMARY_PATH`
- `API_PORT`, `UI_PORT`, `API_BASE_URL`

## Local Run (Without Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

1. Preprocess IMDB dataset:

```bash
python scripts/preprocess.py
```

2. Fine-tune model and generate artifacts:

```bash
python scripts/train.py
```

3. Validate required artifacts:

```bash
python scripts/validate_artifacts.py
```

4. Start FastAPI:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

5. Start Streamlit UI (new terminal):

```bash
streamlit run src/ui.py --server.address 0.0.0.0 --server.port 8501
```

For local runs, set `API_BASE_URL=http://localhost:8000` in `.env`.

6. Run batch prediction:

```bash
python scripts/batch_predict.py --input-file data/unseen/predict_data.csv --output-file results/predictions.csv
```

## Docker Run

1. Create `.env` from `.env.example`.
2. Build and run services:

```bash
docker-compose up --build
```

Services:

- API: `http://localhost:${API_PORT}`
- UI: `http://localhost:${UI_PORT}`

Docker Compose overrides `API_BASE_URL` for the UI container to `http://api:8000`.

Sample batch input file:

- `data/unseen/predict_data.csv`

## API Contract

### `GET /health`

Returns model readiness and service health.

### `POST /predict`

Request:

```json
{
	"text": "This movie was absolutely fantastic."
}
```

Response:

```json
{
	"label": "LABEL_1",
	"sentiment": "positive",
	"score": 0.9981
}
```

## Error Handling and Robustness

- Input validation in API and scripts (missing columns, empty inputs, invalid env values)
- Startup model loading with degraded health mode if model is unavailable
- Explicit artifact verification after training
- JSON integrity validation in artifact checker script

## Expected Artifacts

After preprocessing + training:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `model_output/config.json`
- `model_output/tokenizer_config.json`
- `model_output/model.safetensors` (or `model_output/pytorch_model.bin`)
- `results/metrics.json`
- `results/run_summary.json`

## Notes

- Default training settings are lightweight for easier iteration.
- Increase `TRAIN_SAMPLE_SIZE`, `NUM_EPOCHS`, and batch sizes for stronger production performance.

Demo video:
https://drive.google.com/file/d/1lRLRSoZzoZ239y189y8siNqnh3-ggtNT/view?usp=sharing

Live demo:

https://drive.google.com/file/d/1-dK4GrUNi0_nHDaoL5pyFTiupo87yOvi/view?usp=sharing


Author
K.Poojitha


commands used for testing:

python scripts/preprocess.py

dir data\processed

python scripts/train.py

dir model_output

type results\metrics.json

type results\run_summary.json

python -m uvicorn src.api:app --reload

http://localhost:8000/docs

python scripts/batch_predict.py --input-file data/unseen/predict_data.csv --output-file results/predictions.csv

python -m streamlit run src/ui.py

http://localhost:8501

I really enjoyed this movie

docker compose up --build

docker ps