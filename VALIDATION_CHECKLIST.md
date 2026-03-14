# Validation Checklist

Use this checklist to confirm the project meets all required deliverables.

## Dataset and Preprocessing

- [ ] IMDB dataset is loaded from Hugging Face (`datasets.load_dataset('imdb')`)
- [ ] `scripts/preprocess.py` runs successfully
- [ ] `data/processed/train.csv` exists
- [ ] `data/processed/test.csv` exists

## Training and Artifacts

- [ ] `scripts/train.py` fine-tunes BERT/DistilBERT
- [ ] Model artifacts are saved to `model_output/`
- [ ] `results/metrics.json` exists and contains valid JSON
- [ ] `results/run_summary.json` exists and contains valid JSON
- [ ] `python scripts/validate_artifacts.py` passes

## API

- [ ] FastAPI app starts successfully
- [ ] `GET /health` returns success payload
- [ ] `POST /predict` returns label, sentiment, and score

## Batch Prediction

- [ ] `scripts/batch_predict.py` reads a CSV input
- [ ] Predictions CSV is generated at configured output path
- [ ] Output contains `predicted_label`, `predicted_sentiment`, and `confidence`

## Streamlit UI

- [ ] Streamlit UI starts successfully
- [ ] UI can call FastAPI `/health`
- [ ] UI can call FastAPI `/predict`

## Containerization

- [ ] `Dockerfile.api` builds successfully
- [ ] `Dockerfile.ui` builds successfully
- [ ] `docker-compose.yml` starts both services
- [ ] Docker Compose health checks pass for both services

## Configuration and Quality

- [ ] All runtime configuration is controlled via environment variables
- [ ] Scripts and services include robust error handling
- [ ] Code is modular and beginner-readable
- [ ] `docker-compose up --build` runs end-to-end
