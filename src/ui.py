import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))


st.set_page_config(page_title="Sentiment Analysis UI", page_icon=":speech_balloon:", layout="centered")
st.title("Sentiment Analysis with BERT")
st.caption("Streamlit frontend that calls the FastAPI prediction service")


with st.sidebar:
	st.subheader("Configuration")
	api_url = st.text_input("FastAPI Base URL", value=API_BASE_URL)
	st.caption("Example: http://localhost:8000")


def fetch_health(base_url: str) -> tuple[bool, str]:
	try:
		response = requests.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_SECONDS)
		response.raise_for_status()
		payload = response.json()
		return True, f"Health: {payload}"
	except requests.RequestException as exc:
		return False, f"Health check failed: {exc}"


health_ok, health_message = fetch_health(api_url)
if health_ok:
	st.success(health_message)
else:
	st.warning(health_message)


text_input = st.text_area("Enter text", placeholder="Type a movie review...")
submit = st.button("Predict Sentiment", type="primary")


if submit:
	cleaned = text_input.strip()
	if not cleaned:
		st.error("Please enter non-empty text before predicting.")
	else:
		try:
			response = requests.post(
				f"{api_url}/predict",
				json={"text": cleaned},
				timeout=REQUEST_TIMEOUT_SECONDS,
			)
			if response.status_code >= 400:
				st.error(f"API returned error {response.status_code}: {response.text}")
			else:
				prediction = response.json()
				st.success(
					f"Sentiment: {prediction.get('sentiment', 'unknown')} | "
					f"Label: {prediction.get('label', 'unknown')} | "
					f"Confidence: {prediction.get('score', 0.0):.4f}"
				)
		except requests.RequestException as exc:
			st.error(f"Request failed: {exc}")
