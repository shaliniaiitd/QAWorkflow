# API regression suite

`test_api_regression.py` exercises the FastAPI route contracts with dependency calls mocked, so CI does not need Ollama, Chroma, or hosted model credentials. Launch the service with `uvicorn api:app --host 0.0.0.0 --port 8000`.
