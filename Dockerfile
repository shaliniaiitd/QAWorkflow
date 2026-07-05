# Dockerfile -- containerizes the Streamlit app.
# Does NOT include Ollama itself (see docker-compose.yml for that) --
# this container just needs to reach an Ollama instance via OLLAMA_BASE_URL.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# OLLAMA_BASE_URL is set by docker-compose.yml when running via compose;
# defaults to localhost if run standalone (won't work unless Ollama is
# reachable at that address from inside the container).
ENV OLLAMA_BASE_URL=http://localhost:11434

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]