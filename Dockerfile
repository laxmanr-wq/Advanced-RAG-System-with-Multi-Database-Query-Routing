FROM python:3.12-slim

WORKDIR /app

# uv for fast, locked installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY rag_agent ./rag_agent
COPY app.py .
COPY .streamlit ./.streamlit

# Persist the vector store across restarts
ENV QDRANT_PATH=/data/qdrant
VOLUME /data/qdrant

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
