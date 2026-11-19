# Who said what, website only: serves app/index.html and forwards /diarize to the diarization server that
# `uv run python app/server.py` runs natively on this machine (DIARIZE_URL in compose.yaml).
# server.py uses only the standard library in this mode, so the image needs no packages: no torch, no token.
FROM python:3.13-slim

WORKDIR /srv
COPY app/server.py app/index.html app/

ENV HOST=0.0.0.0 PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "app/server.py"]
