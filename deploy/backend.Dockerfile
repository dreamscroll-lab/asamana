# Backend image — Python API + WebSocket + engine. No frontend (deployed
# separately as its own nginx image). Build context is the repository root.
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install Python deps first (layer-cacheable before source copy).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source (explicit — no tests/, data/, .git, frontend/).
# tuning/ is dev tooling but ships too: the observability UI runs the trace
# audit as a `python -m tuning audit` subprocess (interaction/api/dev/jobs.py).
COPY agent/       agent/
COPY config/      config/
COPY core/        core/
COPY engine/      engine/
COPY interaction/ interaction/
COPY providers/   providers/
COPY world/       world/
COPY worlds/      worlds/
COPY examples/    examples/
COPY tuning/      tuning/
COPY main.py      ./

# No frontend/dist here → the backend serves the API only and returns a small
# info page at "/". data/ is mounted as a volume at runtime.

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/worlds')" || exit 1

CMD ["python", "main.py", "web", "--host", "0.0.0.0", "--port", "7860"]
