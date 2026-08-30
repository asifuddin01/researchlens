# Python 3.12, not 3.13 or 3.14: onnxruntime — which fastembed builds on, and
# which is what keeps the retrieval service small enough to suspend — publishes
# wheels for 3.12 well ahead of newer releases. Building it from source in a
# container image is not a trade worth making.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# pdfplumber needs no system libraries; this is for the ONNX runtime's BLAS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies are copied and installed before the source, so editing a Python
# file does not invalidate the layer that took four minutes to build.
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY researchlens/ ./researchlens/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

# Runs unprivileged. The container parses PDFs supplied by whoever is using it,
# and a parser is exactly the kind of thing that should not be root.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/data \
 && chown -R app:app /app /home/app
USER app

EXPOSE 8000

CMD ["uvicorn", "researchlens.api.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
