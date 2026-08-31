# Hugging Face Space image.
#
# Separate from the root Dockerfile because a Space has its own conventions and
# mixing them would make the local `docker compose up` worse to satisfy a host:
# the app must listen on 7860, and it must run as UID 1000 with a writable
# HOME, because a Space's filesystem is read-only except for that user's home.
#
# Generation goes to a hosted OpenAI-compatible endpoint rather than a local
# model. A free Space has no persistent volume, so Ollama would re-download two
# gigabytes on every cold start — a minute of waiting, repeated, for a model
# weaker than the hosted one. The local path stays the default in the repo,
# where it is the point; here it would only be theatre.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run containers as UID 1000. Anything written outside this user's home
# fails at runtime, including the ONNX model cache — which is why HOME is set
# explicitly rather than left to the base image.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user pyproject.toml ./
RUN pip install --no-cache-dir --user .

COPY --chown=user researchlens/ ./researchlens/
COPY --chown=user eval/ ./eval/
COPY --chown=user scripts/ ./scripts/

# The prebuilt index: passages and vectors, no PDFs. ~24 MB.
COPY --chown=user data/bundle/ ./data/bundle/

# Fetch the ONNX weights at build time rather than on first request.
#
# Measured in this image: the first query took 36 seconds, almost all of it
# downloading and loading the embedding and reranker models. A Space sleeps
# when idle, so that cost is not paid once — it is paid on every wake, by
# whoever happens to arrive first. About 150 MB of image against 30-odd
# seconds off every cold start is a trade worth making here, and the wrong one
# for the local build, where the image is pulled far more often than it is
# started cold.
# The cache path is set *before* the download. Setting it afterwards was a real
# bug: the build wrote the weights to fastembed's default /tmp/fastembed_cache,
# the runtime looked in the configured path, found nothing, and downloaded them
# again — so baking them in bought nothing while appearing to have worked.
ENV FASTEMBED_CACHE_PATH=/home/user/app/.cache/fastembed

RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2'); print('model weights cached')"

# fastembed caches under the current working directory unless told otherwise,
# so the cache is pinned where the build wrote it and the runtime can find it.

ENV RESEARCHLENS_MODE=demo \
    DATA_DIR=/home/user/app/data \
    PROVIDERS=hosted \
    ALLOWED_ORIGINS="https://asifuddin.com,https://www.asifuddin.com" \
    HOSTED_BASE_URL="https://router.huggingface.co/v1" \
    HOSTED_MODEL="Qwen/Qwen2.5-7B-Instruct"

EXPOSE 7860

# One worker: each holds its own copy of the embedding and reranker models, so
# a second doubles memory for no throughput on shared CPU.
CMD ["uvicorn", "researchlens.api.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
