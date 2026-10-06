# VisionInspect AI backend (FastAPI + PyTorch CPU).
# Not in the image, mounted at run time (see docker-compose.yml):
#   /app/ai_models  trained models + _pretrained/ backbone weights (read-only)
#   /dataset        MVTec AD (read-only)
#   /app/storage    uploads and heatmaps (writable)
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    UPLOAD_STORAGE_ROOT=/app/storage/uploads \
    HEATMAP_STORAGE_ROOT=/app/storage/heatmaps \
    DATASET_ROOT=/dataset

WORKDIR /app

# torch first, from PyTorch's CPU-only index, so no CUDA libraries are downloaded. The "+cpu" build satisfies
# the torch==<version> pin, so the requirements install below keeps it.
COPY requirements.txt ./
RUN TORCH_VERSION="$(sed -n 's/^torch==\([^ ]*\).*/\1/p' requirements.txt)" \
    && test -n "$TORCH_VERSION" \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && pip install -r requirements.txt

# Application code. Owned by root, so the app user cannot modify it. scripts/experiments/ is copied unchanged:
# serving verifies the SHA-256 of two files in scripts/experiments/patchcore_wrn50/ before loading Grid.
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY scripts ./scripts
COPY docker ./docker
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Fails the build if a pin does not match, torch is not the CPU build, or a Grid study file changed.
RUN python docker/check_image.py

RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod 0755 /usr/local/bin/docker-entrypoint.sh \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/ai_models /app/storage/uploads /app/storage/heatmaps /dataset \
    && chown -R app:app /app/storage

USER app

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
