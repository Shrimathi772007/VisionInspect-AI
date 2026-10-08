#!/bin/sh
# Backend container start-up: check the mounted models, wait for PostgreSQL, apply migrations, start uvicorn.
# With arguments (e.g. `docker compose run --rm backend python scripts/seed_demo_data.py ...`) it waits for the
# database and runs the command instead of the server.
set -eu

MODELS_DIR=/app/ai_models
PRETRAINED_DIR="$MODELS_DIR/_pretrained"

if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    echo "WARNING: $MODELS_DIR is missing or empty. AI_MODELS_DIR must point at backend/ai_models on the host." >&2
    echo "WARNING: The API will start, but every AI prediction will fail until the models are mounted." >&2
else
    for weights in wide_resnet50_2-95faca4d.pth resnet18-f37072fd.pth; do
        if [ ! -f "$PRETRAINED_DIR/$weights" ]; then
            echo "WARNING: $PRETRAINED_DIR/$weights is missing; models that use this backbone cannot be served." >&2
        fi
    done
fi

if [ ! -d "${DATASET_ROOT:-/dataset}" ] || [ -z "$(ls -A "${DATASET_ROOT:-/dataset}" 2>/dev/null)" ]; then
    echo "WARNING: ${DATASET_ROOT:-/dataset} is missing or empty; the dataset browser and import will not work." >&2
fi

DB_WAIT_SECONDS="${DB_WAIT_SECONDS:-60}"
echo "Waiting for PostgreSQL at ${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432} (up to ${DB_WAIT_SECONDS}s)..."
elapsed=0
until python -c "from app.database.session import engine; engine.connect().close()" >/dev/null 2>&1; do
    elapsed=$((elapsed + 2))
    if [ "$elapsed" -ge "$DB_WAIT_SECONDS" ]; then
        echo "ERROR: PostgreSQL not reachable after ${DB_WAIT_SECONDS}s. Check POSTGRES_* settings and the db service." >&2
        exit 1
    fi
    sleep 2
done
echo "PostgreSQL is reachable."

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "Applying database migrations (alembic upgrade head)..."
alembic upgrade head
echo "Migrations applied. Current revision: $(alembic current 2>/dev/null | tail -n 1)"

echo "Starting uvicorn with ${UVICORN_WORKERS:-1} worker(s)."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "${UVICORN_WORKERS:-1}" \
    --proxy-headers \
    --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}"
