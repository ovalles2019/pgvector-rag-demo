#!/bin/bash
set -e

export DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER:-rag}:${POSTGRES_PASSWORD:-ragpassword}@127.0.0.1:5432/${POSTGRES_DB:-ragdb}}"

echo "Starting PostgreSQL..."
docker-entrypoint.sh postgres &

echo "Waiting for Postgres + pgvector + seeded chunks (first boot may take 2–4 min)..."
for i in $(seq 1 150); do
  if ! pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; then
    sleep 2
    continue
  fi

  HAS_VECTOR=$(psql -v ON_ERROR_STOP=0 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
    "SELECT 1 FROM pg_extension WHERE extname = 'vector'" 2>/dev/null || echo "0")
  CHUNKS=$(psql -v ON_ERROR_STOP=0 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
    "SELECT COUNT(*) FROM chunks" 2>/dev/null || echo "0")

  if [ "${HAS_VECTOR}" = "1" ] && [ "${CHUNKS:-0}" -gt 0 ] 2>/dev/null; then
    echo "Ready: ${CHUNKS} chunks indexed (after ~$((i * 2))s)"
    break
  fi

  sleep 2
done

# Fallback if initdb scripts did not run (e.g. volume already existed without data)
CHUNKS=$(psql -v ON_ERROR_STOP=0 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
  "SELECT COUNT(*) FROM chunks" 2>/dev/null || echo "0")
if [ "${CHUNKS:-0}" -eq 0 ] 2>/dev/null; then
  echo "No chunks found — running ingest fallback..."
  psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "CREATE EXTENSION IF NOT EXISTS vector;"
  cd /app
  python3 main.py init-db
  python3 main.py ingest data/sample_docs --category tutorials
fi

CHUNKS=$(psql -v ON_ERROR_STOP=0 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
  "SELECT COUNT(*) FROM chunks" 2>/dev/null || echo "0")
if [ "${CHUNKS:-0}" -eq 0 ] 2>/dev/null; then
  echo "ERROR: database seeding did not complete"
  exit 1
fi

cd /app
PORT="${PORT:-10000}"
echo "Starting Streamlit on port ${PORT}..."
exec python3 -m streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
