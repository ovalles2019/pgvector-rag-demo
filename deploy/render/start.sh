#!/bin/bash
set -e

export DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER:-rag}:${POSTGRES_PASSWORD:-ragpassword}@127.0.0.1:5432/${POSTGRES_DB:-ragdb}}"

echo "Starting PostgreSQL..."
docker-entrypoint.sh postgres &
for _ in $(seq 1 30); do
  if pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

cd /app

if [ "${PGVECTOR_RAG_DEMO}" = "1" ]; then
  echo "Initializing demo knowledge base..."
  python3 main.py init-db
  python3 main.py ingest data/sample_docs --category tutorials
fi

PORT="${PORT:-10000}"
exec python3 -m streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
