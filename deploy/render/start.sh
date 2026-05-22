#!/bin/bash
set -e

# Unix socket works during and after init; TCP 127.0.0.1 is only up after init completes.
PG_SOCKET_DIR=/var/run/postgresql
export PGHOST="${PG_SOCKET_DIR}"
export DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER:-rag}:${POSTGRES_PASSWORD:-ragpassword}@/${POSTGRES_DB:-ragdb}?host=${PG_SOCKET_DIR}}"
SEED_MARKER=/var/lib/postgresql/data/.demo_seeded

echo "Starting PostgreSQL..."
docker-entrypoint.sh postgres &
PG_PID=$!

echo "Waiting for PostgreSQL (TCP — init must finish first)..."
TCP_READY=0
for i in $(seq 1 180); do
  if ! kill -0 "${PG_PID}" 2>/dev/null; then
    echo "ERROR: PostgreSQL process exited during startup"
    exit 1
  fi
  if pg_isready -h 127.0.0.1 -p 5432 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; then
    TCP_READY=1
    echo "PostgreSQL accepting TCP connections (after ~$((i * 2))s)"
    break
  fi
  sleep 2
done
if [ "${TCP_READY}" != "1" ]; then
  echo "ERROR: PostgreSQL did not become ready in time"
  exit 1
fi

if [ ! -f "${SEED_MARKER}" ]; then
  echo "Seeding demo knowledge base (first boot)..."
  cd /app
  psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;"
  python3 main.py init-db
  python3 main.py ingest data/sample_docs --category tutorials
  touch "${SEED_MARKER}"
  echo "Demo seed complete."
fi

CHUNKS=$(psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
  "SELECT COUNT(*) FROM chunks")
if [ "${CHUNKS:-0}" -eq 0 ] 2>/dev/null; then
  echo "ERROR: chunks table is empty after seeding"
  exit 1
fi
echo "Ready: ${CHUNKS} chunks indexed."

cd /app
PORT="${PORT:-10000}"
echo "Starting Streamlit on port ${PORT}..."
exec python3 -m streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
