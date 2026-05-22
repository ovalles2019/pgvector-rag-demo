#!/bin/bash
set -e
# Runs once on first Postgres init (empty data volume).
cd /app
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:5432/${POSTGRES_DB}"
echo "Seeding demo documents..."
python3 main.py init-db
python3 main.py ingest data/sample_docs --category tutorials
echo "Demo ingest complete."
