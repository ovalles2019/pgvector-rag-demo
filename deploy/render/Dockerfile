FROM pgvector/pgvector:pg16

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

COPY requirements.txt .
# CPU-only PyTorch keeps the image smaller and fits Render free tier RAM
RUN pip3 install --break-system-packages --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --break-system-packages --no-cache-dir -r requirements.txt \
    && python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
COPY deploy/render/docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/
COPY deploy/render/start.sh /usr/local/bin/start-demo.sh
RUN chmod +x /usr/local/bin/start-demo.sh \
    && chown -R postgres:postgres /app/.cache /var/lib/postgresql/data

ENV POSTGRES_USER=rag \
    POSTGRES_PASSWORD=ragpassword \
    POSTGRES_DB=ragdb \
    DATABASE_URL=postgresql://rag:ragpassword@/ragdb?host=/var/run/postgresql \
    PGVECTOR_RAG_DEMO=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 10000

USER postgres
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/usr/local/bin/start-demo.sh"]
