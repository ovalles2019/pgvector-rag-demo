FROM pgvector/pgvector:pg16

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt \
    && python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
COPY deploy/render/start.sh /usr/local/bin/start-demo.sh
RUN chmod +x /usr/local/bin/start-demo.sh

ENV POSTGRES_USER=rag \
    POSTGRES_PASSWORD=ragpassword \
    POSTGRES_DB=ragdb \
    DATABASE_URL=postgresql://rag:ragpassword@127.0.0.1:5432/ragdb \
    PGVECTOR_RAG_DEMO=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 10000

USER postgres
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/usr/local/bin/start-demo.sh"]
