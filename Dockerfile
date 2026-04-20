# HuggingFace Spaces uses port 7860
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces: set LLM_PROVIDER=groq and GROQ_API_KEY in Space secrets
ENV LLM_PROVIDER=groq
ENV CHROMA_PERSIST_DIR=/app/vectorstore/chroma_db

RUN mkdir -p /app/vectorstore/chroma_db /app/data/sample_docs

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
