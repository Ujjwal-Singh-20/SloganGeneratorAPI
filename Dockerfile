# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git-lfs && apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy application code and model
COPY main.py .
COPY config.json .
COPY model.safetensors .
COPY tokenizer_config.json .
COPY vocab.json .
COPY merges.txt .
COPY generation_config.json .
COPY special_tokens_map.json .

# Fix permissions
RUN chmod -R 755 /

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

#CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "7860"]

# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y git-lfs && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt && \
#     python -m spacy download en_core_web_sm

# # Copy all application files
# COPY . .

# # Expose port for Gradio
# EXPOSE 7860

# # Command to run the Gradio app
# CMD ["python", "app.py"]
