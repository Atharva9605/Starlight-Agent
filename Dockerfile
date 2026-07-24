# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for ChromaDB and Static files to ensure they are writable
RUN mkdir -p chroma_db static/page_images static/attachments out_emails_streamlit data/custom_templates
RUN chmod -R 777 chroma_db static/page_images static/attachments out_emails_streamlit data

# Hugging Face Spaces runs on port 7860
EXPOSE 7860

# Start the FastAPI API (for Lovable connection)
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
