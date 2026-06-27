# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for compilation (e.g. for certain C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose ports (8000 for FastAPI API, 8501 for Streamlit Dashboard)
EXPOSE 8000
EXPOSE 8501

# Set default command to run the FastAPI app
# To run Streamlit instead, override the CMD with: ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["python", "api.py"]
