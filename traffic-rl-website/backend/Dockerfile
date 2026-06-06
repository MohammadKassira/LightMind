FROM python:3.11-slim

# Install system dependencies including SUMO
RUN apt-get update && apt-get install -y \
    sumo \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set SUMO_HOME
ENV SUMO_HOME=/usr/share/sumo

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Install PyTorch CPU-only wheel first (avoids downloading the large CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Install remaining requirements (torch already satisfied above)
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Create data directories
RUN mkdir -p data/web_jobs data/uploads data/sessions static

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
