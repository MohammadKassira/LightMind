FROM python:3.11-slim

# Install system dependencies including SUMO and display stack
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    wget \
    curl \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    fluxbox \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set environment
ENV SUMO_HOME=/usr/share/sumo
ENV DISPLAY=:99
ENV ML_PROJECT_ROOT=/app

# Set working directory (== ML_PROJECT_ROOT inside the container)
WORKDIR /app

# Copy requirements and install Python dependencies
COPY website/backend/requirements.txt .
# Install PyTorch CPU-only wheel first (avoids downloading the large CUDA build)
RUN pip install --no-cache-dir --timeout 600 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu
# Install remaining requirements (torch already satisfied above)
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

# Copy backend code
COPY website/backend/ .

# Copy model directory so model/train.py and its dependencies are accessible
COPY model/ model/

# Copy supervisord config
COPY supervisord.conf supervisord.conf

# Create data directories
RUN mkdir -p data/web_jobs data/uploads data/sessions static

# Expose port
EXPOSE 7860

# Start via supervisord (manages Xvfb, x11vnc, noVNC, uvicorn)
CMD ["supervisord", "-c", "supervisord.conf"]
