# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY website/frontend/package*.json ./
RUN npm ci
COPY website/frontend/ .
# Empty string → same-origin API calls (no host prefix), correct for HF Spaces
ENV VITE_API_URL=""
RUN npm run build

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    sumo \
    sumo-gui \
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

ENV SUMO_HOME=/usr/share/sumo
ENV DISPLAY=:99
ENV ML_PROJECT_ROOT=/app

WORKDIR /app

COPY website/backend/requirements.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

COPY website/backend/ .
COPY model/ model/

# Built frontend served as static files by FastAPI
COPY --from=frontend-builder /frontend/dist static/

COPY supervisord.conf supervisord.conf

RUN mkdir -p data/web_jobs data/uploads data/sessions

EXPOSE 7860

CMD ["supervisord", "-c", "supervisord.conf"]
