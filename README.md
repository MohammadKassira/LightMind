---
title: Try2
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
---
# LightMind — AI Traffic Signal Control

LightMind is a full-stack AI traffic control system. Upload a SUMO network file, configure training, and watch the Independent DQN v2 agent learn to optimise traffic signals in real time.

## Running with Docker

### Prerequisites
- Docker Desktop installed
- Copy `.env.example` to `.env` and fill in your `ANTHROPIC_API_KEY`

### Start everything
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
docker-compose up --build
```

### Access the app
- Frontend: http://localhost:5173
- Backend API: http://localhost:7860
- API docs: http://localhost:7860/docs

### Stop
```bash
docker-compose down
```
