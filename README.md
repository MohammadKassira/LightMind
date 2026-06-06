<<<<<<< HEAD
# LightMind
Adaptive traffic signal control using graph-based RL with message passing over a real road network simulated in SUMO.
=======
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
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Stop
```bash
docker-compose down
```

---

## Running without Docker (local development)

**Terminal 1 — Backend:**
```bash
cd traffic-rl-website/backend
export SUMO_HOME=/usr/share/sumo  # or your local SUMO path
source .venv/bin/activate
uvicorn main:app --reload
```

**Terminal 2 — Frontend:**
```bash
cd traffic-rl-website/frontend
npm run dev
```

---

## Project structure

```
traffic-rl-website/
├── backend/          FastAPI + SUMO training pipeline
│   ├── main.py
│   ├── routers/
│   ├── services/
│   └── traffic_rl/   Independent DQN v2 controller
└── frontend/         Vite + React UI
    └── src/
        ├── App.jsx
        └── components/
```
>>>>>>> 7806029d18c60a3ee9a644209ef7638ec1d889b7
