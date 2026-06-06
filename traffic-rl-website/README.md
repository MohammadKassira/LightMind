# LightMind

LightMind is an MVP web dashboard for professional traffic reinforcement learning demos. This build uses fake live training data and local JSON/session files so the backend and frontend contracts are ready before real SUMO and RL integration.

## Stack

- Backend: FastAPI
- Frontend: React + Vite
- Styling: Tailwind CSS
- Map: React Leaflet
- Charts: Chart.js

## Features

- OSM upload flow with session creation
- Optional demand upload for `.xlsx` or `.csv`
- Fake live training stream over WebSocket
- Live traffic map with moving vehicles and signal states
- Streaming LightMind vs baseline metrics
- Hourly results comparison for 24 hours
- Local filesystem storage with JSON session state

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API Overview

- `POST /api/upload/osm`
- `POST /api/upload/demand`
- `POST /api/train`
- `GET /api/results/{session_id}`
- `WS /ws/live/{session_id}`

## Notes

- No database is included in this MVP.
- No authentication is included in this MVP.
- No real SUMO integration is included yet.
- No real RL model loading is included yet.
- Session artifacts are written under `backend/data/uploads/` and `backend/data/sessions/`.
