# Module 3 — REST API + Docker Deployment

## Overview
XGBoost failure prediction model deployed as a REST API using FastAPI, containerized with Docker for reproducible deployment.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/health` | GET | Health check for monitoring |
| `/predict` | POST | Predict failure probability |

## Input format
```json
{
  "type": "M",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551.0,
  "torque": 42.8,
  "tool_wear": 0.0
}
```

## Output format
```json
{
  "failure_probability": 0.0312,
  "prediction": "NO FAILURE",
  "risk_level": "LOW"
}
```

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Run with Docker
```bash
docker build -t predictive-maintenance-api .
docker run -p 8000:8000 predictive-maintenance-api
```

## Key decisions
- **Threshold: 0.3** — optimized in Module 1 via precision-recall curve. Lower than default 0.5 to maximize recall in a maintenance context.
- **Pydantic validation** — input schema rejects malformed requests before they reach the model, preventing silent errors.
- **Layer caching** — requirements copied before code in Dockerfile so dependency install is cached and only rebuilt when requirements change.
- **Health endpoint** — standard production pattern for monitoring systems.