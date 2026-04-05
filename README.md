# Predictive Maintenance AI

End-to-end AI system for industrial equipment failure prediction,
combining tabular ML, retrieval-augmented generation, REST API
deployment, and agentic AI.

Built as part of preparation for the Mastère Spécialisé Data & IA
at Telecom Paris.

---

## The problem

Unexpected machine failures in industrial environments cause
significant production losses. A broken machine costs far more
than a preventive inspection — but inspecting every machine
constantly is not feasible either.

This project builds an AI system that:
- **Predicts** which machines are at risk before they fail
- **Explains** why each prediction was made
- **Answers** maintenance questions in natural language
- **Routes** user requests to the right tool automatically

---

## Architecture
┌─────────────────────────────────────────────────────────┐
│                    User / Application                    │
└─────────────────────┬───────────────────────────────────┘
│
┌───────────▼───────────┐
│    ReAct Agent        │  ← Module 4
│  (Gemini + LangChain) │
└──────┬────────┬───────┘
│        │
┌────────────▼──┐  ┌──▼──────────────┐
│  predict_     │  │   search_docs   │
│  failure      │  │   (RAG Pipeline)│  ← Module 2
│  (FastAPI)    │  │   LangChain +   │
└──────┬────────┘  │   FAISS +       │
│           │   Gemini API    │
┌──────▼────────┐  └────────┬────────┘
│  XGBoost      │           │
│  Classifier   │  ┌────────▼────────┐
│  + SHAP       │  │  Maintenance    │
└───────────────┘  │  Documents      │
↑             └─────────────────┘
← Module 1
← Module 3 (Docker)

---

## Modules

### Module 1 — Tabular ML
`1_tabular_ml/`

XGBoost binary classifier predicting machine failure from sensor
readings. Full pipeline from EDA to deployment-ready model.

**Key results:**
- 84% recall on a 3.4% imbalanced target
- SHAP explanations revealing per-prediction failure drivers
- Threshold optimized at 0.3 via precision-recall curve
- Hyperparameter tuning via GridSearchCV with stratified 5-fold CV

**Technical highlights:**
- Feature engineering: `temp_diff` encoding the HDF physical trigger
- Imbalance handling: `scale_pos_weight = 28`
- Data leakage prevention: failure sub-columns excluded from features
- SHAP finding: globally torque and tool wear dominate, but HDF
  failures are driven by temp_diff and rotational speed — invisible
  to global importance metrics

---

### Module 2 — RAG Pipeline
`2_rag/`

Natural language Q&A over maintenance documentation using
retrieval-augmented generation.

**Pipeline:**
Documents → Chunk (500 chars, 50 overlap) → Embed
(all-MiniLM-L6-v2, 384-dim) → FAISS index → Retrieve top-3
→ Gemini API → Grounded answer with source attribution

**Key decisions:**
- Chunking at paragraph boundaries to preserve context
- Explicit anti-hallucination prompt: answers only from retrieved context
- Temperature 0.1 for factual, consistent outputs
- Verified embedding quality: semantically similar sentences
  score >0.85 cosine similarity vs <0.2 for unrelated sentences

---

### Module 3 — REST API + Docker
`3_api/`

XGBoost model deployed as a containerized REST API.
```bash
docker build -t predictive-maintenance-api .
docker run -p 8000:8000 predictive-maintenance-api
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/health` | GET | Health check |
| `/predict` | POST | Failure prediction |

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature":298.1,
       "process_temperature":308.6,
       "rotational_speed":1363.0,
       "torque":68.0,"tool_wear":220.0}'
```

**Response:**
```json
{
  "failure_probability": 0.94,
  "prediction": "FAILURE",
  "risk_level": "HIGH"
}
```

---

### Module 4 — Agentic AI
`4_agent/`

ReAct agent routing user requests to the appropriate tool based
on question content.

**Routing logic:**

| Input | Tool called |
|-------|-------------|
| Sensor readings provided | predict_failure → FastAPI |
| General maintenance question | search_docs → RAG |
| Both | Both tools in sequence |
| Out of scope | Graceful refusal |

**Why manual ReAct over LangChain agent wrappers:**
Full visibility into the reasoning trace at each step — essential
for debugging routing failures and explaining decisions.

---

## Dataset

**AI4I 2020 Predictive Maintenance** — UCI Machine Learning Repository
- 10,000 production cycles, 6 sensor features
- 3.4% failure rate across 5 distinct failure modes
- Synthetic dataset simulating real manufacturing conditions

---

## Tech stack

| Layer | Technologies |
|-------|-------------|
| ML | Python, XGBoost, scikit-learn, SHAP |
| RAG | LangChain, FAISS, sentence-transformers, Gemini API |
| Deployment | FastAPI, Docker, uvicorn |
| Agent | LangChain, Gemini API, ReAct pattern |
| Data | pandas, numpy, matplotlib, seaborn |

---

## Run the full system

**1 — Clone and install:**
```bash
git clone https://github.com/leithmaj/predictive-maintenance-ai
cd predictive-maintenance-ai
pip install -r 3_api/requirements.txt
pip install langchain langchain-community langchain-google-genai \
            faiss-cpu sentence-transformers
```

**2 — Set up API key:**
```bash
echo "GOOGLE_API_KEY=your_key_here" > 2_rag/.env
```

**3 — Start the API:**
```bash
cd 3_api && uvicorn main:app --port 8000
```

**4 — Run the agent:**
Open `4_agent/01_agent.ipynb` and run all cells.

---

## Project structure
predictive-maintenance-ai/
├── 1_tabular_ml/
│   ├── 01_eda.ipynb          # EDA and data understanding
│   ├── 02_model.ipynb        # XGBoost, SHAP, threshold tuning
│   └── 03_tuning.ipynb       # GridSearchCV hyperparameter optimization
├── 2_rag/
│   └── 01_rag_pipeline.ipynb # Document loading, chunking, retrieval, generation
├── 3_api/
│   ├── main.py               # FastAPI application
│   ├── Dockerfile            # Container definition
│   └── requirements.txt      # Pinned dependencies
└── 4_agent/
└── 01_agent.ipynb        # ReAct agent with failure mode analysis