# Module 4 — Agentic AI: ReAct Maintenance Assistant

## Overview
An LLM agent using the ReAct pattern that dynamically routes user questions
to the appropriate tool — predicting machine failure from sensor readings
or retrieving information from maintenance documentation.

## Architecture
User question
↓
Gemini (ReAct loop)
↓
Reason → which tool does this need?
↓
┌─────────────────────────────────┐
│  predict_failure  │  search_docs │
│  (FastAPI/XGBoost)│  (RAG/FAISS) │
└─────────────────────────────────┘
↓
Observe → is this enough to answer?
↓
Final answer with reasoning trace

## Tool routing logic

| Question type | Tool called |
|---|---|
| Specific sensor readings provided | predict_failure |
| General maintenance question | search_docs |
| Both sensor data and general question | Both tools in sequence |
| Out of scope | Neither — graceful refusal |

## Key design decisions
- **Manual ReAct loop** over LangChain agent abstraction — full
  visibility into reasoning trace, easier to debug and explain
- **Tool descriptions as routing logic** — the LLM routes based
  on natural language descriptions, not hard-coded rules
- **Failure mode documentation** — agent tested on ambiguous,
  partial, and contradictory inputs to understand reliability limits

## Failure modes identified
- Partial sensor data: agent [behavior observed]
- Contradictory framing: agent [behavior observed]
- Out of scope: agent [behavior observed]

## Files
- `01_agent.ipynb` — full agent implementation with failure analysis

## Run
Requires FastAPI server running on port 8000 and FAISS index built:
```bash
# Terminal 1
cd 3_api && uvicorn main:app --port 8000

# Terminal 2
cd 4_agent && jupyter lab
```