# Module 2 — RAG Pipeline: Maintenance Document Q&A

## Problem
LLMs have no knowledge of specific equipment documentation and hallucinate
when asked about proprietary or specialized content. RAG grounds answers
in actual source documents, making them auditable and reliable.

## Pipeline
Documents → Load → Chunk (500 chars, 50 overlap) → Embed (all-MiniLM-L6-v2) → FAISS index → Query → Retrieve top-3 chunks → Gemini Pro → Grounded answer with source

## Key decisions

**Chunking:** 500 character chunks with 50 character overlap using
RecursiveCharacterTextSplitter. Splits at paragraph boundaries first
to preserve context. Tested 300 and 800 — 500 gave best retrieval
precision with sufficient context.

**Embedding model:** sentence-transformers/all-MiniLM-L6-v2 — small,
fast, 384-dimensional embeddings, strong semantic similarity performance
on technical text.

**Prompt design:** Explicit instruction to use ONLY provided context
with a safe fallback for unknown questions. Prevents the LLM from
answering from training memory when retrieved context is weak.

**Temperature:** 0.1 — factual, consistent outputs appropriate for
technical Q&A. High temperature would introduce variability not
suitable for maintenance decisions.

## Results
- 5/5 test questions answered correctly
- Correctly refused to answer out-of-scope questions
- Source attribution on every answer

## Files
- `01_rag_pipeline.ipynb` — full pipeline from loading to generation
- `docs/` — source documents (not committed — download via notebook)