# Predictive Maintenance AI

An end-to-end Artificial Intelligence project demonstrating the full lifecycle of a production-ready AI system — from tabular machine learning to retrieval-augmented generation (RAG), API deployment, and agent-based orchestration.

This repository showcases modular, scalable, and deployment-oriented AI development practices applied to predictive maintenance in industrial environments.

---

## 📌 Project Overview

The objective is to build an intelligent predictive maintenance system capable of:

- Detecting potential equipment failures from sensor data  
- Providing explainable model insights  
- Leveraging documentation through Retrieval-Augmented Generation (RAG)  
- Exposing models via a production-ready API  
- Enabling agent-based reasoning with tool usage  

The project emphasizes clean architecture, reproducibility, and real-world deployment considerations.

---

## 🧩 Project Structure

### `1_tabular_ml`
Machine learning pipeline for structured sensor data:
- XGBoost model training  
- Feature engineering  
- Model evaluation  
- SHAP-based explainability  

### `2_rag`
Retrieval-Augmented Generation pipeline:
- Document preprocessing  
- Vector indexing  
- Semantic retrieval  
- Context-aware response generation  

### `3_api`
Production-oriented deployment:
- FastAPI service  
- Docker containerization  
- Model serving endpoints  

### `4_agent`
LLM-powered agent system:
- Tool integration  
- Autonomous reasoning workflows  
- Modular agent architecture  

---

## 🏭 Application Domain

**Predictive maintenance for industrial equipment**  
Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)

The system predicts machine failures using structured operational data and augments decision-making through intelligent document retrieval and agentic reasoning.

---

## 🚀 Key Technologies

- Python  
- XGBoost  
- SHAP  
- FastAPI  
- Docker  
- Vector databases  
- Large Language Models (LLMs)  

---

## 🎯 Goals

- Build a modular AI system spanning multiple paradigms  
- Apply explainable machine learning techniques  
- Integrate RAG into an industrial use case  
- Deploy models in a production-like environment  
- Explore agent-based AI architectures  

