# Module 1 — Tabular ML: Predictive Maintenance

## Problem
Predict machine failure before it happens using sensor readings from
industrial equipment. An unexpected breakdown costs significantly more
than a preventive inspection — so the model prioritizes catching real
failures (recall) over avoiding false alarms (precision).

## Dataset
**AI4I 2020 Predictive Maintenance** — UCI Machine Learning Repository
- 10,000 production cycles, 6 input features
- 3.4% failure rate — heavily imbalanced
- 5 failure modes: HDF, PWF, OSF, TWF, RNF
- Synthetic dataset generated to simulate real manufacturing conditions

## Approach

**Feature engineering**
Created `temp_diff = Process temperature - Air temperature` to explicitly
encode the heat dissipation failure trigger condition identified in EDA.

**Imbalance handling**
Used `scale_pos_weight = 28` to tell XGBoost that failure samples are
28× more important than non-failure samples. Evaluated with recall and
F1 rather than accuracy.

**Explainability**
Used SHAP to explain predictions at both global and individual level.
Key finding: globally torque and tool wear dominate, but for HDF-type
failures temp_diff and rotational speed become the primary drivers —
a distinction invisible to global importance metrics.

**Threshold tuning**
Optimized the classification threshold beyond the 0.5 default using
the precision-recall curve, selecting the value that maximizes F1
while maintaining acceptable recall for the maintenance context.

## Results

| Metric | Default model | Tuned model |
|--------|--------------|-------------|
| Recall (failure) | 84% | X% |
| Precision (failure) | 59% | X% |
| F1 (failure) | 0.69 | X |

## Key learnings
- Data leakage: failure sub-columns must be excluded as features
- Accuracy is misleading on imbalanced data — 97% accuracy with
  16% of failures missed is not a good model
- SHAP reveals per-prediction reasoning that global metrics hide
- Threshold is a business decision, not a model decision

## Files
- `01_eda.ipynb` — exploratory data analysis
- `02_model.ipynb` — XGBoost training, evaluation, SHAP, threshold tuning
- `03_tuning.ipynb` — GridSearchCV hyperparameter optimization