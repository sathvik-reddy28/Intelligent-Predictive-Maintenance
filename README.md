# Intelligent Predictive Maintenance

## Overview

This project explores predictive maintenance using the AI4I 2020 Predictive Maintenance dataset. It combines exploratory data analysis, supervised machine learning, model explainability, failure-mode analysis, and unsupervised anomaly detection.

## Project Structure

- `data/raw/` — Raw AI4I dataset
- `notebooks/data_understanding.ipynb` — Data exploration and quality checks
- `notebooks/predictive_maintenance.ipynb` — Model training, evaluation, SHAP analysis, failure-mode analysis, and anomaly detection
- `results/` — Saved SHAP visualizations
- `src/` — Source-code directory
- `requirements.txt` — Python dependencies

## Methods

- Exploratory data analysis
- Logistic Regression, Decision Tree, Random Forest, and XGBoost baselines
- Hyperparameter tuning and threshold analysis
- SHAP-based model explainability
- Failure-mode frequency and overlap analysis
- Isolation Forest anomaly detection

## Model Evaluation

The selected XGBoost model was evaluated on a held-out test set of 2,000 observations using a classification threshold of 0.20.

| Metric | Test Result |
|---|---:|
| Precision | 0.639 |
| Recall | 0.779 |
| F1-score | 0.702 |
| ROC-AUC | 0.979 |
| PR-AUC | 0.843 |

The test-set confusion matrix contained 1,902 true negatives, 30 false positives, 15 false negatives, and 53 true positives.

The threshold of 0.20 was selected using out-of-fold predictions on the training set. The held-out test set was evaluated once.

## Anomaly Detection

Isolation Forest flagged 2,206 of 10,000 observations as anomalous.

| Metric against recorded failure labels | Result |
|---|---:|
| Precision | 0.105 |
| Recall | 0.681 |
| F1-score | 0.182 |

These metrics measure agreement between anomaly flags and recorded failure labels. Anomalies represent unusual operating patterns and are not necessarily machine failures.

## Notes

The failure-mode indicators are excluded from predictive-model inputs to prevent target leakage. SHAP values describe model behaviour and do not establish causation. Isolation Forest anomalies indicate unusual operating patterns, not confirmed machine failures.

## Limitations

The dataset is synthetic. Model performance and maintenance recommendations should be validated on real industrial data before operational use.