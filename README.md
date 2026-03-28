# 🌍 Air Quality Index (AQI) Predictor: An MLOps & Experiment Tracking Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet)
![Optuna](https://img.shields.io/badge/Optuna-Optimization-red)

##  Project Overview
Building a machine learning model is easy; tracking, tuning, and proving its effectiveness is the real engineering challenge. This project is an end-to-end Machine Learning pipeline designed to predict the Air Quality Index (AQI) based on environmental and human factors. 

Instead of a standard `model.fit()` script, this project focuses on **Production-Ready MLOps Practices**. It features automated hyperparameter tuning, robust cross-validation, mathematical explainability, and comprehensive experiment tracking.

## 🚀 Key Features
* **Controlled Synthetic Data:** Generates a custom dataset with built-in mathematical rules and $\pm 15$ points of random noise to simulate real-world sensor unreliability and explore the concept of **Irreducible Error** ($y = f(x) + \epsilon$).
* **Experiment Tracking (MLflow):** Replaces manual spreadsheets by systematically logging baseline metrics, parameters, and model artifacts (like `.pkl` files and plot images) in a centralized dashboard.
* **Tuning at Scale (Optuna):** Utilizes Bayesian optimization alongside 5-Fold Stratified Cross-Validation to automatically hunt down the optimal `n_estimators` and `max_depth` for a Random Forest Regressor.
* **Model Explainability:** Opens the "black box" of the Random Forest by extracting and plotting Feature Importances to prove the model learned the actual physical drivers of AQI.

##  Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **MLOps / Tracking:** MLflow
* **Hyperparameter Optimization:** Optuna
* **Data Manipulation & Math:** Pandas, NumPy
* **Data Visualization:** Matplotlib

## 📈 The Pipeline & Results

### 1. The Baseline
A default Random Forest model was trained to establish an anchor point. 
* **Baseline RMSE:** 17.01
* **Baseline R² Score:** 0.71

### 2. The Optuna Sweep
An Optuna study was launched with a 10-trial budget, using 5-Fold CV to ensure fair grading. 
* **Winning Parameters:** `n_estimators: 229`, `max_depth: 8`
* **Optimized RMSE:** 15.96 (Successfully beat the baseline)

### 3. The Irreducible Error Reality
The RMSE dropped from 17.01 to roughly 15.96. Because the synthetic dataset was intentionally injected with 15 points of random Gaussian noise, an RMSE of ~15 is the absolute mathematical limit. This project successfully squeezed out all learnable signals from the noise!

### 4. Explainability 
Feature importance extraction successfully reverse-engineered the hidden rules of the dataset, correctly identifying **Wind Speed** and **Temperature** as the primary drivers of AQI changes, validating the model's logic.

