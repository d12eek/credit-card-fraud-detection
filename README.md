# 💳 Credit Card Fraud Detection using Anomaly Detection  
### End-to-End | Banking-Grade | Real-Time Simulation | Interactive Dashboard

An industry-inspired, end-to-end fraud detection system built using **unsupervised anomaly detection**, designed to reflect how **real banking and digital payment systems** detect fraudulent transactions under extreme class imbalance.

> ⚠️ This is **not a Kaggle-style classification project**.  
> The system prioritizes **risk scoring, recall, interpretability, and operational realism**, exactly as used in production fraud pipelines.

---

## 🏦 Industry Context

- **Domain:** Finance / Banking / Digital Payments  
- **Fraud Rate:** < 1% (extreme imbalance)  
- **Key Challenge:** Fraud patterns are rare, evolving, and often unlabeled  
- **Business Priority:**  
  - Minimize missed fraud (high recall)  
  - Control false positives (customer experience)  
  - Enable risk-based decisioning (not binary labels)

---

## 🎯 Problem Statement

Traditional supervised classifiers struggle in fraud detection due to:
- Extreme class imbalance
- Delayed or missing fraud labels
- Concept drift in fraud patterns

**This project addresses these challenges by:**
- Learning *normal transaction behavior*
- Flagging deviations using anomaly detection
- Converting anomaly scores into **risk-based alerts**
- Simulating **real-time fraud decisioning**

---

## 🧠 Approach & Methodology

### 🔹 Modeling Strategy
- **Normal-only training** (fraud labels used only for evaluation)
- **Unsupervised anomaly detection**
- **Threshold-based alerting** instead of hard classification

### 🔹 Models Implemented
- **Isolation Forest** – Primary, scalable global detector  
- **One-Class SVM** – High-recall safety model  
- **Local Outlier Factor (LOF)** – Local density-based anomaly detection  

Each model captures **different fraud signals**, mirroring real-world bank deployments.

---

## ⚙️ Key Features

- Extreme class imbalance handling (fraud < 1%)
- Robust evaluation beyond accuracy:
  - Precision, Recall, F1-score
  - ROC-AUC
  - Precision–Recall AUC
- Business-driven threshold tuning
- **Real-time transaction simulation**
- Tiered fraud alerts:
  - **LOW** → Allow
  - **MEDIUM** → Step-up authentication (OTP)
  - **HIGH** → Block & review
- **Interactive Gradio dashboard** (Colab-compatible)

---

## 📊 Interactive Fraud Monitoring Dashboard

The dashboard enables:
- 🔁 Model selection (Isolation Forest / OCSVM / LOF)
- 🎚️ Threshold tuning (percentile-based)
- 📈 Live fraud recall vs alert volume analysis
- 📋 Alert-level fraud breakdown
- 📉 Risk score distribution visualization
- 💡 What-if analysis for business decisions

This simulates how **fraud operations teams** monitor and tune live systems.

---

## 🧪 Real-Time Fraud Simulation

Transactions are processed **sequentially**, mimicking streaming behavior:
- Each transaction receives an anomaly score
- Scores are mapped to risk levels
- Alerts are logged and analyzed
- Missed frauds are explicitly identified for review

This step validates the system’s **operational readiness**, not just offline metrics.
