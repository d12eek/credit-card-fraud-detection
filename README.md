# 💳 Credit Card Fraud Detection System

**End-to-End ML | Anomaly Detection | Real-Time Simulation | Production-Ready Pipeline**

---

## 🚀 Overview

An industry-inspired **fraud detection system** built on real-world transaction data, designed to handle **extreme class imbalance (0.173% fraud rate)** using unsupervised anomaly detection techniques.

Unlike typical classification projects, this system focuses on:

* Risk scoring instead of binary prediction
* High recall for fraud detection
* Real-world deployment readiness

---

## 📊 Dataset

* **Total Transactions:** 284,807
* **Fraud Cases:** 492
* **Fraud Rate:** 0.173%
* Highly imbalanced dataset simulating real banking scenarios

---

## 🧠 Models Implemented

* **Isolation Forest** → ROC-AUC: **0.9522**
* **Local Outlier Factor (LOF)** → ROC-AUC: **0.9487**
* **One-Class SVM** → ROC-AUC: **0.8056**

Each model captures different fraud patterns:

* Global anomalies (Isolation Forest)
* Local density deviations (LOF)
* Boundary-based detection (OCSVM)

---

## ⚙️ Feature Engineering

Engineered **6 domain-driven features**:

* Log-transformed transaction amount
* Hour-of-day behavioral signal
* PCA feature magnitude (V-space)
* Fraud cluster mean/std features
* Amount-to-feature ratio

Also tuned contamination parameter to match **true fraud rate (0.001728)** for improved precision.

---

## 🔗 Ensemble Strategy

Built a **weighted ensemble scoring system**:

* Isolation Forest → 50%
* LOF → 30%
* One-Class SVM → 20%

Techniques used:

* Sigmoid normalization
* Threshold optimization

📈 **Performance:**

* Detected **90/148 frauds** in test set
* **PR-AUC: 0.1975** (strong for extreme imbalance)

---

## 🏗️ System Architecture

Modular production-ready pipeline:

```
data_loader → preprocessing → feature_engineering → train → evaluate → predict
```

Key components:

* StandardScaler for normalization
* Stratified train-test split (70/30)
* Joblib for model persistence
* Reproducible pipeline design

---

## 📊 Interactive Dashboard (Streamlit)

Deployed a **real-time fraud monitoring system** with:

* 🔍 Single transaction fraud scoring
* 📂 Batch CSV upload
* 📈 Alert distribution visualization
* 📊 Model comparison (ROC-AUC, PR-AUC, F2-score)
* ⚡ Real-time inference simulation

---

## 🔄 Project Evolution

### 🔹 Version 1 (Colab Prototype)

* Built in Google Colab
* Focused on experimentation
* Implemented anomaly detection models
* Interactive dashboard using Gradio

### 🔹 Version 2 (Production-Ready System)

* Refactored into modular Python pipeline
* Advanced feature engineering + ensemble modeling
* Streamlit deployment
* Clean dependency and project structure

---

## ⚠️ Real-World Challenges Addressed

* Extreme class imbalance (<1% fraud)
* Lack of labeled fraud data
* Concept drift in fraud patterns
* Trade-off between recall and false positives

---

## ⚙️ Installation & Usage

```bash
git clone https://github.com/d12eek/credit-card-fraud-detection.git
cd credit-card-fraud-detection

pip install -r requirements.txt
python main.py
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn, Plotly
* Streamlit
* Joblib

---

## 🎯 Key Takeaways

* Designed system for **real-world fraud detection constraints**
* Focused on **recall and risk-based decisioning**
* Built **scalable and modular ML pipeline**
* Simulated **production fraud monitoring environment**

---

## 📌 Future Improvements

* Deploy as REST API (FastAPI)
* Add real-time streaming (Kafka / Spark)
* Online learning for concept drift
* Model explainability (SHAP / LIME)

---


