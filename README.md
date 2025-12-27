# Interpretable-Anomaly-Detection-for-Government-Audits-Script-
ML-based fraud detection for government audit data using Isolation Forest and statistical analysis, reducing audit scope to the top 4–5% high-risk cases.
# Fraud Detection in Government Financial Audit Data

This repository contains the work completed during my **AI/ML & Data Analyst Internship** at the **Office of the Principal Accountant General (A&E), Jaipur, Rajasthan** from **May 2025 to July 2025**.

The project focuses on building an **interpretable, ML-driven fraud detection system** for large-scale government financial transaction data, designed to assist auditors in identifying high-risk cases efficiently.

---

## 📌 Project Overview

Government financial audits involve reviewing massive volumes of transaction data, where traditional manual scanning is time-consuming and inefficient.  
This project addresses that challenge by combining:

- **Machine Learning–based anomaly detection**
- **Statistical validation techniques**
- **Domain-driven financial rules**

The system narrows audit attention to a **small, high-risk subset (≈4–5%)** of transactions while preserving transparency and explainability for human auditors.

---

## 🧠 Methodology

### 1. Data Processing & Feature Engineering
Engineered **40+ domain-specific features**, including:

- **Election proximity indicators**
- **GST & IT deduction flags**
- **Rolling transaction statistics**
- **Temporal and cyclic patterns**
- **Statistical deviation measures**
- **Interaction and aggregation features**

These features were designed in consultation with audit workflows to reflect real-world financial risk signals.

---

### 2. Anomaly Detection
- Applied **Isolation Forest** for unsupervised anomaly detection
- Integrated **Z-score logic** to validate statistical deviations
- Implemented **duplicate transaction checks**
- Used **rolling window mechanisms** to capture short-term irregularities

Each transaction receives a risk indication based on combined model and rule-based signals.

---

### 3. Visualization & Audit Support
- Visualized anomalous spending behavior using **Matplotlib** and **Seaborn**
- Generated plots to highlight:
  - Sudden expenditure spikes
  - Abnormal department-wise trends
  - Temporal clustering of anomalies

These visual insights support **human-in-the-loop audit decision-making**.

---

## ⚙️ Key Outcomes

- Replaced manual full-dataset scanning with targeted ML-driven screening
- Reduced audit scope to **~4–5% high-risk transactions**
- Improved efficiency while retaining interpretability
- Designed the system to align with real government audit constraints

---

## 🧪 Tools & Technologies

- **Python**
- **Pandas, NumPy**
- **Scikit-learn (Isolation Forest)**
- **Matplotlib, Seaborn**
- **Statistical Analysis & Time-Series Techniques**

---

## 🏛️ Internship Details

- **Organization:** Office of the Principal Accountant General (A&E), Jaipur, Rajasthan  
- **Role:** AI/ML & Data Analyst Intern  
- **Duration:** May 2025 – July 2025  
- **Mentor:** *Shri Pravindra Yadav*, Principal Accountant General (A&E), Jaipur  

---

## ⚠️ Data Disclaimer

Due to confidentiality and government data policies:
- Actual financial datasets are **not included**
- Any sample data (if present) is anonymized or synthetically generated
- The focus of this repository is on **methodology, feature engineering, and system design**

---

## 📈 Future Enhancements

- Risk score calibration using auditor feedback
- Explainability layers for anomaly reasoning
- Office-level and department-level anomaly aggregation
- Deployment-ready pipeline with logging and monitoring

---

## 📬 Contact

**Jay Vardhan Sharma**  
B.Tech, Computer Science Engineering  
Birla Institute of Technology, Mesra  

Feel free to reach out for discussions related to:
- Applied ML in public finance
- Interpretable anomaly detection
- Audit analytics systems
