# Interpretable-Anomaly-Detection-for-Government-Audits-Script-
ML-based fraud detection for government audit data using Isolation Forest and statistical analysis, reducing audit scope to the top 4–5% high-risk cases.
# Fraud Detection in Government Financial Audit Data

This repository contains the work completed during my **AI/ML & Data Analyst Internship** at the **Office of the Principal Accountant General (A&E), Jaipur, Rajasthan** from **May 2025 to July 2025**.

The project focuses on building an **interpretable, data-driven fraud detection system** for large-scale government financial transaction data, designed to assist auditors in identifying high-risk cases efficiently.

---

## 📌 Project Overview

Government financial audits involve reviewing massive volumes of transaction data, where traditional manual scanning is time-consuming and inefficient.  
This project addresses that challenge through a **data-driven approach** that combines:

- Machine learning–based anomaly detection  
- Statistical validation techniques  
- Domain-specific financial rules  

The system narrows audit attention to a **small, high-risk subset (approximately 4–5%)** of transactions while preserving transparency and explainability for human auditors.

---

## 🧠 Methodology

### 1. Data Processing & Feature Engineering

A **data-driven and domain-aware feature engineering process** was followed, resulting in **40+ engineered features**, including:

- Election proximity indicators  
- GST and IT deduction flags  
- Rolling transaction window statistics  
- Temporal and cyclic behavior patterns  
- Statistical deviation and interaction features  

These features were designed to reflect real audit risk signals observed in government financial data.

---

### 2. Anomaly Detection & Risk Identification

- Applied **Isolation Forest** for unsupervised anomaly detection  
- Integrated **Z-score–based statistical logic** to validate deviations  
- Implemented **duplicate transaction checks**  
- Used **rolling window mechanisms** to capture short-term irregularities  

This hybrid, data-driven pipeline allows each transaction to be evaluated using both model-based and rule-based signals.

---

### 3. Visualization & Audit Support

- Visualized anomalous spending behavior using **Matplotlib** and **Seaborn**  
- Generated plots to highlight:
  - Sudden expenditure spikes  
  - Abnormal department-wise trends  
  - Temporal clustering of anomalies  

These visualizations support **data-driven audit decision-making** and enhance interpretability for auditors.

---

## ⚙️ Key Outcomes

- Replaced manual full-dataset scanning with ML-driven risk prioritization  
- Reduced audit scope to the **top ~4–5% high-risk transactions**  
- Improved audit efficiency while maintaining transparency  
- Designed the system to align with real government audit workflows and constraints  

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
- This repository focuses on **methodology, feature engineering, and system design**

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

📧 **Email:** vardhanworks14@gmail.com  

For discussions related to:
- Data-driven fraud detection  
- Interpretable anomaly detection  
- Audit analytics and public finance systems
