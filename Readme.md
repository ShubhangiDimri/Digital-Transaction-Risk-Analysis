# Digital Payments Transaction Risk Analytics

![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Framework](https://img.shields.io/badge/Analysis-Statsmodels%20|%20Scikit--Learn-green?style=for-the-badge)

## 📌 Project Overview
This repository contains a comprehensive, end-to-end data analytics pipeline for detecting fraudulent activities in digital payment systems. The project follows a modular 6-unit structure to explore data from raw collection to advanced time-series forecasting, utilizing a large-scale simulated credit card transaction dataset.

## 📂 Repository Structure
The project is split into six dedicated notebooks to ensure modularity and academic compliance:

1.  **[Unit 1: Data Engineering](Unit1_Data_Cleaning.ipynb)** - Cleaning, Outlier Capping (IQR), and Log Transformations.
2.  **[Unit 2: Descriptive Analysis](Unit2_Descriptive_Analysis.ipynb)** - Central Tendency, Dispersion, and Data Integrity Profiling.
3.  **[Unit 3: Inferential Statistics](Unit3_Inferential_Testing.ipynb)** - Hypothesis Testing (Welch's t-test, Chi-Square) and Effect Sizes.
4.  **[Unit 4: Predictive Modeling](Unit4_Predictive_Modeling.ipynb)** - Linear, Polynomial, and **Logistic Regression**.
5.  **[Unit 5: Multivariate Analysis](Unit5_Dimensionality_Reduction.ipynb)** - Geographic Engineering and **PCA**.
6.  **[Unit 6: Time-Series Analysis](Unit6_Time_Series_Analysis.ipynb)** - Trend Decomposition and **ARIMA Forecasting**.

---

## 🛠️ Detailed Methodology

### **Unit I: Preprocessing & Cleaning**
*   Handled extreme skewness in transaction amounts using **Log Transformation (`np.log1p`)**.
*   Implemented **IQR-based Winsorization** to cap outliers.
*   Engineered temporal features: `transaction_hour`, `transaction_day`, and `is_weekend`.

### **Unit II: Descriptive Diagnostics**
*   Profiled the distribution using **Skewness and Kurtosis** metrics.
*   Verified data integrity for 1.29M+ records.

### **Unit III: Hypothesis Testing**
*   **Two-Sample t-Test:** Compared mean transaction amounts (Fraud vs. Legit). Found a significant difference (p < 0.05) with a very large **Cohen's d of 1.67**.
*   **Chi-Square Test:** Confirmed a strong association between transaction category and fraud risk (**Cramer's V: 0.41**).

### **Unit IV: Regression & Machine Learning**
*   Developed a **Logistic Regression** classifier for fraud detection.
*   Evaluated performance using **ROC-AUC Curves** and **Confusion Matrices**.

### **Unit V: Dimensionality Reduction**
*   Engineered **Haversine Distance** to track spatial anomalies between cardholders and merchants.
*   Applied **Principal Component Analysis (PCA)** to visualize high-dimensional risk clusters in 2D space.

### **Unit VI: Forecasting**
*   Decomposed the time series into **Trend** and **Seasonality** (weekly patterns).
*   Built an **ARIMA(1,1,1)** model to forecast future transaction volume for the next 30 days.

---

## 📈 Audited Statistical Highlights
| Metric | Result / Impact |
| :--- | :--- |
| **Skewness (Raw vs. Log)** | 42.28 ➡️ **-0.30** (Normalized) |
| **Cohen's d (Effect Size)** | **1.67** (Statistically Massive) |
| **Cramer's V (Category Association)** | **0.41** (Strong Relationship) |
| **Logistic Regression ROC-AUC** | High Discriminative Power |
| **PCA Variance** | First 3 components capture primary data signals |

---

## 🚀 Setup & Execution
1.  Clone the repository.
2.  Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn statsmodels`.
3.  Ensure `fraudTrain.csv` is in the root directory.
4.  Execute the notebooks in order (Unit 1 through Unit 6).

---
**Term Project Submission - Digital Transaction Risk Analysis**
