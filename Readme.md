## Digital Payments Transaction Risk Analytics

## 📌 Project Overview
This repository contains the end-to-end data analytics pipeline for detecting fraudulent activities in digital payment systems. The project utilizes a large-scale simulated credit card dataset to apply statistical methods, predictive modeling, and time-series analysis to identify high-risk transactions.

## 📊 Dataset Features
The analysis is performed on a structured dataset containing the following key attributes:
*   **trans_date_trans_time:** Timestamp of the digital transaction.
*   **amt:** The quantitative transaction amount (Primary feature for risk profiling).
*   **category:** The type of merchant/expense (e.g., grocery, travel, entertainment).
*   **gender:** Demographic information of the cardholder.
*   **city_pop:** Population of the transaction location.
*   **lat / long:** Geographic coordinates for spatial risk analysis.
*   **is_fraud:** The binary target variable (0 for Legitimate, 1 for Fraud).

## 🛠️ Project Methodology
The project is divided into six core units as per the academic guidelines:

**Unit I: Data Engineering & Cleaning**
*   Comprehensive data profiling to identify inconsistencies.
*   Handling extreme outliers (Winsorization) and applying log-transformations to address skewness.
*   Feature engineering to convert raw timestamps into usable temporal features.

**Unit II: Descriptive Analysis**
*   Analysis of Central Tendency: Mean, Median, and Mode.
*   Analysis of Dispersion: Standard Deviation, Variance, and Interquartile Range (IQR).
*   Distribution diagnostics using Skewness and Kurtosis metrics.

**Unit III: Inferential Statistics**
*   Application of Welch's Two-Sample T-Tests to compare fraud vs. legitimate transaction behavior.
*   Chi-Square Tests of Independence to determine associations between merchant category and fraud risk.

**Unit IV: Predictive Modeling**
*   Implementation of Logistic Regression for binary classification.
*   Evaluation via Confusion Matrix, Accuracy, Precision, and Recall.

**Unit V: Multivariate Analysis**
*   Implementation of K-Means Clustering or PCA to segment risk profiles and geographic hotspots.

**Unit VI: Time-Series Analysis**
*   Decomposition of transaction trends to identify seasonality.
*   Application of Exponential Smoothing or ARIMA models to forecast future risk patterns.

## 📂 Repository Structure
*   `RiskAnalysis.ipynb`: Main Python notebook for data processing and analysis.
*   `Stats_DA_Digital_Risk_Analysis.pdf`: Comprehensive academic report for Units I-III.
*   `README.md`: Project documentation and methodology.

## 🚀 How to Replicate
1.  Clone this repository.
2.  Download the `fraudTrain.csv` dataset from Kaggle (Sparkov Fraud Dataset).
3.  Open the `.ipynb` file in Google Colab or Jupyter Notebook.
4.  Upload the dataset and execute the cells sequentially.

---

## 📈 Summary of Audited Statistical Findings
From the comparison analysis, we observe significant changes in the descriptive statistics after applying the log transformation (`np.log1p`) to the **amt** feature:

*   **Mean and Median:** In the original data, the mean ($70.35) was higher than the median ($47.52), indicating right-skewness. After transformation, they converged (Mean: 3.53, Median: 3.88), suggesting a balanced, near-normal distribution.
*   **Skewness:** The original 'amt' exhibited extreme positive skewness (**42.28**). The log transformation successfully reduced this to **-0.30**, bringing the distribution significantly closer to symmetry.
*   **Kurtosis:** The kurtosis dropped dramatically from **4,545.64** to **-0.53**. This indicates that the transformation effectively mitigated the impact of extreme high-value outliers and heavy tails.

### **Data Analysis Key Findings (Final Audit)**
| Metric | Original 'amt' | Log-Transformed 'amt' |
| :--- | :--- | :--- |
| **Mean** | $70.35 | 3.53 |
| **Median** | $47.52 | 3.88 |
| **Skewness** | 42.28 | -0.30 |
| **Kurtosis** | 4,545.64 | -0.53 |

**Impact:** The transformation successfully addressed the volatility of transaction amounts, making the feature more suitable for statistical models (like Logistic Regression) that assume normality or benefit from less skewed data.

---


