## Digital Payments Transaction Risk Analytics


## 📌 Project Overview

This repository contains the end-to-end data analytics pipeline for detecting fraudulent activities in digital payment systems. The project utilizes a large-scale simulated credit card dataset to apply statistical methods, predictive modeling, and time-series analysis to identify high-risk transactions.


## 📊 Dataset Features

The analysis is performed on a structured dataset containing the following key attributes:

trans_date_trans_time: Timestamp of the digital transaction.

amt: The quantitative transaction amount (Primary feature for risk profiling).

category: The type of merchant/expense (e.g., grocery, travel, entertainment).

gender: Demographic information of the cardholder.

city_pop: Population of the transaction location.

lat / long: Geographic coordinates for spatial risk analysis.

is_fraud: The binary target variable (0 for Legitimate, 1 for Fraud).

## 🛠️ Project Methodology

The project is divided into six core units as per the academic guidelines:

Unit I: Data Engineering & Cleaning

Comprehensive data profiling to identify inconsistencies.

Handling extreme outliers and applying scaling/transformations to address skewness.

Feature engineering to convert raw timestamps into usable time-period data.

Unit II: Descriptive Analysis

Analysis of Central Tendency: Mean, Median, and Mode.

Analysis of Dispersion: Standard Deviation, Range, and Interquartile Range (IQR).

Distribution diagnostics using Skewness and Kurtosis metrics.

Unit III: Inferential Statistics

Application of Two-Sample T-Tests to compare transaction behaviors.

Chi-Square Tests to determine dependencies between categorical features and fraud risk.

Unit IV: Predictive Modeling

Implementation of Logistic Regression for binary classification.

Evaluation via Confusion Matrix, Accuracy, Precision, and Recall.

Unit V: Multivariate Analysis

Implementation of K-Means Clustering or PCA to segment risk profiles and geographic hotspots.

Unit VI: Time-Series Analysis

Decomposition of transaction trends to identify seasonality.

Application of Exponential Smoothing or ARIMA models to forecast future risk patterns.

## 📂 Repository Structure

RiskAnalysis.ipynb: Main Python notebook for data processing and analysis.

README.md: Project documentation and methodology.

🚀 How to Replicate

Clone this repository.

Download the fraudTrain.csv dataset from Kaggle (Sparkov Fraud Dataset).

Open the .ipynb file in Google Colab or Jupyter Notebook.

Upload the dataset and execute the cells sequentially.
