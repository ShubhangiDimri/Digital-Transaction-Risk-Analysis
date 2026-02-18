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

Summary of Observed Differences and Impact of Transformation
From the comparison table and plot, we can observe significant changes in the descriptive statistics after applying the log transformation to the 'amt' column:

Mean and Median: The values for both mean and median have been drastically reduced and are much closer to each other in the log-transformed data. For the original data, the mean ( 71.54)wasmuchhigherthanthemedian( 47.86), indicating a right-skewed distribution. After transformation, both are around ~3.5-3.9, which suggests a more balanced distribution.

Skewness: The most notable change is in skewness. The original 'amt' column had a very high positive skewness (29.17), indicating a long tail to the right due to a few very large transaction amounts. After the log transformation, the skewness became negative (-0.29), indicating that the distribution is now slightly skewed to the left, but is significantly closer to zero (symmetrical) compared to the original.

Kurtosis: Similarly, the kurtosis value dropped dramatically from 1878.62 (original) to -0.52 (log-transformed). High kurtosis in the original data implied heavy tails and potential outliers. The negative kurtosis in the transformed data suggests a platykurtic distribution, meaning it has lighter tails than a normal distribution, and is much closer to the kurtosis of a normal distribution (0).

Impact of Transformation: The log transformation has successfully addressed the high positive skewness and kurtosis present in the original 'amt' column. This transformation compresses the range of values, particularly reducing the influence of extreme outliers (large transaction amounts). The resulting distribution is much closer to a normal distribution, which is often a desirable property for many statistical models and machine learning algorithms. This makes the data more suitable for analysis methods that assume normality or require less skewed data.

Final Task
Subtask:
Summarize the differences observed in the descriptive statistics before and after the log transformation, explaining the impact of the transformation on the data distribution.

Summary:
Q&A
The log transformation on the 'amt' column resulted in a significant normalization of the data distribution. The highly right-skewed original distribution, characterized by a large difference between mean and median and extremely high skewness and kurtosis, was transformed into a much more symmetrical distribution. The mean and median became much closer, skewness was drastically reduced and became slightly negative, and kurtosis dropped substantially, indicating a reduction in heavy tails and outliers.

Data Analysis Key Findings
Log Transformation Applied: A new column, amt_log_transformed, was successfully created by applying np.log1p to the original 'amt' column.
Original 'amt' Statistics:
Mean: $71.54
Median: $47.86
Skewness: 29.17 (highly positively skewed)
Kurtosis: 1878.62 (leptokurtic, indicating heavy tails and outliers)
Log-Transformed 'amt' Statistics:
Mean: $3.54
Median: $3.89
Skewness: -0.29 (close to symmetrical, slightly negatively skewed)
Kurtosis: -0.52 (platykurtic, indicating lighter tails than a normal distribution)
Significant Reduction in Skewness: The skewness was reduced from 29.17 to -0.29, indicating a substantial shift towards a more symmetrical distribution.
Dramatic Decrease in Kurtosis: Kurtosis decreased from 1878.62 to -0.52, suggesting that the transformation effectively mitigated the influence of extreme values and heavy tails.
Convergence of Mean and Median: The mean and median values, initially $71.54 and $47.86 respectively, became much closer after transformation (mean: $3.54, median: $3.89), further indicating a more balanced distribution.
Visual Confirmation: A bar plot visually confirmed these changes, clearly illustrating the dramatic shifts in mean, median, skewness, and kurtosis after the log transformation.
Insights or Next Steps
The log transformation successfully addressed the high positive skewness and kurtosis of the 'amt' column, making its distribution significantly closer to a normal distribution. This is beneficial for statistical models that assume normality.
The transformed 'amt' column is now more suitable for further analysis and modeling tasks, such as regression or classification, where highly skewed features can negatively impact model performance.
