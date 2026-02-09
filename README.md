#Credit Risk Classification: Estimator Comparison on Statlog (German Credit)
## 📌 Project Overview:
This project performs a comprehensive data analysis and machine learning benchmark on the Statlog (German Credit Data) dataset (OpenML ID: 31). The primary objective is to classify individuals as either "good" or "bad" credit risks based on a set of 20 attributes (e.g., checking account status, credit history, purpose, and savings).
This repository covers the full data science lifecycle:
1. Exploratory Data Analysis (EDA): Feature correlation and distribution analysis.
2. Feature Engineering: Identifying key drivers of credit risk.
3. Model Benchmarking: Comparing multiple estimators to find the most robust predictive model.
## 📊 Dataset Description
The dataset consists of 1,000 instances with 21 variables:
1. Target: class (good / bad)
2. Features: A mix of categorical and numerical attributes including checking_status, duration, credit_history, savings_status, and employment.
## 🛠️ Technical Workflow
### 1. Feature Analysis:
I performed a correlation analysis to identify multi-collinearity and used Feature Importance scores to prune the dataset. This ensures the models focus on the most predictive signals, reducing noise and training time.
### 2. Model Selection & Evaluation:
The project compares several popular machine learning algorithms. Given the nature of credit risk (where classes are often imbalanced and the cost of False Negatives is high), I used ROC AUC (Area Under the Receiver Operating Characteristic Curve) as the primary evaluation metric.
The top-performing estimators identified were:
SVM (Support Vector Machine)
Random Forest
AdaBoostGaussian 
Naive Bayes
## 📈 Results
The benchmarking phase revealed that ensemble methods and non-linear classifiers significantly outperform baseline models on this specific feature set.

## 🚀 Getting Started
### Dependencies
1. Python 3.x
2. Pandas / Numpy
3. Scikit-learn
4. Matplotlib / Seaborn
### Installation
```bash
#!/bin/bash
Bashgit clone https://github.com/sauravdas101/OpenML-credit-g.git
cd OpenML-credit-g
```
# 📝 Conclusion
Based on the ROC AUC scores, SVM and Random Forest provided the best trade-off between sensitivity and specificity, making them the most suitable candidates for deployment in a credit risk assessment context.
