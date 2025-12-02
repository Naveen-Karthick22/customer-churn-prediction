# Customer Churn Prediction

This project predicts whether a customer will **leave a telecom service** (churn) using machine learning.  
It demonstrates the complete ML workflow: **data preprocessing, model training, evaluation, and feature analysis**.

---

## 🔹 Dataset

- **Source:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Description:** Contains customer demographics, account information, and services used.
- **Rows/Columns:** 7043 rows, 21 columns
- **Target:** `Churn` (Yes=1, No=0)

---

## 🔹 Project Overview

**Steps performed in this project:**

1. **Data Exploration & Visualization**
   - Check data types, missing values, and basic statistics
   - Visualize churn distribution

2. **Data Preprocessing**
   - Drop irrelevant columns (`customerID`)
   - Handle missing values (`TotalCharges`)
   - Encode categorical variables (binary & one-hot encoding)
   - Scale numeric features

3. **Train-Test Split**
   - 80% training, 20% testing
   - Stratified split to preserve class distribution

4. **Model Training**
   - Logistic Regression
   - Random Forest Classifier

5. **Model Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix
   - Comparison of models

6. **Feature Importance**
   - Identify most important factors contributing to churn

---

## 🔹 Skills Applied

- **Programming & Data Analysis:** Python, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, Logistic Regression, Random Forest, Model Evaluation
- **Data Preprocessing:** Missing value handling, categorical encoding, feature scaling
- **Visualization:** Bar plots, heatmaps, confusion matrix, feature importance
- **Project Workflow:** End-to-end ML project structure, resume-ready code

---

## 🔹 Files in This Project



