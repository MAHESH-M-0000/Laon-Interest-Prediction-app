# 💰 Loan Interest Rate Prediction using XGBoost & SHAP

🚀 **[Live Streamlit App](https://mahesh-m-0000-laon-interest-prediction-app-app-480gsf.streamlit.app/)**  
📂 **[GitHub Repository](https://github.com/MAHESH-M-0000/Laon-Interest-Prediction-app)**

---


## 📌 Project Overview

This project develops a machine learning system to predict loan interest rates using borrower, loan, and credit-related information available at the time of loan issuance.

The project uses **XGBoost Regression** for prediction and **SHAP (SHapley Additive exPlanations)** for model explainability.

A **Streamlit web application** provides an interactive interface for making loan interest-rate predictions.

---

## 🎯 Objectives

- Predict loan interest rates using machine learning.
- Perform data cleaning and feature engineering.
- Handle numerical and categorical variables.
- Build an XGBoost regression model.
- Evaluate model performance using MAE, RMSE and R².
- Explain model predictions using SHAP.
- Deploy the trained model through a Streamlit application.

---

## 📊 Dataset

The dataset contains loan-level information including:

- Loan amount
- Funded amount
- Loan term
- Sub-grade
- Annual income
- Home ownership
- Verification status
- Loan purpose
- Debt-to-income ratio
- Credit utilization
- Credit history
- Public records
- Credit accounts
- Employment length

The original dataset contains **39,717 loan records and 111 columns**.

The raw dataset is not included in this repository.

---

## 🔧 Technologies Used

### Programming
- Python

### Data Analysis
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- XGBoost

### Explainable AI
- SHAP

### Deployment
- Streamlit

### Model Persistence
- Joblib

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

1. Converting interest rate from percentage strings to numeric values.
2. Converting revolving utilization to numeric form.
3. Handling missing values.
4. Converting employment length into numerical years.
5. Removing constant columns.
6. Removing identifier columns.
7. Removing post-origination variables.
8. Engineering loan issue year and month.
9. Engineering credit history length.
10. One-hot encoding categorical variables.
11. Median imputation for numerical variables.
12. Most-frequent imputation for categorical variables.

---

## 🚫 Target Leakage Prevention

Variables that depend on loan performance after origination were excluded from the final model.

The `installment` variable was also excluded because it is mathematically related to loan amount, interest rate and loan term.

This helps ensure that the model uses information that is available at loan origination rather than information generated after the loan is issued.

---

## 🤖 Machine Learning Model

The final model uses:

**XGBoost Regressor**

Key parameters:

- `n_estimators = 500`
- `learning_rate = 0.05`
- `max_depth = 6`
- `min_child_weight = 3`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `random_state = 42`

---

## 📈 Model Performance

The final XGBoost model achieved:

| Metric | Result |
|---|---:|
| MAE | 0.2407 |
| RMSE | 0.3463 |
| R² | 0.9913 |

The target variable is the loan interest rate expressed as a percentage.

---

## 🔍 Explainable AI with SHAP

SHAP was used to understand how individual features contribute to the model's predictions.

The analysis showed that **sub-grade** is a major predictive feature in this dataset.

Other variables contributing to predictions include:

- Loan term
- Revolving utilization
- Loan issue year
- Loan issue month
- Credit-related variables

SHAP was used for both:

- Global feature importance
- Individual prediction explanations

---

## 🌐 Streamlit Application

The application allows users to enter loan and borrower information and receive a predicted interest rate.

### Application workflow

```text
User Input
    ↓
Data Preprocessing
    ↓
XGBoost Model
    ↓
Predicted Interest Rate
    ↓
SHAP Explanation
