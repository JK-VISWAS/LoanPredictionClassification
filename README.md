# Loan Prediction Classification

A machine learning project that predicts whether a loan application will be approved based on applicant details such as income, education, employment status, credit history, and property area.

## 📌 Problem Statement

Financial institutions need a reliable way to assess loan applications quickly and consistently. This project builds an end-to-end ML pipeline — from data cleaning to model evaluation — to classify whether a loan application should be approved (`Loan_Status`).

## 📊 Dataset

The dataset contains applicant information including:
- Gender, Marital Status, Dependents, Education, Self-Employment status
- Applicant & Co-applicant Income
- Loan Amount & Loan Term
- Credit History, Property Area
- Target: Loan_Status (Approved / Not Approved)

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn (SMOTE)

## 🔍 Workflow

1. **Data Cleaning** — Handled missing values using mode (categorical) and median (numerical) imputation
2. **Exploratory Data Analysis** — Visualized categorical vs target relationships and numerical distributions
3. **Preprocessing** — Label encoding for categorical variables, feature scaling with StandardScaler
4. **Class Imbalance Handling** — Applied SMOTE to balance the approved/not-approved classes
5. **Model Training & Comparison** — Trained and evaluated multiple classification algorithms:
   - Logistic Regression
   - K-Nearest Neighbors
   - Naive Bayes
   - Support Vector Machine (RBF, Linear, Polynomial kernels)
   - Decision Tree
   - Random Forest
   - AdaBoost
6. **Evaluation** — Compared models using accuracy, confusion matrix, and classification report

## 📈 Results

Logistic Regression achieved the strongest baseline performance at **~81% accuracy** on the initial train-test split. After further tuning and balancing with SMOTE, model performance was re-evaluated across all algorithms to identify the best-generalizing model.

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
jupyter notebook loan_prediction_classification.ipynb
```

## 📚 Key Learnings

- Handling missing data with domain-appropriate imputation strategies
- Encoding categorical features for ML models
- Addressing class imbalance using SMOTE
- Comparing multiple classification algorithms systematically
- Importance of careful feature/target separation to avoid data leakage

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV) for top-performing models
- Feature importance analysis
- Deploy as a simple web app for interactive predictions

---
*Built as part of a Data Science learning project.*
