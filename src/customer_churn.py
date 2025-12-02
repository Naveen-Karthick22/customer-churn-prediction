import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Binary columns
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1, 'No':0})

    # One-Hot Encoding for other categorical columns
    multi_cols = [col for col in df.columns if df[col].dtype == "object" and col not in binary_cols]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    return df

def split_and_scale(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return log_model, rf_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm

def save_model(model, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

# Example Usage
if __name__ == "__main__":
    df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_scale(df)
    log_model, rf_model = train_models(X_train, y_train)

    acc_rf, report_rf, cm_rf = evaluate_model(rf_model, X_test, y_test)
    print("Random Forest Accuracy:", acc_rf)
    print(report_rf)
    print("Confusion Matrix:\n", cm_rf)

    save_model(rf_model, "../model/customer_churn_model.pkl")
