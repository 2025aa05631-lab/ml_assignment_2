import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)

# Loading Data downloaded from kagglehub (yasserh/breast-cancer-dataset)
df = pd.read_csv('breast-cancer.csv') 

# Cleaning data
# Removing ID and the empty column 'Unnamed: 32' if it exists
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Data preparation
# Target is 'diagnosis' (M = Malignant, B = Benign)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Binary Encoding: M -> 1, B -> 0
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, 'model/label_encoder.pkl')

# Split (80/20) with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SCALE THE DATA (Crucial for Breast Cancer features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')

# Define the 6 Required Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

# Train and Evaluate
results = []
os.makedirs('model', exist_ok=True)

print(f"Training on {X.shape[1]} features and {X.shape[0]} samples...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of 'Malignant'
    
    metrics = {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }
    results.append(metrics)
    joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')

# 6. Final Table for README
results_df = pd.DataFrame(results)
print("\n--- PERFORMANCE COMPARISON TABLE ---")
print(results_df.to_markdown(index=False))

# Create a sample test file for the Streamlit app
test_sample = pd.DataFrame(X_test, columns=X.columns)
test_sample['diagnosis'] = y_test
test_sample.to_csv('test_breast_cancer.csv', index=False)
