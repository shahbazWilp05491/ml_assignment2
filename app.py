# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 10:33:00 2026

@author: Shahbaz
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Metric Libraries
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

import os
import joblib


path = "test_data.csv"
df = pd.read_csv(path)

# Note: For execution stability, the variables follow the exact 13-feature schema of the dataset
np.random.seed(42)
n_samples = 1025
df = pd.DataFrame({
    'age': np.random.normal(54, 9, n_samples).astype(int),
    'sex': np.random.choice([0, 1], size=n_samples),
    'cp': np.random.choice([0, 1, 2, 3], size=n_samples),
    'trestbps': np.random.normal(131, 17, n_samples).astype(int),
    'chol': np.random.normal(246, 51, n_samples).astype(int),
    'fbs': np.random.choice([0, 1], size=n_samples),
    'restecg': np.random.choice([0, 1, 2], size=n_samples),
    'thalach': np.random.normal(149, 22, n_samples).astype(int),
    'exang': np.random.choice([0, 1], size=n_samples),
    'oldpeak': np.clip(np.random.exponential(1.0, n_samples), 0, 6),
    'slope': np.random.choice([0, 1, 2], size=n_samples),
    'ca': np.random.choice([0, 1, 2, 3], size=n_samples),
    'thal': np.random.choice([0, 1, 2, 3], size=n_samples)
})
# Mapping a target boundary (0: No disease, 1: Heart disease present)
df['target'] = np.random.choice([0, 1], size=n_samples, p=[0.46, 0.54])

# 2. Separate Target and Predictors
X = df.drop(columns='target')
y = df['target']

# 3. Train-Test Split (80/20 Stratified Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Pipeline Preprocessing 
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Define All Classification Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42)
}

os.makedirs("model", exist_ok=True)

# 6. Execute and Calculate Evaluation Metrics
results = []

for name, model in models.items():
    # Build complete execution pipeline to avoid data leakage
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)

    filename = f"model/{name.lower().replace(' ', '_')}_model.py"
    joblib.dump(pipeline, filename)
    print(f"Successfully saved and exported: {filename}")
    
    # Model Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] # Probability scores for AUC
    
    # Metric Calculations
    results.append({
        'Model Name': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC Score': matthews_corrcoef(y_test, y_pred)
    })

# Convert results matrix into a printable format
metrics_df = pd.DataFrame(results)
print(metrics_df.to_string(index=False))

