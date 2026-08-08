# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 10:33:00 2026

@author: Shahbaz
"""

import streamlit as st
import pandas as pd
import numpy as np

# Model Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Processing & Split Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Evaluation & Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Heart Disease Model Evaluator", layout="wide")

st.title("🏥 Machine Learning Model Evaluation Dashboard")
st.write("This app runs entirely on Python code scripts (`.py`). It trains models dynamically on the training data and evaluates them on your uploaded test data.")

@st.cache_data
def get_training_data():
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
        'ca': np.random.choice([0, 1, 2, 3, 4], size=n_samples),
        'thal': np.random.choice([0, 1, 2, 3], size=n_samples)
    })
    df['target'] = np.random.choice([0, 1], size=n_samples, p=[0.46, 0.54])
    
    X = df.drop(columns='target')
    y = df['target']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, y_train

X_train, y_train = get_training_data()


st.sidebar.header("📁 Step 1: Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload your test dataset (CSV format)", type=["csv"])

st.sidebar.header("🤖 Step 2: Choose Model")
model_options = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}
selected_model_name = st.sidebar.selectbox("Select a Classification Model to Evaluate", list(model_options.keys()))

# Main Window Logic execution
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    if 'target' not in test_df.columns:
        st.error("Error: The uploaded CSV test file must contain a 'target' column for benchmarking.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        st.subheader("📋 Uploaded Test Set Snippet")
        st.dataframe(test_df.head(5))
        
        try:
            numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])
            
            # Select the chosen model from our pure code dictionary
            raw_model = model_options[selected_model_name]
            
            # Combine preprocessing and model into a single pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', raw_model)])
            
            # Train the model instantly in memory
            with st.spinner(f"Training {selected_model_name} on baseline data..."):
                pipeline.fit(X_train, y_train)
            
            # Perform evaluation predictions using the freshly trained model
            y_pred = pipeline.predict(X_test)
            
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred
            
            st.success(f"Successfully trained and evaluated data using: **{selected_model_name}**")
            
            # --- FEATURE C: Display of Evaluation Metrics ---
            st.subheader("📊 Model Performance Metrics")
            
            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5  # Fallback for single-class test slices
                
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric(label="Accuracy", value=f"{acc:.4f}")
            col2.metric(label="AUC Score", value=f"{auc:.4f}")
            col3.metric(label="Precision", value=f"{prec:.4f}")
            col4.metric(label="Recall", value=f"{rec:.4f}")
            col5.metric(label="F1 Score", value=f"{f1:.4f}")
            col6.metric(label="MCC Score", value=f"{mcc:.4f}")
            
            # --- FEATURE D: Confusion Matrix and Classification Report ---
            st.markdown("---")
            left_col, right_col = st.columns(2)
            
            with left_col:
                st.subheader("🧱 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm, 
                    index=["Actual Healthy (0)", "Actual Disease (1)"], 
                    columns=["Predicted Healthy (0)", "Predicted Disease (1)"]
                )
                st.dataframe(cm_df, use_container_width=True)
            
            with right_col:
                st.subheader("📄 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Processing Error: An unexpected execution break occurred - {str(e)}")

else:
    st.info("💡 Waiting for validation dataset upload. Please upload your test sample CSV file in the sidebar to run the code-only training loop.")
