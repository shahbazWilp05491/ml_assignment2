# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 10:33:00 2026

@author: Shahbaz
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Evaluation & Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# App Configuration Settings
st.set_page_config(page_title="Heart Disease Model Evaluator", layout="wide")

st.title("🏥 Machine Learning Model Evaluation Dashboard")
st.write("Upload your validation/test CSV data file below to evaluate the performance metrics of your pre-trained models.")

# --- FEATURE A: Dataset Upload Option (CSV) ---
st.sidebar.header("📁 Step 1: Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload your test dataset (CSV format)", type=["csv"])

# --- FEATURE B: Model Selection Dropdown (Renamed folder pathway to 'model') ---
st.sidebar.header("🤖 Step 2: Choose Model")
model_options = {
    "Logistic Regression": "model/logistic_regression_model.joblib",
    "Decision Tree": "model/decision_tree_model.joblib",
    "K-Nearest Neighbors": "model/k-nearest_neighbors_model.joblib",
    "Gaussian Naive Bayes": "model/gaussian_naive_bayes_model.joblib",
    "Random Forest": "model/random_forest_model.joblib"
}
selected_model_name = st.sidebar.selectbox("Select a Classification Model to Evaluate", list(model_options.keys()))

# Main Window Logic execution
if uploaded_file is not None:
    # Read the uploaded test file data
    test_df = pd.read_csv(uploaded_file)
    
    # Check if target class is missing from verification set
    if 'target' not in test_df.columns:
        st.error("Error: The uploaded CSV test file must contain a 'target' column for benchmarking.")
    else:
        # Separate features and targets
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Display short snippet of the uploaded matrix data
        st.subheader("📋 Uploaded Test Set Snippet")
        st.dataframe(test_df.head(5))
        
        model_path = model_options[selected_model_name]
        
        # Check if the serialized binaries exist inside the working tree path
        if not os.path.exists(model_path):
            st.error(f"Missing Binary Error: '{model_path}' was not found. Please upload your model files into the 'model/' directory on GitHub.")
        else:
            try:
                # Load the full execution pipeline (preprocessor + model weights)
                pipeline = joblib.load(model_path)
                
                # Perform model evaluation predictions
                y_pred = pipeline.predict(X_test)
                
                # Capture target probabilities safely for ROC AUC score tracking
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred # Fallback condition for non-probabilistic classifiers
                
                st.success(f"Successfully evaluated data using: **{selected_model_name}**")
                
                # --- FEATURE C: Display of Evaluation Metrics ---
                st.subheader("📊 Model Performance Metrics")
                
                # Calculate evaluation values
                acc = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_prob)
                except ValueError:
                    auc = 0.5  # Edge case tracking fallback if only one target class exists in small test slices
                    
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                # Display structural metric cards side-by-side
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric(label="Accuracy", value=f"{acc:.4f}")
                col2.metric(label="AUC Score", value=f"{auc:.4f}")
                col3.metric(label="Precision", value=f"{prec:.4f}")
                col4.metric(label="Recall", value=f"{rec:.4f}")
                col5.metric(label="F1 Score", value=f"{f1:.4f}")
                col6.metric(label="MCC Score", value=f"{mcc:.4f}")
                
                # --- FEATURE D: Confusion Matrix or Classification Report ---
                st.markdown("---")
                left_col, right_col = st.columns(2)
                
                with left_col:
                    st.subheader("🧱 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Convert raw confusion matrix arrays into structural pandas tables for sleek visualization
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
    st.info("💡 Waiting for validation dataset upload. Please drag and drop a test sample CSV file in the sidebar to view evaluation metrics.")
