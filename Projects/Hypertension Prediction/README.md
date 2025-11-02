# Hypertension Risk Prediction Project

## Project Overview

This project predicts the risk of hypertension using multiple machine learning algorithms and provides an interactive web app for user inputs.

## Contents

- hypertension_data.csv : Dataset with health features & hypertension label
- Hypertension_ML_Notebook.ipynb : Full ML workflow
- app.py : Streamlit web application
- hypertension_rf_model.pkl : Saved Random Forest model

## How to Run

1. Run the Jupyter Notebook to train models and save `hypertension_rf_model.pkl`.
2. Install Streamlit: pip install streamlit scikit-learn pandas numpy
3. Run Streamlit app: streamlit run app.py
4. Input health parameters to get prediction.

## Features

- Multiple ML algorithms: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost
- Model comparison and evaluation
- Feature importance & ROC curve visualization
- Hyperparameter tuning for Random Forest
- Cross-validation for robust evaluation
- Interactive web app for real-time prediction
