# ML_Kit
Reusable and demo kit for ml diagnostics

Base CLI currently supports baseline scikit-learn models for tabular supervised learning:

Regression: Ridge Regression (default), Random Forest Regressor

Classification: Logistic Regression (default), Random Forest Classifier

All models are trained and saved as full scikit-learn pipelines, including preprocessing (imputation, scaling, and one-hot encoding), to ensure reproducible inference.

CLI has capability to add sklearn-compatible models with framework for custom model integration.


Testing ran on 
