
# Car Failure Prediction Model

## Overview
This project predicts car failure types using tabular sensor and usage data. The pipeline includes data ingestion, feature engineering, preprocessing, model training, and prediction. 
The model takes various car attributes as input and predicts the most likely failure type, supporting maintenance and reliability analysis.

## Dependencies
This module relies on the following:
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `seaborn`, `matplotlib`
- **Machine Learning:** `scikit-learn`, `catboost`, `xgboost`
- **Serialization:** `dill`, `pickle`
- **Custom Modules:** `src.exception`, `src.logger`, `src.utils`
- **File Handling:** `os`, `sys`
- **Logging:** `logging`

## Data Ingestion
- **Raw Data Loading:** Reads raw car failure data from `notebook/data/Failure.csv` using pandas.
- **Data Splitting:** Splits the dataset into training and testing sets to ensure robust model evaluation.
- **Saving Processed Data:** Stores the split datasets as CSV files in the `artifacts` directory:
  - `artifacts/train.csv`
  - `artifacts/test.csv`
  - `artifacts/data.csv`
- **Error Handling & Logging:** Utilizes custom exception and logging modules to track and handle issues during ingestion.

## Data Transformation
- **Feature Engineering:**
  - Extracts year from the car model string.
  - Calculates car age from model year.
  - Encodes failure types as categorical variables.
  - Maps usage levels and processes temperature readings.
- **Outlier Removal:** Detects and removes outliers from numeric features to improve model robustness.
- **Missing Value Imputation:** Fills missing values using median for numeric features and most frequent value for categorical features.
- **Scaling & Encoding:**
  - Applies standard scaling to numeric features.
  - Uses one-hot encoding for categorical variables.
- **Saving Transformers:** Stores the fitted preprocessor as `artifact/preprocessor.pkl` for consistent transformation during inference.

## Model Trainer
1. **Models Evaluated:**
   - K-Neighbors Classifier
   - XGBoost Classifier
   - CatBoost Classifier
   - AdaBoost Classifier
   - Gradient Boosting Classifier
   - Random Forest Classifier
   - Decision Tree Classifier
2. **Hyperparameter Tuning:** Uses grid search for each model.
3. **Model Selection:** Selects the best model based on test accuracy.
4. **Model Saving:** Best model is saved as `artifact/model.pkl`.

## Prediction Pipeline
- Loads the trained model and preprocessor.
- Accepts user input via a Streamlit web interface (`app.py`).
- Transforms input and predicts the failure type.

## Project Structure
- `src`: Core modules for data ingestion, transformation, model training, and utilities.
- `artifact`: Stores trained models and preprocessors.
- `artifacts`: Stores processed datasets.
- `logs`: Logging output for debugging and monitoring.
- `notebook`: EDA and data exploration notebooks.

---

For more details, see the code in `src` and the EDA in `notebook/data/EDA.ipynb`.
