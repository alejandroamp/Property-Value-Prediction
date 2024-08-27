# train_xgboost.py

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from azureml.core import Run

# Start an Azure ML run
run = Run.get_context()

# Load the data
X_train_scaled = np.load('datasets/X_train_scaled.npy')
y_train = np.load('datasets/y_train.npy')
X_test_scaled = np.load('datasets/X_test_scaled.npy')
y_test = np.load('datasets/y_test.npy')

# Create and configure the XGBoost model
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.1,
    max_depth=3,
    n_estimators=200
)

# Train the model
xgb_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = xgb_reg.predict(X_train_scaled)
y_pred_test = xgb_reg.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# Log metrics to Azure ML
run.log('Training MSE', train_mse)
run.log('Training R^2', train_r2)
run.log('Test MSE', test_mse)
run.log('Test R^2', test_r2)

# Save the model
import joblib
model_path = 'outputs/xgboost_model_azure.pkl'
joblib.dump(xgb_reg, model_path)

# Upload the model to Azure ML
run.upload_file('outputs/xgboost_model_azure.pkl', model_path)

print("Model training complete and model saved.")
