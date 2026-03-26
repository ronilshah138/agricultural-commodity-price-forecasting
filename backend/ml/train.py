import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os
import json
from ..config import settings
from ..database.db import SessionLocal
from ..database.models import ModelMetric
from .preprocess import preprocess_data

def train_model():
    print("Starting model training...")
    if not os.path.exists(settings.PROCESSED_DATA_PATH):
        preprocess_data()
        
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)
    
    # Features and target
    features = ['commodity_code', 'state_code', 'district_code', 'market_code', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']
    target = 'modal_price'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost Parameters
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'tree_method': 'hist', # Default to CPU hist
        'device': 'cpu'
    }
    
    if not settings.CPU_MODE:
        params['tree_method'] = 'gpu_hist'
        params['device'] = 'cuda'
        
    start_time = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate Global
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse**0.5
    r2 = max(0.0, r2_score(y_test, preds))
    
    # Save model
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
    model.save_model(settings.MODEL_PATH)
    
    # Save metrics to DB
    db = SessionLocal()
    
    # Extract dynamic mapping from the processed data
    commodity_map = dict(zip(df.commodity_code, df.commodity))
    
    # 1. Save Global Metric (commodity=None)
    global_metric = ModelMetric(
        commodity=None,
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2_score=r2,
        training_time=training_time,
        model_version="1.0.0"
    )
    db.add(global_metric)
    
    # 2. Save Per-Commodity Metrics
    for code, name in commodity_map.items():
        # Mask for this commodity in test set
        mask = X_test['commodity_code'] == code
        if mask.any():
            c_X_test = X_test[mask]
            c_y_test = y_test[mask]
            c_preds = model.predict(c_X_test)
            
            c_mae = mean_absolute_error(c_y_test, c_preds)
            c_mse = mean_squared_error(c_y_test, c_preds)
            c_r2 = max(0.0, r2_score(c_y_test, c_preds))
            
            cm = ModelMetric(
                commodity=name,
                mae=c_mae,
                mse=c_mse,
                rmse=c_mse**0.5,
                r2_score=c_r2,
                training_time=training_time,
                model_version="1.0.0"
            )
            db.add(cm)
            
    db.commit()
    db.close()
    
    print(f"Model trained in {training_time:.2f}s. Global R²: {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "training_time": training_time}

if __name__ == "__main__":
    train_model()
