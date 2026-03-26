import pandas as pd
import xgboost as xgb
from ..config import settings
import os
from datetime import datetime, timedelta

# Global model instance for efficiency in a real app, but here we load per prediction for simplicity
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(settings.MODEL_PATH):
            return None
        _model = xgb.XGBRegressor()
        _model.load_model(settings.MODEL_PATH)
    return _model

def make_predictions(commodity, region, district=None, market=None, start_date=None, end_date=None):
    model = load_model()
    if model is None:
        return []

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Load processed data to get category codes
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)
    try:
        comm_code = df[df['commodity'] == commodity]['commodity_code'].iloc[0]
        state_code = df[df['state'] == region]['state_code'].iloc[0]
        
        # Localized codes
        district_code = 0
        market_code = 0
        
        if district and 'district' in df.columns:
            matches = df[df['district'] == district]
            if not matches.empty:
                district_code = matches['district_code'].iloc[0]
                
        if market and 'market' in df.columns:
            matches = df[df['market'] == market]
            if not matches.empty:
                market_code = matches['market_code'].iloc[0]

        # Find the last known row for this specific locality to seed lags
        query = (df['commodity'] == commodity) & (df['state'] == region)
        if district:
            query &= (df['district'] == district)
        
        loc_df = df[query]
        if not loc_df.empty:
            last_known_row = loc_df.iloc[-1]
        else:
            # Fallback to state-level if district-level history is missing in the sample
            fallback_df = df[(df['commodity'] == commodity) & (df['state'] == region)]
            if not fallback_df.empty:
                last_known_row = fallback_df.iloc[-1]
            else:
                # If no data at all for this state, fallback to global commodity data
                global_df = df[df['commodity'] == commodity]
                if not global_df.empty:
                    last_known_row = global_df.iloc[-1]
                else:
                    return [] # Cannot predict without any historical lag seed
            
    except Exception as e:
        print(f"Prediction Error: {e}")
        return []

    # Seed lag features
    curr_lags = {
        'lag_1': last_known_row['modal_price'],
        'lag_2': last_known_row['lag_1'],
        'lag_3': last_known_row['lag_2'],
        'rolling_mean_3': last_known_row['rolling_mean_3']
    }

    preds = []
    current_date = start_dt
    while current_date <= end_dt:
        # Construct feature vector matching train.py order:
        # ['commodity_code', 'state_code', 'district_code', 'market_code', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']
        features = pd.DataFrame([{
            'commodity_code': comm_code,
            'state_code': state_code,
            'district_code': district_code,
            'market_code': market_code,
            **curr_lags
        }])
        
        pred_price = float(model.predict(features)[0])
        
        preds.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted_price": round(pred_price, 2)
        })
        
        # Update lags for next step (very basic iterative prediction)
        curr_lags['lag_3'] = curr_lags['lag_2']
        curr_lags['lag_2'] = curr_lags['lag_1']
        curr_lags['lag_1'] = pred_price
        # Note: rolling means would be updated properly in a production system
        
        current_date += timedelta(days=1)
        
    return preds
