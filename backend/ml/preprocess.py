import pandas as pd
from ..config import settings
import os

def preprocess_data():
    print("Preprocessing data...")
    if not os.path.exists(settings.RAW_DATA_PATH):
        print(f"Error: {settings.RAW_DATA_PATH} not found.")
        return

    df = pd.read_csv(settings.RAW_DATA_PATH)
    
    # Drop nulls and duplicates
    df = df.dropna().drop_duplicates()
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['commodity', 'state', 'district', 'market', 'date'])
    
    # Create lag features (Grouped by localized keys)
    group_keys = ['commodity', 'state', 'district', 'market']
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby(group_keys)['modal_price'].shift(lag)
        
    # Create rolling mean features
    for window in [3]:
        df[f'rolling_mean_{window}'] = df.groupby(group_keys)['modal_price'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        
    # Drop rows with NaN from lags/rolling
    df = df.dropna()
    
    # Encode categorical columns
    for col in ['commodity', 'state', 'district', 'market', 'variety', 'grade']:
        df[f'{col}_code'] = df[col].astype('category').cat.codes
    
    processed_dir = os.path.dirname(settings.PROCESSED_DATA_PATH)
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {settings.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()
