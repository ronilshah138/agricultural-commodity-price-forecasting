import pandas as pd
from datetime import datetime
from .db import SessionLocal, engine
from .models import Base, CommodityPrice
import os

def import_data():
    csv_path = "Price_Agriculture_commodities_Week.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Map column names to match the database schema
    print(f"Initial columns: {df.columns.tolist()}")
    mapping = {
        'State': 'state',
        'District': 'district',
        'Market': 'market',
        'Commodity': 'commodity',
        'Variety': 'variety',
        'Grade': 'grade',
        'Arrival_Date': 'date',
        'Min Price': 'min_price',  # New CSV uses spaces instead of underscores
        'Max Price': 'max_price',
        'Modal Price': 'modal_price'
    }
    df = df.rename(columns=mapping)

    # Parse dates explicitly (format in new CSV is DD-MM-YYYY)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Drop rows where date parsing failed
    df = df.dropna(subset=['date'])

    # Sync schema (Drop and recreate to ensure clean slate)
    print("Recreating database tables...")
    CommodityPrice.__table__.drop(bind=engine, checkfirst=True)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Drop Commodity_Code if it exists (it doesn't in the new CSV, but safe to keep)
    if 'Commodity_Code' in df.columns:
        df = df.drop(columns=['Commodity_Code'])

    print(f"Importing {len(df)} records into the database...")
    
    # Batch insert for performance
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        records = [
            CommodityPrice(**row.to_dict())
            for _, row in batch.iterrows()
        ]
        db.add_all(records)
        db.commit()
        print(f"Processed {min(i + batch_size, len(df))}/{len(df)} records")

    print("Import complete.")
    db.close()
    
    # Save a copy to raw data path for ML pipeline
    raw_path = "backend/data/raw/commodity_prices.csv"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Sync complete. Raw data saved to {raw_path}")

if __name__ == "__main__":
    import_data()
