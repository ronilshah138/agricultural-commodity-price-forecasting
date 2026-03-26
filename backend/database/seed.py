import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .db import SessionLocal, engine
from .models import Base, CommodityPrice
import os

def seed_data():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Check if data exists
    if db.query(CommodityPrice).first():
        print("Database already seeded.")
        db.close()
        return

    print("Seeding realistic mock data...")
    commodities = ["Rice", "Wheat", "Maize", "Onion", "Tomato"]
    states = ["Tamil Nadu", "Maharashtra", "Punjab", "UP", "Karnataka"]
    
    start_date = datetime.now() - timedelta(days=730)
    data = []

    for state in states:
        for commodity in commodities:
            base_price = np.random.randint(1500, 3000)
            for day in range(730):
                date = start_date + timedelta(days=day)
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day / 365)
                # Volatility varies by commodity
                noise_scale = 50
                if commodity in ["Tomato", "Onion"]:
                    noise_scale = 150 # Higher volatility
                elif commodity in ["Rice", "Wheat"]:
                    noise_scale = 20  # More stable
                    
                noise = np.random.normal(0, noise_scale)
                price = base_price * seasonal_factor + noise
                
                cp = CommodityPrice(
                    date=date,
                    commodity=commodity,
                    state=state,
                    district=f"{state} District",
                    min_price=price * 0.9,
                    max_price=price * 1.1,
                    modal_price=price
                )
                data.append(cp)

    db.add_all(data)
    db.commit()
    
    # Also save as CSV for ML pipeline
    df_list = []
    for item in data:
        df_list.append({
            "date": item.date,
            "commodity": item.commodity,
            "state": item.state,
            "district": item.district,
            "min_price": item.min_price,
            "max_price": item.max_price,
            "modal_price": item.modal_price
        })
    df = pd.DataFrame(df_list)
    os.makedirs("backend/data/raw", exist_ok=True)
    df.to_csv("backend/data/raw/commodity_prices.csv", index=False)
    
    print(f"Successfully seeded {len(data)} records and created CSV.")
    db.close()

if __name__ == "__main__":
    seed_data()
