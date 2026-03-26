from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.models import Prediction
from ..ml.predict import make_predictions
from datetime import datetime

router = APIRouter(prefix="/predict", tags=["Predictions"])

class PredictionRequest(BaseModel):
    commodity: str
    region: str
    district: str = None
    market: str = None
    start_date: str
    end_date: str

@router.post("/")
def predict_commodity(request: PredictionRequest, db: Session = Depends(get_db)):
    try:
        results = make_predictions(
            request.commodity, 
            request.region, 
            district=request.district,
            market=request.market,
            start_date=request.start_date, 
            end_date=request.end_date
        )
        
        # Save to DB
        prediction_records = [
            Prediction(
                commodity=request.commodity,
                state=request.region,
                district=request.district,
                market=request.market,
                date=datetime.strptime(r["date"], "%Y-%m-%d"),
                predicted_price=r["predicted_price"],
                model_version="1.0.0"
            ) for r in results
        ]
        db.add_all(prediction_records)
        db.commit()
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
    history = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(100).all()
    return [{
        "id": h.id,
        "commodity": h.commodity,
        "state": h.state,
        "date": h.date.strftime("%Y-%m-%d"),
        "predicted_price": h.predicted_price,
        "created_at": h.created_at
    } for h in history]
