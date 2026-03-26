from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.models import ModelMetric

router = APIRouter(prefix="/metrics", tags=["Metrics"])

@router.get("/")
def get_latest_metrics(commodity: str = None, db: Session = Depends(get_db)):
    if commodity == "Global" or commodity == "":
        commodity = None
        
    metric = db.query(ModelMetric).filter(
        ModelMetric.commodity == commodity
    ).order_by(ModelMetric.id.desc()).first() # Use ID to ensure latest
    
    if not metric:
        return {"mae": 0, "mse": 0, "rmse": 0, "r2_score": 0, "commodity": commodity or "Global"}
        
    return {
        "mae": round(metric.mae, 2),
        "mse": round(metric.mse, 2),
        "rmse": round(metric.rmse, 2),
        "r2_score": max(0.0, round(metric.r2_score, 4)),
        "training_time": round(metric.training_time, 2),
        "commodity": metric.commodity or "Global"
    }

@router.get("/comparison")
def get_comparison(db: Session = Depends(get_db)):
    # In a real app, this would query multiple versions; here we return mock comparison table data
    return [
        {"method": "ARIMA", "mae": 68.4, "rmse": 72.1, "r2": 0.88, "training_time": "0.2s"},
        {"method": "XGBoost (Current)", "mae": 42.5, "rmse": 46.3, "r2": 0.94, "training_time": "1.5s"}
    ]
