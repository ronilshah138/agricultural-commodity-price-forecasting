from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.models import ModelMetric
from ..ml.train import train_model
import os
from ..config import settings

router = APIRouter(prefix="/model", tags=["Model"])

@router.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model)
    return {"status": "Training started in background"}

@router.get("/status")
def get_model_status(db: Session = Depends(get_db)):
    last_metric = db.query(ModelMetric).order_by(ModelMetric.created_at.desc()).first()
    exists = os.path.exists(settings.MODEL_PATH)
    return {
        "model_exists": exists,
        "last_trained": last_metric.created_at if last_metric else None,
        "status": "ready" if exists else "needs_training"
    }
