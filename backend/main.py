from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import commodities, predictions, metrics, data, model
from .database.seed import seed_data
from .ml.train import train_model
from .config import settings
import os

app = FastAPI(title=settings.PROJECT_NAME)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(commodities.router)
app.include_router(predictions.router)
app.include_router(metrics.router)
app.include_router(data.router)
app.include_router(model.router)

@app.on_event("startup")
def startup_event():
    print("Starting up AgroDash Backend...")
    # 1. Seed database if empty
    seed_data()
    
    # 2. Train model if doesn't exist
    if not os.path.exists(settings.MODEL_PATH):
        print("Model not found. Initializing training...")
        train_model()
    else:
        print("Model found. System ready.")

@app.get("/")
def read_root():
    return {"message": "Welcome to AgroDash API. Visit /docs for documentation."}
