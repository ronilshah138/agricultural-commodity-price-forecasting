import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AgroDash Backend"
    DATABASE_URL: str = "sqlite:///./backend/database/agrodash.db"
    
    # Environment Modes
    CPU_MODE: bool = os.getenv("CPU_MODE", "true").lower() == "true"
    
    # ML Config
    MODEL_PATH: str = "backend/ml/model/xgboost_model.json"
    RAW_DATA_PATH: str = "backend/data/raw/commodity_prices.csv"
    PROCESSED_DATA_PATH: str = "backend/data/processed/cleaned_prices.csv"
    
    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:5173"]

settings = Settings()
