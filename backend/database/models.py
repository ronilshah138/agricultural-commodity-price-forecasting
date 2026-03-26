from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from .db import Base

class CommodityPrice(Base):
    __tablename__ = "commodity_prices"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, index=True)
    commodity = Column(String, index=True)
    variety = Column(String, index=True, nullable=True)
    grade = Column(String, index=True, nullable=True)
    market = Column(String, index=True, nullable=True)
    state = Column(String, index=True)
    district = Column(String, index=True)
    min_price = Column(Float)
    max_price = Column(Float)
    modal_price = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    commodity = Column(String, index=True)
    variety = Column(String, index=True, nullable=True)
    grade = Column(String, index=True, nullable=True)
    market = Column(String, index=True, nullable=True)
    state = Column(String, index=True)
    district = Column(String, index=True, nullable=True)
    date = Column(DateTime)
    predicted_price = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    commodity = Column(String, index=True, nullable=True) # Overall if NULL
    mae = Column(Float)
    mse = Column(Float)
    rmse = Column(Float)
    r2_score = Column(Float)
    training_time = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
