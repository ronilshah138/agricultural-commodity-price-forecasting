from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.models import CommodityPrice
from datetime import datetime

router = APIRouter(prefix="/commodities", tags=["Commodities"])

from sqlalchemy import func

@router.get("/")
def get_commodities(db: Session = Depends(get_db)):
    query = db.query(CommodityPrice.commodity).group_by(CommodityPrice.commodity).having(func.count(CommodityPrice.id) >= 3).all()
    return [c[0] for c in query]

@router.get("/states")
def get_states(commodity: str = Query(None), db: Session = Depends(get_db)):
    query = db.query(CommodityPrice.state)
    if commodity:
        query = query.filter(CommodityPrice.commodity == commodity)
    query = query.group_by(CommodityPrice.state).having(func.count(CommodityPrice.id) >= 3)
    states = query.all()
    return [s[0] for s in states]

@router.get("/districts")
def get_districts(commodity: str = Query(None), state: str = Query(None), db: Session = Depends(get_db)):
    query = db.query(CommodityPrice.district)
    if commodity:
        query = query.filter(CommodityPrice.commodity == commodity)
    if state:
        query = query.filter(CommodityPrice.state == state)
    query = query.group_by(CommodityPrice.district).having(func.count(CommodityPrice.id) >= 3)
    districts = query.all()
    return [d[0] for d in districts]

@router.get("/markets")
def get_markets(commodity: str = Query(None), state: str = Query(None), district: str = Query(None), db: Session = Depends(get_db)):
    query = db.query(CommodityPrice.market)
    if commodity:
        query = query.filter(CommodityPrice.commodity == commodity)
    if state:
        query = query.filter(CommodityPrice.state == state)
    if district:
        query = query.filter(CommodityPrice.district == district)
    query = query.group_by(CommodityPrice.market).having(func.count(CommodityPrice.id) >= 3)
    markets = query.all()
    return [m[0] for m in markets]

@router.get("/{name}/prices")
def get_commodity_prices(
    name: str,
    region: str = Query(None),
    district: str = Query(None),
    market: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(CommodityPrice).filter(CommodityPrice.commodity == name)
    
    if region:
        query = query.filter(CommodityPrice.state == region)
    if district:
        query = query.filter(CommodityPrice.district == district)
    if market:
        query = query.filter(CommodityPrice.market == market)
    
    if start_date:
        query = query.filter(CommodityPrice.date >= datetime.strptime(start_date, "%Y-%m-%d"))
    
    if end_date:
        query = query.filter(CommodityPrice.date <= datetime.strptime(end_date, "%Y-%m-%d"))
        
    prices = query.order_by(CommodityPrice.date.asc()).all()
    return [{
        "date": p.date.strftime("%Y-%m-%d"),
        "min_price": p.min_price,
        "max_price": p.max_price,
        "modal_price": p.modal_price
    } for p in prices]
