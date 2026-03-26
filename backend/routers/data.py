from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.models import CommodityPrice

router = APIRouter(prefix="/data", tags=["Data"])

@router.get("/")
def get_data_records(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    commodity: str = Query(None),
    state: str = Query(None),
    date: str = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(CommodityPrice)
    
    if commodity:
        query = query.filter(CommodityPrice.commodity == commodity)
    if state:
        query = query.filter(CommodityPrice.state == state)
    if date:
        query = query.filter(CommodityPrice.date == date)
        
    total = query.count()
    records = query.order_by(CommodityPrice.date.desc()).offset((page - 1) * limit).limit(limit).all()
    
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "records": [{
            "id": r.id,
            "date": r.date.strftime("%Y-%m-%d"),
            "commodity": r.commodity,
            "state": r.state,
            "district": r.district,
            "min_price": r.min_price,
            "max_price": r.max_price,
            "modal_price": r.modal_price
        } for r in records]
    }
