from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db

app = FastAPI(title="VisionInspect AI Backend")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "VisionInspect AI backend is running"}


@app.get("/health/db")
def health_check_db(db: Session = Depends(get_db)):
    try:
        db.execute(select(1))
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database connection failed")
    return {"status": "ok", "database": "connected"}
