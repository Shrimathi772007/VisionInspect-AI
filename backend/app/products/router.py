from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.models.product import Product
from app.models.user import User
from app.products.schemas import ProductCreate, ProductOut

router = APIRouter(prefix="/products", tags=["products"])


@router.post("", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(
    payload: ProductCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing_product = db.execute(
        select(Product).where(Product.product_code == payload.product_code)
    ).scalar_one_or_none()
    if existing_product is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Product code already exists")

    product = Product(product_name=payload.product_name, product_code=payload.product_code)
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


@router.get("", response_model=list[ProductOut])
def list_products(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.execute(select(Product).order_by(Product.id)).scalars().all()
