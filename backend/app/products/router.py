from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user, require_role
from app.database import get_db
from app.models.inspection import Inspection
from app.models.product import Product
from app.models.user import User, UserRole
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


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(
    product_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    has_inspections = db.execute(
        select(Inspection.id).where(Inspection.product_id == product_id).limit(1)
    ).scalar_one_or_none()
    if has_inspections is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete product while inspections are associated with it",
        )

    db.delete(product)
    db.commit()
