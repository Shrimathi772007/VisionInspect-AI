from app.database.base import Base

from .defect import Defect
from .inspection import Inspection
from .product import Product
from .user import User, UserRole

__all__ = ["Base", "User", "UserRole", "Product", "Inspection", "Defect"]
