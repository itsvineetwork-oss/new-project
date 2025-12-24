from sqlalchemy import Column, Integer, String, Boolean
from database import Base

# User Table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_license_active = Column(Boolean, default=False) # Default False rahega jab tak key na dale

# License Keys Table (Jo aap generate karke DB mein dalenge)
class LicenseKey(Base):
    __tablename__ = "license_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)  # Jaise "PRO-2025-ABC"
    is_used = Column(Boolean, default=False)       # Use hone ke baad True ho jayega