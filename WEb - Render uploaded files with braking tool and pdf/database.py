from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Step 2 mein set kiya gaya URL yahan aayega
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# --- IMPORTANT FIX FOR RENDER ---
# Render ka URL "postgres://" deta hai, lekin SQLAlchemy ko "postgresql://" chahiye hota hai.
# Agar hum yeh replace nahi karenge to error aayega.
if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Agar URL nahi mila (Local testing ke liye), toh SQLite use karein
if not SQLALCHEMY_DATABASE_URL:
    SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# Database Connection Engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Session Maker (Isse hum database se baat karenge)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base Model (Isse hum Tables banayenge)
Base = declarative_base()

# Dependency (Har request ke baad DB close karne ke liye)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()